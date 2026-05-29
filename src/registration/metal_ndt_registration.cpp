// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#include "pylot_lio/registration/metal_ndt_registration.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <sstream>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Geometry>
#include <Eigen/LU>

#include "metal_gpu_kernels/metal_gaussian_voxelmap_builder.hpp"
#include "metal_gpu_kernels/metal_ndt_linearizer.hpp"
#include "pylot_lio/lie_algebra.hpp"
#include "pylot_lio/map/i_point_cloud_map.hpp"

namespace pylot_lio
{

namespace
{

std::vector<Eigen::Vector3d> toFinitePointVector(const PointCloud & cloud_in)
{
  std::vector<Eigen::Vector3d> finite_points;
  finite_points.reserve(cloud_in.points.size());
  for (const Point & input_point : cloud_in.points) {
    if (!std::isfinite(input_point.x) ||
        !std::isfinite(input_point.y) ||
        !std::isfinite(input_point.z))
    {
      continue;
    }
    finite_points.emplace_back(input_point.x, input_point.y, input_point.z);
  }
  return finite_points;
}

}  // namespace

MetalNdtRegistration::MetalNdtRegistration(const Config & config)
: config_(config),
  engine_(std::make_shared<metal_gpu_kernels::MetalNdtEngine>())
{
}

bool MetalNdtRegistration::isAvailable()
{
  metal_gpu_kernels::MetalNdtEngine probe_engine;
  return probe_engine.isValid();
}

IRegistration::AlignResult MetalNdtRegistration::align(
  const PointCloud & source_cloud_body,
  const IPointCloudMap & map_world,
  const Eigen::Isometry3d & initial_transform_world_body)
{
  AlignResult result;
  result.transform_world_body = initial_transform_world_body;

  const std::vector<Eigen::Vector3d> source_points_body =
    toFinitePointVector(source_cloud_body);
  const std::vector<Eigen::Vector3d> target_points_world =
    toFinitePointVector(*map_world.toPointCloud());

  const std::size_t min_required =
    static_cast<std::size_t>(std::max(1, config_.min_points_for_alignment));
  if (source_points_body.size() < min_required ||
      target_points_world.size() < min_required)
  {
    return result;
  }

  // target をボクセル化してガウシアン化する (VGICP の VoxelMapBuilder を再利用)。
  metal_gpu_kernels::VoxelMapBuildConfig build_config;
  build_config.voxel_size_m = static_cast<float>(config_.resolution_m);
  build_config.min_points_per_voxel = config_.min_points_per_voxel;
  // 注意: ガウシアン構築の eigen_floor は NDT 用に十分大きく取る (1e-3 程度)。
  // covariance_det_floor は別物 (Metal shader 側で逆行列計算時の det 下限)。
  build_config.eigen_floor = static_cast<float>(config_.gaussian_eigenvalue_floor);
  const metal_gpu_kernels::VgicpVoxelTable voxel_table =
    metal_gpu_kernels::buildVoxelMapCpu(target_points_world, build_config);
  if (voxel_table.capacity <= 0) {
    return result;
  }

  metal_gpu_kernels::NdtLinearizeConfig linearize_config;
  linearize_config.search_radius_voxels = config_.search_radius_voxels;
  linearize_config.max_correspondence_distance_m =
    static_cast<float>(config_.max_correspondence_distance_m);
  linearize_config.covariance_det_floor =
    static_cast<float>(config_.covariance_det_floor);

  // GPU を使うかの判定。 source 点数が小さければ CPU 参照に切り替え (起動オーバヘッド回避)。
  const bool use_gpu_path =
    engine_ && engine_->isValid() &&
    (config_.gpu_min_points <= 0 ||
     static_cast<int>(source_points_body.size()) >= config_.gpu_min_points);

  if (use_gpu_path) {
    engine_->setTarget(voxel_table);
    engine_->setSource(source_points_body);
  }

  Eigen::Isometry3d current_transform = initial_transform_world_body;
  double previous_cost = std::numeric_limits<double>::max();
  int iteration_count = 0;
  bool converged_flag = false;

  for (iteration_count = 0; iteration_count < config_.max_iterations; ++iteration_count) {
    metal_gpu_kernels::NdtLinearization linearization;
    if (use_gpu_path) {
      linearization = engine_->linearize(current_transform, linearize_config);
    } else {
      linearization = metal_gpu_kernels::linearizeNdtCpu(
        source_points_body, voxel_table, current_transform, linearize_config);
    }

    if (linearization.valid_correspondences < config_.min_points_for_alignment) {
      break;
    }

    // Newton step: H * delta = -g
    Eigen::Matrix<double, 6, 1> delta_xi;
    Eigen::LDLT<Eigen::Matrix<double, 6, 6>> ldlt_solver(linearization.hessian);
    if (ldlt_solver.info() == Eigen::Success) {
      delta_xi = ldlt_solver.solve(-linearization.gradient);
    } else {
      delta_xi =
        linearization.hessian.fullPivLu().solve(-linearization.gradient);
    }
    if (!delta_xi.allFinite()) {
      break;
    }

    const Eigen::Matrix<double, 6, 1> applied_delta = config_.step_size * delta_xi;
    Eigen::Isometry3d updated_transform = current_transform;
    // 右摂動更新 (Jacobian と整合): R_new = R_old * exp(δω)、 t_new = t_old + δt
    updated_transform.translation() += applied_delta.tail<3>();
    updated_transform.linear() =
      current_transform.linear() * lie::expSO3(applied_delta.head<3>());

    const double translation_step_norm = applied_delta.tail<3>().norm();
    const double rotation_step_norm = applied_delta.head<3>().norm();
    current_transform = updated_transform;
    previous_cost = linearization.cost;

    if (translation_step_norm < config_.convergence_translation_m &&
        rotation_step_norm < config_.convergence_rotation_rad)
    {
      converged_flag = true;
      ++iteration_count;
      break;
    }
  }

  result.transform_world_body = current_transform;
  result.iterations = iteration_count;
  result.converged = converged_flag;
  result.final_cost = previous_cost;
  result.num_correspondences = static_cast<int>(source_points_body.size());
  return result;
}

std::string MetalNdtRegistration::describe() const
{
  std::ostringstream summary_stream;
  summary_stream << "metal_ndt"
                 << ":resolution=" << config_.resolution_m
                 << ",max_iterations=" << config_.max_iterations
                 << ",gpu=" << (isAvailable() ? "available" : "unavailable");
  return summary_stream.str();
}

}  // namespace pylot_lio
