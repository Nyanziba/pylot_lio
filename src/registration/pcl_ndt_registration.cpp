// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#include "pylot_lio/registration/pcl_ndt_registration.hpp"

#include <cmath>
#include <sstream>

#include <pcl/registration/ndt.h>

#include "pylot_lio/map/i_point_cloud_map.hpp"

namespace pylot_lio
{

namespace
{

// pylot_lio の PointCloud (= pcl::PointCloud<pcl::PointXYZI>) から非有限点 (NaN/Inf) を
// 除外したコピーを作る。 NDT の内部で voxel grid 構築時に NaN を含むと未定義動作に
// なるため、 念のため呼び出し側で弾いておく。
PointCloudPtr removeNonFinitePoints(const PointCloud & cloud_in)
{
  PointCloudPtr finite_cloud(new PointCloud());
  finite_cloud->points.reserve(cloud_in.points.size());
  for (const Point & input_point : cloud_in.points) {
    if (!std::isfinite(input_point.x) ||
        !std::isfinite(input_point.y) ||
        !std::isfinite(input_point.z))
    {
      continue;
    }
    finite_cloud->points.push_back(input_point);
  }
  finite_cloud->width = static_cast<std::uint32_t>(finite_cloud->points.size());
  finite_cloud->height = 1;
  finite_cloud->is_dense = true;
  return finite_cloud;
}

}  // namespace

PclNdtRegistration::PclNdtRegistration(const Config & config)
: config_(config)
{
}

IRegistration::AlignResult PclNdtRegistration::align(
  const PointCloud & source_cloud_body,
  const IPointCloudMap & map_world,
  const Eigen::Isometry3d & initial_transform_world_body)
{
  AlignResult result;
  result.transform_world_body = initial_transform_world_body;

  const PointCloudPtr source_finite = removeNonFinitePoints(source_cloud_body);
  PointCloudConstPtr map_cloud = map_world.toPointCloud();
  const PointCloudPtr target_finite = removeNonFinitePoints(*map_cloud);

  const std::size_t min_required =
    static_cast<std::size_t>(std::max(1, config_.min_points_for_alignment));
  if (source_finite->size() < min_required || target_finite->size() < min_required) {
    return result;
  }

  pcl::NormalDistributionsTransform<Point, Point> ndt;
  ndt.setResolution(static_cast<float>(config_.resolution_m));
  ndt.setStepSize(config_.step_size_m);
  ndt.setTransformationEpsilon(config_.transformation_epsilon_m);
  ndt.setMaximumIterations(config_.max_iterations);

  ndt.setInputTarget(target_finite);
  ndt.setInputSource(source_finite);

  PointCloud aligned_cloud;
  const Eigen::Matrix4f initial_guess =
    initial_transform_world_body.matrix().cast<float>();
  ndt.align(aligned_cloud, initial_guess);

  Eigen::Isometry3d estimated_transform = Eigen::Isometry3d::Identity();
  estimated_transform.matrix() = ndt.getFinalTransformation().cast<double>();

  result.transform_world_body = estimated_transform;
  result.converged = ndt.hasConverged();
  result.iterations = static_cast<int>(ndt.getFinalNumIteration());
  // PCL NDT は対応点数を直接公開していないため、 source 点数を上限値として記録する。
  // (NDT は対応点ベースの手法ではなく、 全 source 点が尤度に寄与する。)
  result.num_correspondences = static_cast<int>(source_finite->size());
  // 最終コストはトランスフォーム尤度 (大きいほどよい) を符号反転して保存。
  // pylot_lio の AlignResult::final_cost は「小さいほど良い」 規約。
  result.final_cost = -static_cast<double>(ndt.getTransformationLikelihood());
  return result;
}

std::string PclNdtRegistration::describe() const
{
  std::ostringstream summary_stream;
  summary_stream << "pcl_ndt"
                 << ":resolution=" << config_.resolution_m
                 << ",step_size=" << config_.step_size_m
                 << ",max_iterations=" << config_.max_iterations;
  return summary_stream.str();
}

}  // namespace pylot_lio
