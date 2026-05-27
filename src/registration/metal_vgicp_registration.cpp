// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#include "pylot_lio/registration/metal_vgicp_registration.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "pylot_lio/gpu/metal_vgicp_linearizer.hpp"
#include "pylot_lio/lie_algebra.hpp"
#include "pylot_lio/map/voxel_map.hpp"

namespace pylot_lio
{

namespace
{

// source 点群の各点について body フレームの局所共分散 C_s を推定する。
// plain_gicp の computeSourceCovariances と同方式 (近傍 k 点の共分散を平面正則化)。
// NOTE: plain_gicp と重複している。 将来 gicp_source_covariance として共通化候補。
std::vector<Eigen::Matrix3d> computeSourceCovariances(
  const PointCloud & source_cloud_body, int num_neighbors, double plane_epsilon)
{
  const int num_points = static_cast<int>(source_cloud_body.points.size());
  std::vector<Eigen::Matrix3d> covariances(
    num_points, Eigen::Matrix3d::Identity() * plane_epsilon);
  if (num_points < 3 || num_neighbors < 3) {
    return covariances;
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>());
  pcl_cloud->points.reserve(num_points);
  std::vector<int> original_index_of;
  original_index_of.reserve(num_points);
  for (int i = 0; i < num_points; ++i) {
    const Point & p = source_cloud_body.points[i];
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
      continue;
    }
    pcl_cloud->points.emplace_back(p.x, p.y, p.z);
    original_index_of.push_back(i);
  }
  pcl_cloud->width = static_cast<std::uint32_t>(pcl_cloud->points.size());
  pcl_cloud->height = 1;
  if (pcl_cloud->points.size() < 3) {
    return covariances;
  }

  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
  kdtree.setInputCloud(pcl_cloud);
  const int k = std::min(num_neighbors, static_cast<int>(pcl_cloud->points.size()));

  std::vector<int> neighbor_indices(k);
  std::vector<float> neighbor_sq_dists(k);
  for (std::size_t query = 0; query < pcl_cloud->points.size(); ++query) {
    if (kdtree.nearestKSearch(
        pcl_cloud->points[query], k, neighbor_indices, neighbor_sq_dists) < 3)
    {
      continue;
    }
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (int n = 0; n < k; ++n) {
      const auto & pt = pcl_cloud->points[neighbor_indices[n]];
      mean += Eigen::Vector3d(pt.x, pt.y, pt.z);
    }
    mean /= static_cast<double>(k);
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (int n = 0; n < k; ++n) {
      const auto & pt = pcl_cloud->points[neighbor_indices[n]];
      const Eigen::Vector3d diff = Eigen::Vector3d(pt.x, pt.y, pt.z) - mean;
      cov += diff * diff.transpose();
    }
    cov /= static_cast<double>(k);

    // 平面正則化 (Segal 2009): 固有値を (epsilon, 1, 1) に置換。
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
    Eigen::Matrix3d eigenvectors = solver.eigenvectors();
    Eigen::Vector3d replaced(plane_epsilon, 1.0, 1.0);  // 昇順固有値: 最小=法線方向
    Eigen::Matrix3d regularized =
      eigenvectors * replaced.asDiagonal() * eigenvectors.transpose();
    covariances[original_index_of[query]] = regularized;
  }
  return covariances;
}

// VoxelMap のボクセルガウス分布を VgicpVoxelTable に変換する。
// 共分散は VoxelMap が findNearestNeighbor で行うのと同じ固有値 floor を適用して
// 退化 (1 サンプルセル等) を持ち上げる。
gpu::VgicpVoxelTable extractVoxelTable(const VoxelMap & voxel_map)
{
  const auto & cells = voxel_map.cells();
  const double voxel_size = voxel_map.config().voxel_size_m;
  const double eigen_floor = voxel_map.config().covariance_eigen_floor;
  const int min_points = voxel_map.config().min_points_per_cell_for_covariance;

  std::vector<Eigen::Vector3i> coords;
  std::vector<Eigen::Vector3d> means;
  std::vector<Eigen::Matrix3d> covs;
  coords.reserve(cells.size());
  means.reserve(cells.size());
  covs.reserve(cells.size());

  for (const auto & [key, cell] : cells) {
    if (cell.sample_count < min_points) {
      continue;
    }
    coords.emplace_back(
      static_cast<int>(key.x), static_cast<int>(key.y), static_cast<int>(key.z));
    means.push_back(cell.mean);

    // 固有値 floor (VoxelMap::findNearestNeighbor と同処方)。
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cell.covariance);
    Eigen::Vector3d eigenvalues = solver.eigenvalues();
    for (int axis = 0; axis < 3; ++axis) {
      if (eigenvalues(axis) < eigen_floor) {
        eigenvalues(axis) = eigen_floor;
      }
    }
    covs.push_back(
      solver.eigenvectors() * eigenvalues.asDiagonal() * solver.eigenvectors().transpose());
  }

  return gpu::VgicpVoxelTable::build(
    static_cast<float>(voxel_size), coords, means, covs);
}

}  // namespace

MetalVgicpRegistration::MetalVgicpRegistration(const Config & config)
: config_(config)
{
}

bool MetalVgicpRegistration::isAvailable()
{
#ifdef PYLOT_LIO_HAS_METAL
  return true;
#else
  return false;
#endif
}

MetalVgicpRegistration::AlignResult MetalVgicpRegistration::align(
  const PointCloud & source_cloud_body,
  const IPointCloudMap & map_world,
  const Eigen::Isometry3d & initial_transform_world_body)
{
  AlignResult result;
  result.transform_world_body = initial_transform_world_body;

  // target マップは VoxelMap のみ対応 (Step2 制約)。
  const VoxelMap * voxel_map = dynamic_cast<const VoxelMap *>(&map_world);
  if (voxel_map == nullptr) {
    result.converged = false;
    return result;
  }

  // source 点を Vector3d 配列に変換 (NaN 点も含めて並びを保つ。 linearizer 側で除外)。
  std::vector<Eigen::Vector3d> source_points;
  source_points.reserve(source_cloud_body.points.size());
  for (const Point & p : source_cloud_body.points) {
    source_points.emplace_back(p.x, p.y, p.z);
  }
  const std::vector<Eigen::Matrix3d> source_covariances = computeSourceCovariances(
    source_cloud_body, config_.source_covariance_num_neighbors,
    config_.source_covariance_plane_epsilon);

  const gpu::VgicpVoxelTable voxel_table = extractVoxelTable(*voxel_map);
  if (voxel_table.capacity <= 0) {
    result.converged = false;
    return result;
  }

  gpu::VgicpLinearizeConfig lin_config;
  lin_config.huber_threshold = static_cast<float>(config_.huber_threshold);
  lin_config.max_correspondence_distance_m =
    static_cast<float>(config_.max_correspondence_distance_m);
  lin_config.search_radius_voxels = config_.search_radius_voxels;

  Eigen::Isometry3d current = initial_transform_world_body;
  for (int iteration = 0; iteration < config_.max_iterations; ++iteration) {
    const gpu::VgicpLinearization lin = gpu::linearizeVgicpMetal(
      source_points, source_covariances, voxel_table, current, lin_config);

    if (!lin.gpu_used) {
      // Metal が使えなかった (非対応ビルド / デバイス無)。 factory がフォールバック
      // するので、 ここでは未収束として返す。
      result.converged = false;
      result.transform_world_body = current;
      return result;
    }
    if (lin.valid_correspondences < 6) {
      break;  // 解が一意に決まらない。
    }

    // δ = H⁻¹ gradient (gradient = -Σ w JᵀΩd 規約)。 δ=[ω; ρ]。
    const Eigen::Matrix<double, 6, 1> delta = lin.hessian.ldlt().solve(lin.gradient);
    const Eigen::Vector3d delta_rotation = delta.head<3>();
    const Eigen::Vector3d delta_translation = delta.tail<3>();

    Eigen::Isometry3d delta_transform = Eigen::Isometry3d::Identity();
    delta_transform.linear() = lie::expSO3(delta_rotation);
    delta_transform.translation() = delta_translation;
    current = delta_transform * current;  // 左摂動
    current.linear() = lie::normalizeRotation(current.linear());

    result.iterations = iteration + 1;
    result.final_cost = lin.cost;
    result.num_correspondences = lin.valid_correspondences;

    if (delta_translation.norm() < config_.convergence_translation_m &&
        delta_rotation.norm() < config_.convergence_rotation_rad)
    {
      result.converged = true;
      break;
    }
  }

  result.transform_world_body = current;
  return result;
}

std::string MetalVgicpRegistration::describe() const
{
  std::ostringstream oss;
  oss << "metal_vgicp:max_iter=" << config_.max_iterations
      << ",max_corr=" << config_.max_correspondence_distance_m
      << ",search_radius=" << config_.search_radius_voxels;
  return oss.str();
}

}  // namespace pylot_lio
