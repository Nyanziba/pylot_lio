// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#include "pylot_lio/map/kd_tree_map.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>
#include <unordered_set>

#include <Eigen/Eigenvalues>

namespace pylot_lio
{

KdTreeMap::KdTreeMap(const Config & config)
: config_(config),
  accumulated_cloud_(std::make_shared<PointCloud>()),
  kd_tree_(std::make_unique<pcl::KdTreeFLANN<Point>>()),
  inserts_since_rebuild_(0)
{
}

void KdTreeMap::rebuildKdTree()
{
  if (accumulated_cloud_->empty()) {
    return;
  }
  kd_tree_->setInputCloud(accumulated_cloud_);
  inserts_since_rebuild_ = 0;
}

void KdTreeMap::insertScan(
  const PointCloud & scan_in_world,
  const Eigen::Isometry3d & /*pose_world_body*/)
{
  // 同一ボクセル内の点は 1 点に間引いてから蓄積する (PCL KdTree 肥大化防止)。
  // 「既に同じボクセルに代表点が居れば追加しない」だけの簡易版。
  // 既存累積点もボクセルを共有しうるので、scan ごとにスキャン側だけで重複判定する。
  const double inverse_voxel_size = 1.0 / config_.subsample_voxel_size_m;
  std::unordered_set<int64_t> seen_voxel_keys_this_scan;
  seen_voxel_keys_this_scan.reserve(scan_in_world.points.size() / 4 + 16);
  for (const Point & input_point : scan_in_world.points) {
    if (!std::isfinite(input_point.x) ||
        !std::isfinite(input_point.y) ||
        !std::isfinite(input_point.z))
    {
      continue;
    }
    const int64_t vx = static_cast<int64_t>(std::floor(input_point.x * inverse_voxel_size));
    const int64_t vy = static_cast<int64_t>(std::floor(input_point.y * inverse_voxel_size));
    const int64_t vz = static_cast<int64_t>(std::floor(input_point.z * inverse_voxel_size));
    // 3 座標 (符号付き 21bit ずつ) を 64bit に詰める。値域 ±2^20 voxel で十分。
    const int64_t voxel_key =
      ((vx & 0x1FFFFF) << 42) | ((vy & 0x1FFFFF) << 21) | (vz & 0x1FFFFF);
    if (!seen_voxel_keys_this_scan.insert(voxel_key).second) {
      continue;
    }
    accumulated_cloud_->points.push_back(input_point);
  }
  accumulated_cloud_->width = static_cast<uint32_t>(accumulated_cloud_->points.size());
  accumulated_cloud_->height = 1;

  // 上限を超えたら古い順に削る単純戦略。
  if (accumulated_cloud_->points.size() > config_.max_total_points) {
    const std::size_t excess =
      accumulated_cloud_->points.size() - config_.max_total_points;
    accumulated_cloud_->points.erase(
      accumulated_cloud_->points.begin(),
      accumulated_cloud_->points.begin() + excess);
    accumulated_cloud_->width =
      static_cast<uint32_t>(accumulated_cloud_->points.size());
  }

  ++inserts_since_rebuild_;
  if (inserts_since_rebuild_ >= config_.rebuild_every_n_insert) {
    rebuildKdTree();
  }
}

PointCorrespondence KdTreeMap::findNearestNeighbor(const Eigen::Vector3d & query_world) const
{
  PointCorrespondence correspondence;
  correspondence.source_point_world = query_world;
  correspondence.valid = false;
  correspondence.squared_distance = std::numeric_limits<double>::infinity();

  if (accumulated_cloud_->empty()) {
    return correspondence;
  }

  Point query_point;
  query_point.x = static_cast<float>(query_world.x());
  query_point.y = static_cast<float>(query_world.y());
  query_point.z = static_cast<float>(query_world.z());
  query_point.intensity = 0.0f;

  std::vector<int> indices;
  std::vector<float> squared_distances;
  const int k_neighbors = std::max(1, config_.neighbor_count_for_covariance);
  if (kd_tree_->nearestKSearch(query_point, k_neighbors, indices, squared_distances) <= 0) {
    return correspondence;
  }

  // 重心と共分散を k 近傍から計算。
  Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
  for (int neighbor_index : indices) {
    const Point & neighbor = accumulated_cloud_->points[neighbor_index];
    centroid += Eigen::Vector3d(neighbor.x, neighbor.y, neighbor.z);
  }
  centroid /= static_cast<double>(indices.size());

  Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
  for (int neighbor_index : indices) {
    const Point & neighbor = accumulated_cloud_->points[neighbor_index];
    const Eigen::Vector3d offset =
      Eigen::Vector3d(neighbor.x, neighbor.y, neighbor.z) - centroid;
    covariance += offset * offset.transpose();
  }
  if (indices.size() > 1) {
    covariance /= static_cast<double>(indices.size() - 1);
  }

  // 共分散の固有値 floor。
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
  Eigen::Vector3d eigenvalues = solver.eigenvalues();
  for (int axis = 0; axis < 3; ++axis) {
    if (eigenvalues(axis) < config_.covariance_eigen_floor) {
      eigenvalues(axis) = config_.covariance_eigen_floor;
    }
  }
  covariance =
    solver.eigenvectors() * eigenvalues.asDiagonal() * solver.eigenvectors().transpose();

  correspondence.target_point_world = centroid;
  correspondence.target_covariance = covariance;
  correspondence.squared_distance = squared_distances.front();
  correspondence.valid = true;
  return correspondence;
}

std::size_t KdTreeMap::size() const
{
  return accumulated_cloud_->points.size();
}

PointCloudPtr KdTreeMap::toPointCloud() const
{
  return accumulated_cloud_;
}

std::string KdTreeMap::describe() const
{
  std::ostringstream oss;
  oss << "kd_tree_map:points=" << accumulated_cloud_->points.size()
      << ",k=" << config_.neighbor_count_for_covariance;
  return oss.str();
}

}  // namespace pylot_lio
