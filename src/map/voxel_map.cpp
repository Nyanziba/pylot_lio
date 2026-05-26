// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#include "pylot_lio/map/voxel_map.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <sstream>
#include <vector>

#include <Eigen/Eigenvalues>

namespace pylot_lio
{

std::size_t VoxelMap::VoxelKeyHash::operator()(const VoxelKey & key) const noexcept
{
  const auto h1 = std::hash<int64_t>{}(key.x);
  const auto h2 = std::hash<int64_t>{}(key.y);
  const auto h3 = std::hash<int64_t>{}(key.z);
  return h1 ^ (h2 * 0x9E3779B97F4A7C15ULL) ^ (h3 * 0xBF58476D1CE4E5B9ULL);
}

VoxelMap::VoxelMap(const Config & config)
: config_(config)
{
}

VoxelMap::VoxelKey VoxelMap::computeKey(const Eigen::Vector3d & position_world) const
{
  const double inverse_voxel_size = 1.0 / config_.voxel_size_m;
  return VoxelKey{
    static_cast<int64_t>(std::floor(position_world.x() * inverse_voxel_size)),
    static_cast<int64_t>(std::floor(position_world.y() * inverse_voxel_size)),
    static_cast<int64_t>(std::floor(position_world.z() * inverse_voxel_size))
  };
}

void VoxelMap::updateCellOnline(GaussianCell & cell, const Eigen::Vector3d & new_point)
{
  // Welford 流オンライン平均・共分散更新 (数値安定)。
  // 古いサンプルの統計は捨てずに増えていくが、max_points_per_cell で頭打ち。
  if (cell.sample_count >= config_.max_points_per_cell) {
    return;
  }
  const int previous_count = cell.sample_count;
  const int new_count = previous_count + 1;
  const Eigen::Vector3d delta_before = new_point - cell.mean;
  cell.mean += delta_before / static_cast<double>(new_count);
  const Eigen::Vector3d delta_after = new_point - cell.mean;
  // M2 行列を覚える代わりに covariance * (n - 1) を更新する戦略
  if (previous_count == 0) {
    cell.covariance = Eigen::Matrix3d::Zero();
  } else {
    cell.covariance *= static_cast<double>(previous_count - 1);
  }
  cell.covariance += delta_before * delta_after.transpose();
  if (new_count > 1) {
    cell.covariance /= static_cast<double>(new_count - 1);
  }
  cell.sample_count = new_count;
}

void VoxelMap::evictOldCellsIfNeeded()
{
  if (cells_.size() <= config_.max_total_cells) {
    return;
  }
  // FIFO 的ではなくサンプル数の少ないセルから捨てる単純戦略。
  std::vector<std::pair<int, VoxelKey>> sortable;
  sortable.reserve(cells_.size());
  for (const auto & [key, cell] : cells_) {
    sortable.emplace_back(cell.sample_count, key);
  }
  std::sort(sortable.begin(), sortable.end(),
    [](const auto & lhs, const auto & rhs) { return lhs.first < rhs.first; });
  const std::size_t target_to_remove = cells_.size() - config_.max_total_cells;
  for (std::size_t i = 0; i < target_to_remove && i < sortable.size(); ++i) {
    cells_.erase(sortable[i].second);
  }
}

void VoxelMap::insertScan(
  const PointCloud & scan_in_world,
  const Eigen::Isometry3d & /*pose_world_body*/)
{
  // pose_world_body は KdTreeMap などで使う想定だが VoxelMap は world 座標を直接受け取るので不要。
  for (const Point & input_point : scan_in_world.points) {
    if (!std::isfinite(input_point.x) ||
        !std::isfinite(input_point.y) ||
        !std::isfinite(input_point.z))
    {
      continue;
    }
    const Eigen::Vector3d position_world(input_point.x, input_point.y, input_point.z);
    const VoxelKey key = computeKey(position_world);
    GaussianCell & cell = cells_[key];
    updateCellOnline(cell, position_world);
  }
  evictOldCellsIfNeeded();
}

PointCorrespondence VoxelMap::findNearestNeighbor(const Eigen::Vector3d & query_world) const
{
  PointCorrespondence correspondence;
  correspondence.source_point_world = query_world;
  correspondence.valid = false;
  correspondence.squared_distance = std::numeric_limits<double>::infinity();

  const VoxelKey base_key = computeKey(query_world);
  const int search_radius = config_.neighbor_search_radius_voxels;

  for (int delta_x = -search_radius; delta_x <= search_radius; ++delta_x) {
    for (int delta_y = -search_radius; delta_y <= search_radius; ++delta_y) {
      for (int delta_z = -search_radius; delta_z <= search_radius; ++delta_z) {
        const VoxelKey neighbor_key{
          base_key.x + delta_x, base_key.y + delta_y, base_key.z + delta_z};
        const auto found = cells_.find(neighbor_key);
        if (found == cells_.end()) {
          continue;
        }
        if (found->second.sample_count < config_.min_points_per_cell_for_covariance) {
          continue;
        }
        const double squared_distance = (found->second.mean - query_world).squaredNorm();
        if (squared_distance < correspondence.squared_distance) {
          correspondence.squared_distance = squared_distance;
          correspondence.target_point_world = found->second.mean;
          correspondence.target_covariance = found->second.covariance;
          correspondence.valid = true;
        }
      }
    }
  }

  if (correspondence.valid) {
    // 共分散の最小固有値を floor で持ち上げる (退化方向の暴走を防ぐ)。
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(correspondence.target_covariance);
    Eigen::Vector3d eigenvalues = solver.eigenvalues();
    for (int axis = 0; axis < 3; ++axis) {
      if (eigenvalues(axis) < config_.covariance_eigen_floor) {
        eigenvalues(axis) = config_.covariance_eigen_floor;
      }
    }
    correspondence.target_covariance =
      solver.eigenvectors() * eigenvalues.asDiagonal() * solver.eigenvectors().transpose();
  }
  return correspondence;
}

std::size_t VoxelMap::size() const
{
  return cells_.size();
}

PointCloudPtr VoxelMap::toPointCloud() const
{
  auto output_cloud = std::make_shared<PointCloud>();
  output_cloud->points.reserve(cells_.size());
  for (const auto & [key, cell] : cells_) {
    (void)key;
    Point output_point;
    output_point.x = static_cast<float>(cell.mean.x());
    output_point.y = static_cast<float>(cell.mean.y());
    output_point.z = static_cast<float>(cell.mean.z());
    output_point.intensity = static_cast<float>(cell.sample_count);
    output_cloud->points.push_back(output_point);
  }
  output_cloud->width = static_cast<uint32_t>(output_cloud->points.size());
  output_cloud->height = 1;
  output_cloud->is_dense = true;
  return output_cloud;
}

std::string VoxelMap::describe() const
{
  std::ostringstream oss;
  oss << "voxel_map:size=" << config_.voxel_size_m
      << ",cells=" << cells_.size();
  return oss.str();
}

const std::unordered_map<VoxelMap::VoxelKey, VoxelMap::GaussianCell, VoxelMap::VoxelKeyHash> &
VoxelMap::cells() const
{
  return cells_;
}

}  // namespace pylot_lio
