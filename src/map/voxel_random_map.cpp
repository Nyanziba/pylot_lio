// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#include "pylot_lio/map/voxel_random_map.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <sstream>
#include <utility>

#include <Eigen/Eigenvalues>

namespace pylot_lio
{

std::size_t VoxelRandomMap::KeyHash::operator()(const Key & key) const noexcept
{
  const auto h1 = std::hash<int64_t>{}(key.x);
  const auto h2 = std::hash<int64_t>{}(key.y);
  const auto h3 = std::hash<int64_t>{}(key.z);
  return h1 ^ (h2 * 0x9E3779B97F4A7C15ULL) ^ (h3 * 0xBF58476D1CE4E5B9ULL);
}

VoxelRandomMap::VoxelRandomMap(const Config & config)
: config_(config),
  random_engine_(config.random_seed)
{
}

VoxelRandomMap::Key VoxelRandomMap::computeKey(const Eigen::Vector3d & position_world) const
{
  const double inverse_voxel_size = 1.0 / config_.voxel_size_m;
  return Key{
    static_cast<int64_t>(std::floor(position_world.x() * inverse_voxel_size)),
    static_cast<int64_t>(std::floor(position_world.y() * inverse_voxel_size)),
    static_cast<int64_t>(std::floor(position_world.z() * inverse_voxel_size))
  };
}

void VoxelRandomMap::offerPointToReservoir(
  Reservoir & reservoir, const Eigen::Vector3d & new_point) const
{
  // Algorithm R を 0-indexed で記述:
  //   capacity 個に達するまでは無条件に push_back
  //   それ以降の n_th 観測点 (n_th = observed_count, 0-indexed) は確率 capacity / (n_th + 1) で採用、
  //   採用されたら捨てるスロットを [0, capacity) から一様乱数で選ぶ。
  // ↑これは「[0, n_th] から 1 つ引いて capacity 未満なら そのスロットを置換」と同値で安価。
  const std::size_t capacity = static_cast<std::size_t>(config_.max_points_per_voxel);
  if (reservoir.retained_points.size() < capacity) {
    reservoir.retained_points.push_back(new_point);
  } else {
    std::uniform_int_distribution<uint64_t> distribution(0, reservoir.observed_count);
    const uint64_t candidate_slot = distribution(random_engine_);
    if (candidate_slot < capacity) {
      reservoir.retained_points[static_cast<std::size_t>(candidate_slot)] = new_point;
    }
  }
  ++reservoir.observed_count;
}

void VoxelRandomMap::evictOldCellsIfNeeded()
{
  if (cells_.size() <= config_.max_total_cells) {
    return;
  }
  // observed_count が少ない (= 訪問が浅い) セルから捨てる。VoxelMap と同じ単純戦略。
  std::vector<std::pair<uint64_t, Key>> sortable;
  sortable.reserve(cells_.size());
  for (const auto & [key, reservoir] : cells_) {
    sortable.emplace_back(reservoir.observed_count, key);
  }
  std::sort(sortable.begin(), sortable.end(),
    [](const auto & lhs, const auto & rhs) { return lhs.first < rhs.first; });
  const std::size_t target_to_remove = cells_.size() - config_.max_total_cells;
  for (std::size_t i = 0; i < target_to_remove && i < sortable.size(); ++i) {
    cells_.erase(sortable[i].second);
  }
}

void VoxelRandomMap::insertScan(
  const PointCloud & scan_in_world,
  const Eigen::Isometry3d & /*pose_world_body*/)
{
  for (const Point & input_point : scan_in_world.points) {
    if (!std::isfinite(input_point.x) ||
        !std::isfinite(input_point.y) ||
        !std::isfinite(input_point.z))
    {
      continue;
    }
    const Eigen::Vector3d position_world(input_point.x, input_point.y, input_point.z);
    const Key key = computeKey(position_world);
    offerPointToReservoir(cells_[key], position_world);
  }
  evictOldCellsIfNeeded();
}

PointCorrespondence VoxelRandomMap::findNearestNeighbor(const Eigen::Vector3d & query_world) const
{
  PointCorrespondence correspondence;
  correspondence.source_point_world = query_world;
  correspondence.valid = false;
  correspondence.squared_distance = std::numeric_limits<double>::infinity();

  // 近傍ボクセル群の全点を集める。最大 (2r+1)^3 * K 点。
  std::vector<std::pair<double, Eigen::Vector3d>> candidate_points;
  const Key base_key = computeKey(query_world);
  const int search_radius = config_.neighbor_search_radius_voxels;
  for (int delta_x = -search_radius; delta_x <= search_radius; ++delta_x) {
    for (int delta_y = -search_radius; delta_y <= search_radius; ++delta_y) {
      for (int delta_z = -search_radius; delta_z <= search_radius; ++delta_z) {
        const Key neighbor_key{
          base_key.x + delta_x, base_key.y + delta_y, base_key.z + delta_z};
        const auto found = cells_.find(neighbor_key);
        if (found == cells_.end()) {
          continue;
        }
        for (const Eigen::Vector3d & stored_point : found->second.retained_points) {
          const double squared_distance = (stored_point - query_world).squaredNorm();
          candidate_points.emplace_back(squared_distance, stored_point);
        }
      }
    }
  }

  if (candidate_points.empty()) {
    return correspondence;
  }

  // 最近傍 k 個を選んで PCA で共分散を組む。partial_sort で O(N log k)。
  const std::size_t k_nearest =
    std::min(static_cast<std::size_t>(config_.k_nearest_for_covariance),
             candidate_points.size());
  std::partial_sort(
    candidate_points.begin(),
    candidate_points.begin() + static_cast<std::ptrdiff_t>(k_nearest),
    candidate_points.end(),
    [](const auto & lhs, const auto & rhs) { return lhs.first < rhs.first; });

  Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
  for (std::size_t i = 0; i < k_nearest; ++i) {
    centroid += candidate_points[i].second;
  }
  centroid /= static_cast<double>(k_nearest);

  Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
  for (std::size_t i = 0; i < k_nearest; ++i) {
    const Eigen::Vector3d offset = candidate_points[i].second - centroid;
    covariance += offset * offset.transpose();
  }
  if (k_nearest > 1) {
    covariance /= static_cast<double>(k_nearest - 1);
  }

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
  correspondence.squared_distance = candidate_points.front().first;
  correspondence.valid = true;
  return correspondence;
}

std::size_t VoxelRandomMap::size() const
{
  return cells_.size();
}

PointCloudPtr VoxelRandomMap::toPointCloud() const
{
  auto output_cloud = std::make_shared<PointCloud>();
  for (const auto & [key, reservoir] : cells_) {
    (void)key;
    for (const Eigen::Vector3d & stored_point : reservoir.retained_points) {
      Point output_point;
      output_point.x = static_cast<float>(stored_point.x());
      output_point.y = static_cast<float>(stored_point.y());
      output_point.z = static_cast<float>(stored_point.z());
      output_point.intensity = static_cast<float>(reservoir.observed_count);
      output_cloud->points.push_back(output_point);
    }
  }
  output_cloud->width = static_cast<uint32_t>(output_cloud->points.size());
  output_cloud->height = 1;
  output_cloud->is_dense = true;
  return output_cloud;
}

std::string VoxelRandomMap::describe() const
{
  std::ostringstream oss;
  oss << "voxel_random_map:size=" << config_.voxel_size_m
      << ",K=" << config_.max_points_per_voxel
      << ",cells=" << cells_.size();
  return oss.str();
}

}  // namespace pylot_lio
