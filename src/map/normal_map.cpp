// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#include "pylot_lio/map/normal_map.hpp"

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

std::size_t NormalMap::KeyHash::operator()(const Key & key) const noexcept
{
  const auto h1 = std::hash<int64_t>{}(key.x);
  const auto h2 = std::hash<int64_t>{}(key.y);
  const auto h3 = std::hash<int64_t>{}(key.z);
  return h1 ^ (h2 * 0x9E3779B97F4A7C15ULL) ^ (h3 * 0xBF58476D1CE4E5B9ULL);
}

NormalMap::NormalMap(const Config & config)
: config_(config)
{
}

NormalMap::Key NormalMap::computeKey(const Eigen::Vector3d & position_world) const
{
  const double inverse_voxel_size = 1.0 / config_.voxel_size_m;
  return Key{
    static_cast<int64_t>(std::floor(position_world.x() * inverse_voxel_size)),
    static_cast<int64_t>(std::floor(position_world.y() * inverse_voxel_size)),
    static_cast<int64_t>(std::floor(position_world.z() * inverse_voxel_size))
  };
}

Eigen::Matrix3d NormalMap::shapeAsPointToPlane(const Eigen::Matrix3d & raw_covariance) const
{
  // 最小固有値方向 = 法線方向。共分散をその方向だけ小さく、他方向は大きく書き換えると、
  // Mahalanobis 距離が「法線に沿った成分」だけを強く罰する形になる (point-to-plane 相当)。
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(raw_covariance);
  Eigen::Vector3d eigenvalues = solver.eigenvalues();
  Eigen::Index min_index = 0;
  eigenvalues.minCoeff(&min_index);
  Eigen::Vector3d shaped_eigenvalues = eigenvalues;
  for (int axis = 0; axis < 3; ++axis) {
    if (axis == min_index) {
      shaped_eigenvalues(axis) = std::min(eigenvalues(axis), config_.normal_covariance_ceiling);
    } else {
      shaped_eigenvalues(axis) = std::max(eigenvalues(axis), config_.tangent_covariance_floor);
    }
  }
  return solver.eigenvectors() * shaped_eigenvalues.asDiagonal()
    * solver.eigenvectors().transpose();
}

void NormalMap::insertScan(
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
    Cell & cell = cells_[key];
    const int previous_count = cell.sample_count;
    const int new_count = previous_count + 1;
    const Eigen::Vector3d delta_before = position_world - cell.mean;
    cell.mean += delta_before / static_cast<double>(new_count);
    const Eigen::Vector3d delta_after = position_world - cell.mean;
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

  if (cells_.size() > config_.max_total_cells) {
    std::vector<std::pair<int, Key>> sortable;
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
}

PointCorrespondence NormalMap::findNearestNeighbor(const Eigen::Vector3d & query_world) const
{
  PointCorrespondence correspondence;
  correspondence.source_point_world = query_world;
  correspondence.valid = false;
  correspondence.squared_distance = std::numeric_limits<double>::infinity();

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
        if (found->second.sample_count < config_.min_points_per_cell) {
          continue;
        }
        const double squared_distance = (found->second.mean - query_world).squaredNorm();
        if (squared_distance < correspondence.squared_distance) {
          correspondence.squared_distance = squared_distance;
          correspondence.target_point_world = found->second.mean;
          correspondence.target_covariance = shapeAsPointToPlane(found->second.covariance);
          correspondence.valid = true;
        }
      }
    }
  }
  return correspondence;
}

std::size_t NormalMap::size() const
{
  return cells_.size();
}

PointCloudPtr NormalMap::toPointCloud() const
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

std::string NormalMap::describe() const
{
  std::ostringstream oss;
  oss << "normal_map:size=" << config_.voxel_size_m
      << ",cells=" << cells_.size();
  return oss.str();
}

}  // namespace pylot_lio
