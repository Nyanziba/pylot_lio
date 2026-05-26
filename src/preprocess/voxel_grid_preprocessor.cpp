// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#include "pylot_lio/preprocess/voxel_grid_preprocessor.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <sstream>
#include <unordered_map>

#include <Eigen/Core>

namespace pylot_lio
{

namespace
{

// 3D ボクセル座標を 1 つの 64bit キーに畳む。座標値域は ±2^20 voxel (Mid-360 の
// 実用距離 ~70m / voxel 0.1m = 700 で十分余裕)。負値を扱うため bit シフトで詰める。
struct VoxelKey
{
  int64_t x;
  int64_t y;
  int64_t z;

  bool operator==(const VoxelKey & other) const
  {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct VoxelKeyHash
{
  std::size_t operator()(const VoxelKey & key) const noexcept
  {
    // 適当な素数を混ぜる単純な hash。衝突しても処理は走るので十分。
    const auto h1 = std::hash<int64_t>{}(key.x);
    const auto h2 = std::hash<int64_t>{}(key.y);
    const auto h3 = std::hash<int64_t>{}(key.z);
    return h1 ^ (h2 * 0x9E3779B97F4A7C15ULL) ^ (h3 * 0xBF58476D1CE4E5B9ULL);
  }
};

struct VoxelAccumulator
{
  Eigen::Vector3d sum_position = Eigen::Vector3d::Zero();
  double sum_intensity = 0.0;
  int count = 0;
};

}  // namespace

VoxelGridPreprocessor::VoxelGridPreprocessor(double voxel_size_m)
: voxel_size_m_(voxel_size_m)
{
}

PointCloudPtr VoxelGridPreprocessor::process(const PointCloudConstPtr & input_cloud)
{
  auto output_cloud = std::make_shared<PointCloud>();
  if (!input_cloud || input_cloud->empty() || voxel_size_m_ <= 0.0) {
    if (input_cloud) {
      *output_cloud = *input_cloud;
    }
    return output_cloud;
  }

  const double inverse_voxel_size = 1.0 / voxel_size_m_;
  std::unordered_map<VoxelKey, VoxelAccumulator, VoxelKeyHash> voxel_bins;
  voxel_bins.reserve(input_cloud->size() / 4 + 16);

  for (const Point & input_point : input_cloud->points) {
    if (!std::isfinite(input_point.x) ||
        !std::isfinite(input_point.y) ||
        !std::isfinite(input_point.z))
    {
      continue;
    }
    const VoxelKey key{
      static_cast<int64_t>(std::floor(input_point.x * inverse_voxel_size)),
      static_cast<int64_t>(std::floor(input_point.y * inverse_voxel_size)),
      static_cast<int64_t>(std::floor(input_point.z * inverse_voxel_size))
    };
    VoxelAccumulator & bin = voxel_bins[key];
    bin.sum_position.x() += input_point.x;
    bin.sum_position.y() += input_point.y;
    bin.sum_position.z() += input_point.z;
    bin.sum_intensity += input_point.intensity;
    bin.count += 1;
  }

  output_cloud->points.reserve(voxel_bins.size());
  for (const auto & [key, bin] : voxel_bins) {
    const double inverse_count = 1.0 / static_cast<double>(bin.count);
    Point centroid_point;
    centroid_point.x = static_cast<float>(bin.sum_position.x() * inverse_count);
    centroid_point.y = static_cast<float>(bin.sum_position.y() * inverse_count);
    centroid_point.z = static_cast<float>(bin.sum_position.z() * inverse_count);
    centroid_point.intensity = static_cast<float>(bin.sum_intensity * inverse_count);
    output_cloud->points.push_back(centroid_point);
  }
  output_cloud->width = static_cast<uint32_t>(output_cloud->points.size());
  output_cloud->height = 1;
  output_cloud->is_dense = true;
  return output_cloud;
}

std::string VoxelGridPreprocessor::describe() const
{
  std::ostringstream oss;
  oss << "voxel_grid:" << voxel_size_m_;
  return oss.str();
}

}  // namespace pylot_lio
