// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#include "pylot_lio/preprocess/voxel_random_sampling_preprocessor.hpp"

#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <random>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace pylot_lio
{

namespace
{

struct VoxelKey
{
  int64_t x;
  int64_t y;
  int64_t z;
  bool operator==(const VoxelKey & other) const noexcept
  {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct VoxelKeyHash
{
  std::size_t operator()(const VoxelKey & key) const noexcept
  {
    const auto h1 = std::hash<int64_t>{}(key.x);
    const auto h2 = std::hash<int64_t>{}(key.y);
    const auto h3 = std::hash<int64_t>{}(key.z);
    return h1 ^ (h2 * 0x9E3779B97F4A7C15ULL) ^ (h3 * 0xBF58476D1CE4E5B9ULL);
  }
};

}  // namespace

VoxelRandomSamplingPreprocessor::VoxelRandomSamplingPreprocessor(const Config & config)
: config_(config),
  random_engine_(
    config.random_seed != 0u ? config.random_seed : std::random_device{}())
{
}

PointCloudPtr VoxelRandomSamplingPreprocessor::process(
  const PointCloudConstPtr & input_cloud)
{
  auto output_cloud = std::make_shared<PointCloud>();
  if (!input_cloud || input_cloud->empty()) {
    output_cloud->width = 0;
    output_cloud->height = 1;
    output_cloud->is_dense = true;
    return output_cloud;
  }

  // 各 voxel に「観測した点の index 配列」を持たせる。 全部メモリに保持して最後に
  // 各 voxel から 1 つだけ選ぶ方式。 voxel あたり 数〜数十点が普通なのでメモリは軽い。
  std::unordered_map<VoxelKey, std::vector<std::size_t>, VoxelKeyHash>
    indices_per_voxel;
  indices_per_voxel.reserve(input_cloud->points.size() / 4 + 1);

  const double inverse_voxel_size = 1.0 / config_.voxel_size_m;
  for (std::size_t point_index = 0; point_index < input_cloud->points.size();
       ++point_index)
  {
    const Point & input_point = input_cloud->points[point_index];
    if (!std::isfinite(input_point.x) || !std::isfinite(input_point.y) ||
        !std::isfinite(input_point.z))
    {
      continue;
    }
    const VoxelKey key{
      static_cast<int64_t>(std::floor(input_point.x * inverse_voxel_size)),
      static_cast<int64_t>(std::floor(input_point.y * inverse_voxel_size)),
      static_cast<int64_t>(std::floor(input_point.z * inverse_voxel_size))
    };
    indices_per_voxel[key].push_back(point_index);
  }

  output_cloud->points.reserve(indices_per_voxel.size());
  for (auto & [key, indices_in_voxel] : indices_per_voxel) {
    (void)key;
    if (indices_in_voxel.empty()) {
      continue;
    }
    // 元々の点から 1 つランダムに選ぶ (= voxel 内の代表点)。
    std::uniform_int_distribution<std::size_t> distribution(
      0, indices_in_voxel.size() - 1);
    const std::size_t selected_index =
      indices_in_voxel[distribution(random_engine_)];
    output_cloud->points.push_back(input_cloud->points[selected_index]);
  }
  output_cloud->width = static_cast<uint32_t>(output_cloud->points.size());
  output_cloud->height = 1;
  output_cloud->is_dense = true;
  return output_cloud;
}

std::string VoxelRandomSamplingPreprocessor::describe() const
{
  std::ostringstream oss;
  oss << "voxel_random_sampling:size=" << config_.voxel_size_m
      << ",seed=" << config_.random_seed;
  return oss.str();
}

}  // namespace pylot_lio
