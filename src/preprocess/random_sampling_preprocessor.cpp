// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#include "pylot_lio/preprocess/random_sampling_preprocessor.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <sstream>

namespace pylot_lio
{

RandomSamplingPreprocessor::RandomSamplingPreprocessor(
  std::size_t target_point_count, uint32_t random_seed)
: target_point_count_(target_point_count),
  random_engine_(random_seed)
{
}

PointCloudPtr RandomSamplingPreprocessor::process(const PointCloudConstPtr & input_cloud)
{
  auto output_cloud = std::make_shared<PointCloud>();
  if (!input_cloud || input_cloud->empty()) {
    return output_cloud;
  }

  // 有限値だけの index リストを作る。
  std::vector<std::size_t> finite_indices;
  finite_indices.reserve(input_cloud->size());
  for (std::size_t i = 0; i < input_cloud->size(); ++i) {
    const Point & point = input_cloud->points[i];
    if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z)) {
      finite_indices.push_back(i);
    }
  }

  if (finite_indices.size() <= target_point_count_) {
    output_cloud->points.reserve(finite_indices.size());
    for (std::size_t index : finite_indices) {
      output_cloud->points.push_back(input_cloud->points[index]);
    }
  } else {
    // Fisher-Yates の前半 target_point_count_ 要素だけシャッフル。
    for (std::size_t i = 0; i < target_point_count_; ++i) {
      std::uniform_int_distribution<std::size_t> distribution(i, finite_indices.size() - 1);
      const std::size_t swap_index = distribution(random_engine_);
      std::swap(finite_indices[i], finite_indices[swap_index]);
    }
    output_cloud->points.reserve(target_point_count_);
    for (std::size_t i = 0; i < target_point_count_; ++i) {
      output_cloud->points.push_back(input_cloud->points[finite_indices[i]]);
    }
  }

  output_cloud->width = static_cast<uint32_t>(output_cloud->points.size());
  output_cloud->height = 1;
  output_cloud->is_dense = true;
  return output_cloud;
}

std::string RandomSamplingPreprocessor::describe() const
{
  std::ostringstream oss;
  oss << "random_sampling:N=" << target_point_count_;
  return oss.str();
}

}  // namespace pylot_lio
