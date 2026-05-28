// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#include "pylot_lio/preprocess/voxel_random_sampling_preprocessor.hpp"

#include <cstdint>
#include <memory>
#include <random>
#include <sstream>
#include <vector>

namespace pylot_lio
{

VoxelRandomSamplingPreprocessor::VoxelRandomSamplingPreprocessor(const Config & config)
: config_(config),
  random_engine_(
    config.random_seed != 0u ? config.random_seed : std::random_device{}()),
  downsampler_(
    config.use_gpu ? std::make_shared<gpu::MetalVoxelDownsampler>() : nullptr)
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

  // 入力点を Vector3d 配列に変換 (非有限点も並びを保つ。 ダウンサンプラ側でセルに
  // 割り当てられず選択対象から外れる)。 index ベースで返るので元の Point の全フィールド
  // (intensity 等) を保ったままコピーできる。
  std::vector<Eigen::Vector3d> points;
  points.reserve(input_cloud->points.size());
  for (const Point & input_point : input_cloud->points) {
    points.emplace_back(input_point.x, input_point.y, input_point.z);
  }

  // seed=0 はマシン乱数。 GPU/CPU で決定的にするため呼び出し側で 1 回解決して渡す。
  gpu::VoxelDownsampleConfig downsample_config;
  downsample_config.voxel_size_m = static_cast<float>(config_.voxel_size_m);
  downsample_config.sampling_rate = static_cast<float>(config_.sampling_rate);
  downsample_config.random_seed =
    config_.random_seed != 0u
      ? config_.random_seed
      : static_cast<std::uint32_t>(random_engine_());

  // use_gpu なら Metal で間引く (GPU 無効ビルド / デバイス無では engine が CPU に
  // 自動フォールバック)。 use_gpu=false のときは直接 CPU 参照を呼ぶ。
  const std::vector<std::int32_t> selected =
    (config_.use_gpu && downsampler_)
      ? downsampler_->downsample(points, downsample_config).selected_indices
      : gpu::voxelRandomDownsampleGridCpu(points, downsample_config);

  output_cloud->points.reserve(selected.size());
  for (const std::int32_t index : selected) {
    output_cloud->points.push_back(input_cloud->points[static_cast<std::size_t>(index)]);
  }
  output_cloud->width = static_cast<uint32_t>(output_cloud->points.size());
  output_cloud->height = 1;
  output_cloud->is_dense = true;
  return output_cloud;
}

std::string VoxelRandomSamplingPreprocessor::describe() const
{
  std::ostringstream oss;
  const bool gpu_active = config_.use_gpu && downsampler_ && downsampler_->isValid();
  oss << "voxel_random_sampling:size=" << config_.voxel_size_m
      << ",rate=" << config_.sampling_rate
      << ",seed=" << config_.random_seed
      << ",backend=" << (gpu_active ? "metal_gpu" : "cpu");
  return oss.str();
}

}  // namespace pylot_lio
