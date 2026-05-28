// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
//
// Metal voxel random downsampler (前処理 GPU 化、 GLIM randomgrid 相当) の検証。
//   - サンプリング率 0〜1 が出力点数に効くこと (率保持 m = max(1, round(n*r)))。
//   - sampling_rate=0 で各ボクセル 1 点 (= 従来動作、 後方互換)。
//   - 整数演算 (reservoir + xorshift32) なので GPU と CPU 参照は selected_indices が
//     完全一致する (共分散推定の fp32 と違いビット一致を検証できる)。
//
// Metal 非対応ビルドでは gpu_used=false の CPU フォールバックだけ確認する。

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <set>
#include <vector>

#include <Eigen/Core>

#include "pylot_lio/gpu/metal_voxel_downsampler.hpp"

namespace pylot_lio::gpu
{

namespace
{

// 一辺 extent の立方体内にランダムな点群を作る (密度を上げて 1 ボクセルに複数点入れる)。
std::vector<Eigen::Vector3d> makeRandomCloud(int num_points, double extent, uint32_t seed)
{
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist(-extent, extent);
  std::vector<Eigen::Vector3d> points;
  points.reserve(num_points);
  for (int i = 0; i < num_points; ++i) {
    points.emplace_back(dist(rng), dist(rng), dist(rng));
  }
  return points;
}

}  // namespace

TEST(MetalVoxelDownsampler, RateZeroKeepsOnePointPerVoxel)
{
  // sampling_rate=0 → 各ボクセル 1 点 (従来動作)。 出力 index は重複せず入力範囲内。
  const auto points = makeRandomCloud(5000, 2.0, 1);
  VoxelDownsampleConfig config;
  config.voxel_size_m = 0.3f;
  config.sampling_rate = 0.0f;

  const auto selected = voxelRandomDownsampleGridCpu(points, config);
  EXPECT_GT(selected.size(), 0u);
  EXPECT_LE(selected.size(), points.size());

  std::set<std::int32_t> unique(selected.begin(), selected.end());
  EXPECT_EQ(unique.size(), selected.size()) << "選択 index に重複がある";
  for (const std::int32_t index : selected) {
    EXPECT_GE(index, 0);
    EXPECT_LT(index, static_cast<std::int32_t>(points.size()));
  }
}

TEST(MetalVoxelDownsampler, HigherRateKeepsMorePoints)
{
  // 率を上げると保持点数は単調に増える (rate=1 で全点)。
  const auto points = makeRandomCloud(5000, 2.0, 2);
  VoxelDownsampleConfig config;
  config.voxel_size_m = 0.3f;

  config.sampling_rate = 0.0f;
  const std::size_t n0 = voxelRandomDownsampleGridCpu(points, config).size();
  config.sampling_rate = 0.5f;
  const std::size_t n_half = voxelRandomDownsampleGridCpu(points, config).size();
  config.sampling_rate = 1.0f;
  const std::size_t n_full = voxelRandomDownsampleGridCpu(points, config).size();

  EXPECT_LE(n0, n_half);
  EXPECT_LE(n_half, n_full);
  // rate=1 は全点保持 (NaN 無しなので入力数と一致するはず)。
  EXPECT_EQ(n_full, points.size());
}

TEST(MetalVoxelDownsampler, NonFinitePointsAreNeverSelected)
{
  // NaN/Inf 点はどのボクセルにも属さず選択されない。
  auto points = makeRandomCloud(1000, 2.0, 3);
  points[10] = Eigen::Vector3d(std::nan(""), 0.0, 0.0);
  points[20] = Eigen::Vector3d(0.0, std::numeric_limits<double>::infinity(), 0.0);

  VoxelDownsampleConfig config;
  config.voxel_size_m = 0.3f;
  config.sampling_rate = 1.0f;  // 全点保持でも非有限は除外されるはず

  const auto selected = voxelRandomDownsampleGridCpu(points, config);
  for (const std::int32_t index : selected) {
    EXPECT_NE(index, 10);
    EXPECT_NE(index, 20);
  }
  // 有限点は 998 個。 rate=1 で全部選ばれる。
  EXPECT_EQ(selected.size(), points.size() - 2);
}

#ifdef PYLOT_LIO_HAS_METAL

TEST(MetalVoxelDownsampler, GpuExactlyMatchesCpu)
{
  // 整数演算 (reservoir + 同一 xorshift) なので GPU と CPU は完全一致するはず。
  const auto points = makeRandomCloud(20000, 3.0, 4);
  for (float rate : {0.0f, 0.3f, 0.7f, 1.0f}) {
    VoxelDownsampleConfig config;
    config.voxel_size_m = 0.3f;
    config.sampling_rate = rate;
    config.random_seed = 777u;

    const auto cpu = voxelRandomDownsampleGridCpu(points, config);
    const auto gpu = voxelRandomDownsampleMetal(points, config);
    ASSERT_TRUE(gpu.gpu_used) << "Metal path not taken: " << gpu.error_message;
    ASSERT_EQ(gpu.selected_indices.size(), cpu.size()) << "rate=" << rate;
    // 同じ seed・同じグリッド走査順なので index 列も順序込みで一致する。
    EXPECT_EQ(gpu.selected_indices, cpu) << "rate=" << rate;
  }
}

TEST(MetalVoxelDownsampler, EngineReuseIsDeterministic)
{
  const auto points = makeRandomCloud(10000, 3.0, 5);
  VoxelDownsampleConfig config;
  config.voxel_size_m = 0.3f;
  config.sampling_rate = 0.4f;
  config.random_seed = 12345u;

  MetalVoxelDownsampler downsampler;
  ASSERT_TRUE(downsampler.isValid());
  const auto first = downsampler.downsample(points, config);
  const auto second = downsampler.downsample(points, config);
  ASSERT_TRUE(first.gpu_used);
  EXPECT_EQ(first.selected_indices, second.selected_indices);
}

#else  // PYLOT_LIO_HAS_METAL

TEST(MetalVoxelDownsampler, MetalFallsBackToCpuWhenUnavailable)
{
  const auto points = makeRandomCloud(2000, 2.0, 6);
  VoxelDownsampleConfig config;
  config.voxel_size_m = 0.3f;
  config.sampling_rate = 0.5f;

  const auto gpu = voxelRandomDownsampleMetal(points, config);
  EXPECT_FALSE(gpu.gpu_used);
  const auto cpu = voxelRandomDownsampleGridCpu(points, config);
  EXPECT_EQ(gpu.selected_indices, cpu);
}

#endif  // PYLOT_LIO_HAS_METAL

}  // namespace pylot_lio::gpu

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
