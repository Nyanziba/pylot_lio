// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#include <gtest/gtest.h>

#include <memory>
#include <unordered_set>

#include "pylot_lio/preprocess/voxel_random_sampling_preprocessor.hpp"

namespace pylot_lio
{

namespace
{

PointCloudPtr makeDenseCloud(int per_voxel, int num_voxels_per_axis, double voxel_size)
{
  auto cloud = std::make_shared<PointCloud>();
  for (int ix = 0; ix < num_voxels_per_axis; ++ix) {
    for (int iy = 0; iy < num_voxels_per_axis; ++iy) {
      for (int iz = 0; iz < num_voxels_per_axis; ++iz) {
        for (int point_index = 0; point_index < per_voxel; ++point_index) {
          Point sample;
          // voxel 内部にランダムに散らす
          const double offset = 0.01 * point_index;
          sample.x = static_cast<float>(ix * voxel_size + offset);
          sample.y = static_cast<float>(iy * voxel_size + offset);
          sample.z = static_cast<float>(iz * voxel_size + offset);
          sample.intensity = 1.0f;
          cloud->points.push_back(sample);
        }
      }
    }
  }
  cloud->width = static_cast<uint32_t>(cloud->points.size());
  cloud->height = 1;
  cloud->is_dense = true;
  return cloud;
}

}  // namespace

TEST(VoxelRandomSamplingPreprocessor, ReducesPointsToOnePerVoxel)
{
  VoxelRandomSamplingPreprocessor::Config config;
  config.voxel_size_m = 1.0;
  config.random_seed = 42u;
  VoxelRandomSamplingPreprocessor preprocessor(config);

  // 3x3x3 = 27 voxel、 1 voxel あたり 5 点 → 入力 135 点
  const auto input = makeDenseCloud(5, 3, 1.0);
  const auto output = preprocessor.process(input);

  // 出力は voxel あたり 1 点 = 27 点
  EXPECT_EQ(output->points.size(), 27u);
  EXPECT_EQ(output->width, 27u);
  EXPECT_EQ(output->height, 1u);
}

TEST(VoxelRandomSamplingPreprocessor, SelectsRealPointsNotCentroid)
{
  VoxelRandomSamplingPreprocessor::Config config;
  config.voxel_size_m = 1.0;
  config.random_seed = 42u;
  VoxelRandomSamplingPreprocessor preprocessor(config);

  // 1 voxel に「x=0」と「x=1000」(極端な外れ値) の 2 点を置く。
  // voxel_grid なら重心 x=500 を作るが、 random sampling は実点 (0 or 1000) を返す。
  auto input = std::make_shared<PointCloud>();
  Point near_point;
  near_point.x = 0.0f; near_point.y = 0.0f; near_point.z = 0.0f;
  Point far_point;
  far_point.x = 1000.0f; far_point.y = 0.0f; far_point.z = 0.0f;
  input->points = {near_point, far_point};
  input->width = 2; input->height = 1; input->is_dense = true;

  // 1 voxel に収まるよう大きな voxel
  config.voxel_size_m = 10000.0;
  VoxelRandomSamplingPreprocessor large_voxel_preprocessor(config);
  const auto output = large_voxel_preprocessor.process(input);

  ASSERT_EQ(output->points.size(), 1u);
  // 選ばれたのが 0 か 1000 のどちらか (= 重心 500 にはならない)
  const float selected_x = output->points[0].x;
  EXPECT_TRUE(selected_x == 0.0f || selected_x == 1000.0f);
}

TEST(VoxelRandomSamplingPreprocessor, EmptyInputProducesEmptyOutput)
{
  VoxelRandomSamplingPreprocessor preprocessor(VoxelRandomSamplingPreprocessor::Config{});
  auto input = std::make_shared<PointCloud>();
  const auto output = preprocessor.process(input);
  EXPECT_TRUE(output->points.empty());
  EXPECT_EQ(output->width, 0u);
}

TEST(VoxelRandomSamplingPreprocessor, DeterministicWithFixedSeed)
{
  // 同一 seed なら 2 回の出力が一致する
  VoxelRandomSamplingPreprocessor::Config config;
  config.voxel_size_m = 1.0;
  config.random_seed = 999u;

  const auto input = makeDenseCloud(10, 3, 1.0);

  VoxelRandomSamplingPreprocessor preprocessor_first(config);
  const auto output_first = preprocessor_first.process(input);

  VoxelRandomSamplingPreprocessor preprocessor_second(config);
  const auto output_second = preprocessor_second.process(input);

  ASSERT_EQ(output_first->points.size(), output_second->points.size());
  // 出力順序は unordered_map なので順番一致は期待できない。 「集合として」一致するか
  // を確認する。 各点を簡易ハッシュ (x*1000+y) でまとめて比較。
  std::unordered_set<int64_t> first_signature;
  for (const auto & point : output_first->points) {
    first_signature.insert(
      static_cast<int64_t>(point.x * 100.0f) * 10000 +
      static_cast<int64_t>(point.y * 100.0f));
  }
  for (const auto & point : output_second->points) {
    const int64_t signature =
      static_cast<int64_t>(point.x * 100.0f) * 10000 +
      static_cast<int64_t>(point.y * 100.0f);
    EXPECT_TRUE(first_signature.find(signature) != first_signature.end());
  }
}

}  // namespace pylot_lio
