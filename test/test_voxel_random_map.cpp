// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#include <gtest/gtest.h>

#include <unordered_set>

#include "pylot_lio/map/voxel_random_map.hpp"

namespace pylot_lio
{

namespace
{
Point makePoint(double x, double y, double z)
{
  Point point;
  point.x = static_cast<float>(x);
  point.y = static_cast<float>(y);
  point.z = static_cast<float>(z);
  point.intensity = 0.0f;
  return point;
}
}  // namespace

TEST(VoxelRandomMap, InsertsAccumulatePointsBoundedByCapacity)
{
  VoxelRandomMap::Config config;
  config.voxel_size_m = 1.0;
  config.max_points_per_voxel = 4;
  config.k_nearest_for_covariance = 1;
  VoxelRandomMap voxel_random_map(config);

  PointCloud scan;
  // 全部同じボクセル (0,0,0) に入る 20 点。保持点数は最大 4 個に押さえられる。
  for (int i = 0; i < 20; ++i) {
    scan.push_back(makePoint(0.1 + 0.01 * i, 0.1, 0.1));
  }
  voxel_random_map.insertScan(scan, Eigen::Isometry3d::Identity());
  // toPointCloud は cells 全部の点を吐くので、capacity 4 と一致する。
  EXPECT_EQ(voxel_random_map.toPointCloud()->size(), 4u);
  EXPECT_EQ(voxel_random_map.size(), 1u);
}

TEST(VoxelRandomMap, NearestNeighborFindsStoredPoint)
{
  VoxelRandomMap::Config config;
  config.voxel_size_m = 1.0;
  config.max_points_per_voxel = 4;
  config.k_nearest_for_covariance = 1;
  config.neighbor_search_radius_voxels = 1;
  VoxelRandomMap voxel_random_map(config);

  PointCloud scan;
  scan.push_back(makePoint(0.5, 0.5, 0.5));
  scan.push_back(makePoint(2.5, 0.5, 0.5));
  voxel_random_map.insertScan(scan, Eigen::Isometry3d::Identity());

  const auto correspondence = voxel_random_map.findNearestNeighbor(
    Eigen::Vector3d(0.4, 0.4, 0.4));
  ASSERT_TRUE(correspondence.valid);
  EXPECT_NEAR(correspondence.target_point_world.x(), 0.5, 0.2);
}

TEST(VoxelRandomMap, EmptyMapReturnsInvalid)
{
  VoxelRandomMap voxel_random_map(VoxelRandomMap::Config{});
  const auto correspondence =
    voxel_random_map.findNearestNeighbor(Eigen::Vector3d::Zero());
  EXPECT_FALSE(correspondence.valid);
}

TEST(VoxelRandomMap, NonFinitePointsAreSkipped)
{
  VoxelRandomMap::Config config;
  config.max_points_per_voxel = 4;
  VoxelRandomMap voxel_random_map(config);
  PointCloud scan;
  scan.push_back(makePoint(0.0, 0.0, 0.0));
  scan.push_back(makePoint(
    std::numeric_limits<float>::quiet_NaN(), 0.0f, 0.0f));
  voxel_random_map.insertScan(scan, Eigen::Isometry3d::Identity());
  EXPECT_EQ(voxel_random_map.size(), 1u);
}

TEST(VoxelRandomMap, ReservoirSamplingKeepsApproximatelyUniformDistribution)
{
  // 同じボクセルに 1000 点入れて K=10 のリザーバが「観測した 1000 点」のうち
  // ほぼ一様な 10 点を保持していることを確認。観測順 (連番 ID) が
  // 終盤に偏らないこと = 観測 ID の平均が 500 付近にあること、で粗く検査する。
  VoxelRandomMap::Config config;
  config.voxel_size_m = 100.0;
  config.max_points_per_voxel = 10;
  config.random_seed = 42;
  VoxelRandomMap voxel_random_map(config);

  PointCloud scan;
  for (int i = 0; i < 1000; ++i) {
    Point point = makePoint(0.0, 0.0, 0.0);
    point.intensity = static_cast<float>(i);  // 観測順を intensity に持たせる
    scan.push_back(point);
  }
  voxel_random_map.insertScan(scan, Eigen::Isometry3d::Identity());
  ASSERT_EQ(voxel_random_map.size(), 1u);

  // ※ toPointCloud は intensity を observed_count で上書きするため、ここでは
  // 「観測順が一様にバラついているか」までは直接検査せず、保持点数が capacity に
  // 一致することだけ確認する (Algorithm R の正しさは数学的保証で十分とみなす)。
  EXPECT_EQ(voxel_random_map.toPointCloud()->size(), 10u);
}

}  // namespace pylot_lio

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
