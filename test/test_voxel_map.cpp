// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#include <gtest/gtest.h>

#include <Eigen/Eigenvalues>

#include "pylot_lio/map/voxel_map.hpp"

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

TEST(VoxelMap, InsertsCreateCells)
{
  VoxelMap::Config config;
  config.voxel_size_m = 1.0;
  config.min_points_per_cell_for_covariance = 1;
  VoxelMap voxel_map(config);

  PointCloud scan;
  scan.push_back(makePoint(0.1, 0.1, 0.1));
  scan.push_back(makePoint(0.2, 0.2, 0.2));
  scan.push_back(makePoint(3.0, 0.0, 0.0));
  voxel_map.insertScan(scan, Eigen::Isometry3d::Identity());
  EXPECT_EQ(voxel_map.size(), 2u);
}

TEST(VoxelMap, OnlineMeanMatchesBatchMean)
{
  VoxelMap::Config config;
  config.voxel_size_m = 10.0;
  config.min_points_per_cell_for_covariance = 1;
  VoxelMap voxel_map(config);

  PointCloud scan;
  scan.push_back(makePoint(1.0, 2.0, 3.0));
  scan.push_back(makePoint(2.0, 4.0, 6.0));
  scan.push_back(makePoint(3.0, 6.0, 9.0));
  voxel_map.insertScan(scan, Eigen::Isometry3d::Identity());
  ASSERT_EQ(voxel_map.size(), 1u);
  const auto & cell = voxel_map.cells().begin()->second;
  EXPECT_NEAR(cell.mean.x(), 2.0, 1e-9);
  EXPECT_NEAR(cell.mean.y(), 4.0, 1e-9);
  EXPECT_NEAR(cell.mean.z(), 6.0, 1e-9);
  EXPECT_EQ(cell.sample_count, 3);
}

TEST(VoxelMap, NearestNeighborReturnsClosestCellMean)
{
  VoxelMap::Config config;
  config.voxel_size_m = 1.0;
  config.min_points_per_cell_for_covariance = 1;
  config.neighbor_search_radius_voxels = 1;
  VoxelMap voxel_map(config);

  PointCloud scan;
  scan.push_back(makePoint(0.5, 0.5, 0.5));   // cell (0,0,0)
  scan.push_back(makePoint(1.5, 0.5, 0.5));   // cell (1,0,0)
  voxel_map.insertScan(scan, Eigen::Isometry3d::Identity());

  const auto correspondence = voxel_map.findNearestNeighbor(
    Eigen::Vector3d(0.4, 0.4, 0.4));
  ASSERT_TRUE(correspondence.valid);
  EXPECT_NEAR(correspondence.target_point_world.x(), 0.5, 1e-9);
}

TEST(VoxelMap, NearestNeighborReturnsInvalidWhenNoCellNearby)
{
  VoxelMap::Config config;
  config.voxel_size_m = 1.0;
  config.min_points_per_cell_for_covariance = 1;
  config.neighbor_search_radius_voxels = 1;
  VoxelMap voxel_map(config);

  PointCloud scan;
  scan.push_back(makePoint(0.0, 0.0, 0.0));
  voxel_map.insertScan(scan, Eigen::Isometry3d::Identity());

  const auto correspondence = voxel_map.findNearestNeighbor(
    Eigen::Vector3d(100.0, 100.0, 100.0));
  EXPECT_FALSE(correspondence.valid);
}

TEST(VoxelMap, CovarianceFloorIsApplied)
{
  VoxelMap::Config config;
  config.voxel_size_m = 1.0;
  // 単点でも対応点として返るが、共分散はゼロ行列 → eigen floor が効くことを確認。
  config.min_points_per_cell_for_covariance = 1;
  config.neighbor_search_radius_voxels = 1;
  config.covariance_eigen_floor = 0.5;
  VoxelMap voxel_map(config);

  PointCloud scan;
  scan.push_back(makePoint(0.5, 0.5, 0.5));
  voxel_map.insertScan(scan, Eigen::Isometry3d::Identity());

  const auto correspondence = voxel_map.findNearestNeighbor(
    Eigen::Vector3d(0.5, 0.5, 0.5));
  ASSERT_TRUE(correspondence.valid);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(correspondence.target_covariance);
  EXPECT_GE(solver.eigenvalues().minCoeff(), 0.5 - 1e-9);
}

}  // namespace pylot_lio

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
