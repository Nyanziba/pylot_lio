// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#include <gtest/gtest.h>

#include "pylot_lio/map/kd_tree_map.hpp"

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

TEST(KdTreeMap, InsertsAccumulatePoints)
{
  KdTreeMap::Config config;
  config.neighbor_count_for_covariance = 2;
  KdTreeMap kd_tree_map(config);

  PointCloud scan;
  scan.push_back(makePoint(0.0, 0.0, 0.0));
  scan.push_back(makePoint(1.0, 0.0, 0.0));
  scan.push_back(makePoint(0.0, 1.0, 0.0));
  kd_tree_map.insertScan(scan, Eigen::Isometry3d::Identity());
  EXPECT_EQ(kd_tree_map.size(), 3u);
}

TEST(KdTreeMap, NearestNeighborOnLinePoints)
{
  KdTreeMap::Config config;
  config.neighbor_count_for_covariance = 2;
  KdTreeMap kd_tree_map(config);

  PointCloud scan;
  scan.push_back(makePoint(0.0, 0.0, 0.0));
  scan.push_back(makePoint(1.0, 0.0, 0.0));
  scan.push_back(makePoint(2.0, 0.0, 0.0));
  kd_tree_map.insertScan(scan, Eigen::Isometry3d::Identity());

  const auto correspondence = kd_tree_map.findNearestNeighbor(
    Eigen::Vector3d(0.9, 0.05, 0.0));
  ASSERT_TRUE(correspondence.valid);
  EXPECT_NEAR(correspondence.target_point_world.x(), 0.5, 1.0);
}

TEST(KdTreeMap, EmptyMapReturnsInvalid)
{
  KdTreeMap::Config config;
  KdTreeMap kd_tree_map(config);
  const auto correspondence = kd_tree_map.findNearestNeighbor(Eigen::Vector3d::Zero());
  EXPECT_FALSE(correspondence.valid);
}

}  // namespace pylot_lio

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
