// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#include <gtest/gtest.h>

#include "pylot_lio/preprocess/voxel_grid_preprocessor.hpp"

namespace pylot_lio
{

namespace
{
Point makePoint(float x, float y, float z, float intensity = 0.0f)
{
  Point point;
  point.x = x;
  point.y = y;
  point.z = z;
  point.intensity = intensity;
  return point;
}
}  // namespace

TEST(VoxelGridPreprocessor, EmptyCloudReturnsEmpty)
{
  VoxelGridPreprocessor preprocessor(0.1);
  auto input_cloud = std::make_shared<PointCloud>();
  auto output_cloud = preprocessor.process(input_cloud);
  ASSERT_TRUE(output_cloud);
  EXPECT_EQ(output_cloud->size(), 0u);
}

TEST(VoxelGridPreprocessor, PointsInSameVoxelCollapseToCentroid)
{
  VoxelGridPreprocessor preprocessor(1.0);
  auto input_cloud = std::make_shared<PointCloud>();
  input_cloud->push_back(makePoint(0.1f, 0.2f, 0.3f, 10.0f));
  input_cloud->push_back(makePoint(0.3f, 0.4f, 0.5f, 20.0f));
  input_cloud->push_back(makePoint(0.5f, 0.6f, 0.7f, 30.0f));
  auto output_cloud = preprocessor.process(input_cloud);
  ASSERT_EQ(output_cloud->size(), 1u);
  EXPECT_NEAR(output_cloud->points.front().x, 0.3f, 1e-5f);
  EXPECT_NEAR(output_cloud->points.front().y, 0.4f, 1e-5f);
  EXPECT_NEAR(output_cloud->points.front().z, 0.5f, 1e-5f);
  EXPECT_NEAR(output_cloud->points.front().intensity, 20.0f, 1e-5f);
}

TEST(VoxelGridPreprocessor, PointsInDifferentVoxelsAreKeptSeparate)
{
  VoxelGridPreprocessor preprocessor(1.0);
  auto input_cloud = std::make_shared<PointCloud>();
  input_cloud->push_back(makePoint(0.1f, 0.1f, 0.1f));
  input_cloud->push_back(makePoint(1.5f, 0.1f, 0.1f));
  input_cloud->push_back(makePoint(0.1f, 2.7f, 0.1f));
  auto output_cloud = preprocessor.process(input_cloud);
  EXPECT_EQ(output_cloud->size(), 3u);
}

TEST(VoxelGridPreprocessor, NonFinitePointsAreSkipped)
{
  VoxelGridPreprocessor preprocessor(1.0);
  auto input_cloud = std::make_shared<PointCloud>();
  input_cloud->push_back(makePoint(0.0f, 0.0f, 0.0f));
  input_cloud->push_back(makePoint(
    std::numeric_limits<float>::quiet_NaN(), 0.0f, 0.0f));
  auto output_cloud = preprocessor.process(input_cloud);
  EXPECT_EQ(output_cloud->size(), 1u);
}

TEST(VoxelGridPreprocessor, NegativeVoxelSizeIsPassThrough)
{
  VoxelGridPreprocessor preprocessor(-1.0);
  auto input_cloud = std::make_shared<PointCloud>();
  input_cloud->push_back(makePoint(0.1f, 0.1f, 0.1f));
  input_cloud->push_back(makePoint(0.2f, 0.2f, 0.2f));
  auto output_cloud = preprocessor.process(input_cloud);
  EXPECT_EQ(output_cloud->size(), 2u);
}

}  // namespace pylot_lio

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
