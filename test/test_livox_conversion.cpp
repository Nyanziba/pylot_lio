// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#include <gtest/gtest.h>

#include "pylot_lio/livox_conversion.hpp"

namespace pylot_lio
{

TEST(LivoxConversion, EmptyInputProducesEmptyCloud)
{
  const PointCloud cloud = convertLivoxRawPoints({});
  EXPECT_TRUE(cloud.points.empty());
  EXPECT_EQ(cloud.width, 0u);
  EXPECT_EQ(cloud.height, 1u);
  EXPECT_TRUE(cloud.is_dense);
}

TEST(LivoxConversion, ReflectivityIsCopiedToIntensity)
{
  std::vector<LivoxRawPoint> raw_points;
  raw_points.push_back({1.0f, 2.0f, 3.0f, 200, 1234u});
  raw_points.push_back({-1.0f, 0.5f, -0.5f, 50, 5678u});

  const PointCloud cloud = convertLivoxRawPoints(raw_points);

  ASSERT_EQ(cloud.points.size(), 2u);
  EXPECT_FLOAT_EQ(cloud.points[0].x, 1.0f);
  EXPECT_FLOAT_EQ(cloud.points[0].y, 2.0f);
  EXPECT_FLOAT_EQ(cloud.points[0].z, 3.0f);
  EXPECT_FLOAT_EQ(cloud.points[0].intensity, 200.0f);
  EXPECT_FLOAT_EQ(cloud.points[1].x, -1.0f);
  EXPECT_FLOAT_EQ(cloud.points[1].intensity, 50.0f);
  EXPECT_EQ(cloud.width, 2u);
  EXPECT_EQ(cloud.height, 1u);
}

TEST(LivoxConversion, OrderIsPreserved)
{
  std::vector<LivoxRawPoint> raw_points;
  for (int point_index = 0; point_index < 10; ++point_index) {
    raw_points.push_back({
      static_cast<float>(point_index), 0.0f, 0.0f,
      static_cast<uint8_t>(point_index), static_cast<uint32_t>(point_index * 1000)});
  }
  const PointCloud cloud = convertLivoxRawPoints(raw_points);
  for (int point_index = 0; point_index < 10; ++point_index) {
    EXPECT_FLOAT_EQ(
      cloud.points[point_index].x, static_cast<float>(point_index));
    EXPECT_FLOAT_EQ(
      cloud.points[point_index].intensity, static_cast<float>(point_index));
  }
}

}  // namespace pylot_lio
