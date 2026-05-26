// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#include <gtest/gtest.h>

#include "pylot_lio/preprocess/random_sampling_preprocessor.hpp"

namespace pylot_lio
{

namespace
{
PointCloudPtr makeLineCloud(std::size_t point_count)
{
  auto cloud = std::make_shared<PointCloud>();
  cloud->points.reserve(point_count);
  for (std::size_t i = 0; i < point_count; ++i) {
    Point point;
    point.x = static_cast<float>(i);
    point.y = 0.0f;
    point.z = 0.0f;
    point.intensity = 0.0f;
    cloud->push_back(point);
  }
  return cloud;
}
}  // namespace

TEST(RandomSamplingPreprocessor, ReturnsAllPointsIfBelowTarget)
{
  RandomSamplingPreprocessor preprocessor(/*target=*/100, /*seed=*/42u);
  auto input_cloud = makeLineCloud(50);
  auto output_cloud = preprocessor.process(input_cloud);
  EXPECT_EQ(output_cloud->size(), 50u);
}

TEST(RandomSamplingPreprocessor, ReturnsExactlyTargetPoints)
{
  RandomSamplingPreprocessor preprocessor(/*target=*/30, /*seed=*/42u);
  auto input_cloud = makeLineCloud(1000);
  auto output_cloud = preprocessor.process(input_cloud);
  EXPECT_EQ(output_cloud->size(), 30u);
}

TEST(RandomSamplingPreprocessor, SameSeedProducesSameResult)
{
  auto input_cloud = makeLineCloud(500);
  RandomSamplingPreprocessor first(/*target=*/40, /*seed=*/12345u);
  RandomSamplingPreprocessor second(/*target=*/40, /*seed=*/12345u);
  auto first_output = first.process(input_cloud);
  auto second_output = second.process(input_cloud);
  ASSERT_EQ(first_output->size(), second_output->size());
  for (std::size_t i = 0; i < first_output->size(); ++i) {
    EXPECT_FLOAT_EQ(first_output->points[i].x, second_output->points[i].x);
  }
}

TEST(RandomSamplingPreprocessor, EmptyInputReturnsEmpty)
{
  RandomSamplingPreprocessor preprocessor(100, 1u);
  auto output_cloud = preprocessor.process(std::make_shared<PointCloud>());
  EXPECT_EQ(output_cloud->size(), 0u);
}

}  // namespace pylot_lio

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
