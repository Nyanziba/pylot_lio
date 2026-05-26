// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#include <gtest/gtest.h>

#include "pylot_lio/estimator/gicp_only_estimator.hpp"
#include "pylot_lio/map/voxel_map.hpp"
#include "pylot_lio/registration/plain_gicp_registration.hpp"

namespace pylot_lio
{

TEST(GicpOnlyEstimator, InitializeAndGetState)
{
  GicpOnlyEstimator estimator;
  EXPECT_FALSE(estimator.isInitialized());
  RobotState initial;
  initial.pose_world_body.translation() = Eigen::Vector3d(0.5, 0.0, 0.0);
  estimator.initialize(initial);
  EXPECT_TRUE(estimator.isInitialized());
  EXPECT_NEAR(estimator.getState().pose_world_body.translation().x(), 0.5, 1e-9);
}

TEST(GicpOnlyEstimator, PredictWithImuIsNoOp)
{
  GicpOnlyEstimator estimator;
  estimator.initialize(RobotState{});
  ImuSample sample;
  sample.timestamp_ns = 0;
  sample.linear_acceleration_mps2 = Eigen::Vector3d(1.0, 0.0, 0.0);
  sample.angular_velocity_rps = Eigen::Vector3d::Zero();
  estimator.predictWithImu(sample);
  EXPECT_NEAR(estimator.getState().pose_world_body.translation().norm(), 0.0, 1e-9);
}

TEST(GicpOnlyEstimator, UpdateWithEmptyMapPropagatesIdentity)
{
  GicpOnlyEstimator estimator;
  estimator.initialize(RobotState{});
  VoxelMap map(VoxelMap::Config{});
  PlainGicpRegistration registration(PlainGicpRegistration::Config{});
  PointCloud empty_cloud;
  estimator.updateWithScan(empty_cloud, map, registration, 0);
  EXPECT_NEAR(estimator.getState().pose_world_body.translation().norm(), 0.0, 1e-9);
}

}  // namespace pylot_lio

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
