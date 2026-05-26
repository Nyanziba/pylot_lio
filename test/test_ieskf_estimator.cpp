// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#include <gtest/gtest.h>

#include "pylot_lio/estimator/ieskf_estimator.hpp"
#include "pylot_lio/map/voxel_map.hpp"
#include "pylot_lio/registration/plain_gicp_registration.hpp"

namespace pylot_lio
{

namespace
{
PointCloud makePlaneCloud(double z_offset)
{
  PointCloud cloud;
  for (double x = -2.0; x <= 2.0; x += 0.2) {
    for (double y = -2.0; y <= 2.0; y += 0.2) {
      Point point;
      point.x = static_cast<float>(x);
      point.y = static_cast<float>(y);
      point.z = static_cast<float>(z_offset);
      point.intensity = 0.0f;
      cloud.push_back(point);
    }
  }
  return cloud;
}
}  // namespace

TEST(IeskfEstimator, InitializeSetsPoseAndIsInitialized)
{
  IeskfEstimator estimator(IeskfEstimator::Config{});
  EXPECT_FALSE(estimator.isInitialized());

  RobotState initial;
  initial.pose_world_body.translation() = Eigen::Vector3d(1.0, 2.0, 3.0);
  estimator.initialize(initial);
  EXPECT_TRUE(estimator.isInitialized());

  const auto state = estimator.getState();
  EXPECT_NEAR(state.pose_world_body.translation().x(), 1.0, 1e-9);
  EXPECT_NEAR(state.pose_world_body.translation().y(), 2.0, 1e-9);
  EXPECT_NEAR(state.pose_world_body.translation().z(), 3.0, 1e-9);
}

TEST(IeskfEstimator, GravityOnlyImuKeepsPositionStableInFreeFallCancellation)
{
  // 重力に等しい加速度 (z+gravity_norm) を受けると重力と打ち消し、 位置と速度はゼロ近傍。
  // gravity_estimation_samples を小さくして、 最初の数サンプルで重力方向を確定し、 残りで
  // 予測ループを実走らせる。
  IeskfEstimator::Config config;
  config.gravity_estimation_samples = 10;
  config.gravity_norm = 9.7946;
  IeskfEstimator estimator(config);
  estimator.initialize(RobotState{});

  int64_t timestamp_ns = 0;
  const double delta_t = 0.01;
  for (int i = 0; i < 100; ++i) {
    ImuSample imu_sample;
    imu_sample.timestamp_ns = timestamp_ns;
    // 静止 IMU は +Z 方向に gravity_norm を測定する想定
    imu_sample.linear_acceleration_mps2 = Eigen::Vector3d(0.0, 0.0, config.gravity_norm);
    imu_sample.angular_velocity_rps = Eigen::Vector3d::Zero();
    estimator.predictWithImu(imu_sample);
    timestamp_ns += static_cast<int64_t>(delta_t * 1e9);
  }
  const auto state = estimator.getState();
  EXPECT_LT(std::abs(state.pose_world_body.translation().z()), 0.05);
  EXPECT_LT(state.velocity_world.norm(), 0.05);
}

TEST(IeskfEstimator, ExtrinsicTranslationShiftsScanReferenceFrame)
{
  // extrinsic_translation_imu_from_lidar が非ゼロでも updateWithScan が落ちず、
  // 内部で LiDAR→IMU 変換されることを smoke test。 平面 cloud に対し空マップから始め、
  // map.size() が変わらない (= insertScan は lio_node 側責務) ことだけ確認する。
  IeskfEstimator::Config config;
  config.extrinsic_translation_imu_from_lidar = Eigen::Vector3d(-0.011, -0.02329, 0.04412);
  IeskfEstimator estimator(config);
  estimator.initialize(RobotState{});

  VoxelMap::Config map_config;
  map_config.voxel_size_m = 0.5;
  map_config.min_points_per_cell_for_covariance = 1;
  VoxelMap map(map_config);
  PlainGicpRegistration registration(PlainGicpRegistration::Config{});

  PointCloud cloud;
  for (int point_index = 0; point_index < 20; ++point_index) {
    Point point;
    point.x = 1.0f + 0.1f * static_cast<float>(point_index);
    point.y = 0.0f;
    point.z = 0.0f;
    cloud.push_back(point);
  }
  estimator.updateWithScan(cloud, map, registration, 0);
  EXPECT_EQ(map.size(), 0u);
}

TEST(IeskfEstimator, ScanUpdateLeavesMapInsertionToCaller)
{
  // 設計変更: estimator は map に挿入しない (keyframe ロジックを lio_node 側に移譲)。
  // ここでは「空マップに updateWithScan しても落ちず、map.size() は変わらない」ことを確認。
  IeskfEstimator estimator(IeskfEstimator::Config{});
  estimator.initialize(RobotState{});

  VoxelMap::Config map_config;
  map_config.voxel_size_m = 0.5;
  map_config.min_points_per_cell_for_covariance = 1;
  VoxelMap map(map_config);
  PlainGicpRegistration registration(PlainGicpRegistration::Config{});

  const auto plane_cloud = makePlaneCloud(0.0);
  estimator.updateWithScan(plane_cloud, map, registration, 0);
  EXPECT_EQ(map.size(), 0u);
}

}  // namespace pylot_lio

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
