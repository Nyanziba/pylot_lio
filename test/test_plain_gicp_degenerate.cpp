// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
//
// 縮退正則化 (Tuna 2024 X-ICP) の挙動を検証する単体テスト。
// 平面 (Z=0) 上の点群は xy 並進と z 回転が幾何だけでは拘束されない (= 縮退方向)。
// 正則化 ON ならその方向は initial_guess から大きく離れない、OFF だと暴れる、を確認する。

#include <gtest/gtest.h>

#include "pylot_lio/map/voxel_map.hpp"
#include "pylot_lio/registration/plain_gicp_registration.hpp"

namespace pylot_lio
{

namespace
{
PointCloud makeFlatPlaneCloud()
{
  PointCloud cloud;
  for (double x = -2.0; x <= 2.0; x += 0.2) {
    for (double y = -2.0; y <= 2.0; y += 0.2) {
      Point point;
      point.x = static_cast<float>(x);
      point.y = static_cast<float>(y);
      point.z = 0.0f;
      point.intensity = 0.0f;
      cloud.push_back(point);
    }
  }
  return cloud;
}
}  // namespace

TEST(DegenerateRegularization, PlanarPointCloudRegularizedStaysNearInitialGuess)
{
  // マップとソースに「Z=0 平面の格子点」を入れる。点群そのものは識別可能だが、
  // ノーマルが全部 +Z 方向なので「Z 軸まわり回転」と「xy 平行移動」は縮退方向になる。
  VoxelMap::Config map_config;
  map_config.voxel_size_m = 0.3;
  map_config.min_points_per_cell_for_covariance = 1;
  VoxelMap voxel_map(map_config);

  const auto plane = makeFlatPlaneCloud();
  voxel_map.insertScan(plane, Eigen::Isometry3d::Identity());

  // 初期推定: 単位姿勢 (= 正解と一致)
  const Eigen::Isometry3d initial_guess = Eigen::Isometry3d::Identity();

  // 正則化 ON
  PlainGicpRegistration::Config gicp_config;
  gicp_config.max_iterations = 20;
  gicp_config.max_correspondence_distance_m = 3.0;
  gicp_config.enable_degenerate_regularization = true;
  gicp_config.rotation_eigenvalue_threshold = 10.0;
  gicp_config.translation_eigenvalue_threshold = 1.0;
  gicp_config.regularization_base_factor = 1.0;

  PlainGicpRegistration registration(gicp_config);
  const auto result = registration.align(plane, voxel_map, initial_guess);

  // 縮退方向 (Z 軸まわり / xy 並進) で initial_guess から大きく離れていないこと。
  // 完全に同一にはならないが、5 cm / 0.1 rad 程度に収まるはず。
  const Eigen::Vector3d translation_drift = result.transform_world_body.translation();
  EXPECT_LT(translation_drift.norm(), 0.05);
}

TEST(DegenerateRegularization, DisabledByDefault)
{
  // 縮退正則化はデフォルトで OFF (互換性維持)。
  PlainGicpRegistration::Config gicp_config;
  EXPECT_FALSE(gicp_config.enable_degenerate_regularization);
}

}  // namespace pylot_lio

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
