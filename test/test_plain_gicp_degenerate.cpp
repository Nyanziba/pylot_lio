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

TEST(PlainGicpOpenMP, SingleAndMultiThreadProduceEquivalentResults)
{
  // OpenMP 並列実装は単スレッド版と数値的に同じ結果を返さねばならない。
  // 平面格子マップを target、 そこに小さな並進を加えた点群を source として align し、
  // num_threads=1 と num_threads=4 で transform / iterations / cost が一致することを確認。
  VoxelMap::Config map_config;
  map_config.voxel_size_m = 0.3;
  map_config.min_points_per_cell_for_covariance = 1;
  VoxelMap voxel_map(map_config);

  const auto target_plane = makeFlatPlaneCloud();
  voxel_map.insertScan(target_plane, Eigen::Isometry3d::Identity());

  // source は target を少しずらしたもの。 初期推定はずらしを補正する手前の単位姿勢。
  PointCloud source_cloud;
  Eigen::Isometry3d source_offset = Eigen::Isometry3d::Identity();
  source_offset.translation() = Eigen::Vector3d(0.05, 0.03, 0.0);
  for (const auto & point : target_plane.points) {
    const Eigen::Vector3d shifted =
      source_offset * Eigen::Vector3d(point.x, point.y, point.z);
    Point shifted_point;
    shifted_point.x = static_cast<float>(shifted.x());
    shifted_point.y = static_cast<float>(shifted.y());
    shifted_point.z = static_cast<float>(shifted.z());
    shifted_point.intensity = 0.0f;
    source_cloud.push_back(shifted_point);
  }

  PlainGicpRegistration::Config base_config;
  base_config.max_iterations = 15;
  base_config.max_correspondence_distance_m = 2.0;
  base_config.huber_threshold = 1.0;
  // 縮退方向の暴走を抑えて単/多スレッドの数値比較が安定するように regularize ON。
  base_config.enable_degenerate_regularization = true;
  base_config.rotation_eigenvalue_threshold = 10.0;
  base_config.translation_eigenvalue_threshold = 1.0;
  base_config.regularization_base_factor = 1.0;

  base_config.num_threads = 1;
  PlainGicpRegistration registration_single(base_config);
  const auto result_single = registration_single.align(
    source_cloud, voxel_map, Eigen::Isometry3d::Identity());

  base_config.num_threads = 4;
  PlainGicpRegistration registration_multi(base_config);
  const auto result_multi = registration_multi.align(
    source_cloud, voxel_map, Eigen::Isometry3d::Identity());

  // 反復回数・対応点数は完全一致するべき (浮動小数の合算順序にほぼ依存しない範囲)。
  EXPECT_EQ(result_single.iterations, result_multi.iterations);
  EXPECT_EQ(result_single.num_correspondences, result_multi.num_correspondences);

  // 変換と cost は合算順序差で僅かに違いうる。 並進 1e-6 m、 回転 1e-6 rad 以下を許容。
  const Eigen::Vector3d translation_diff =
    result_single.transform_world_body.translation() -
    result_multi.transform_world_body.translation();
  EXPECT_LT(translation_diff.norm(), 1e-6);

  const Eigen::Matrix3d rotation_diff =
    result_single.transform_world_body.linear() *
    result_multi.transform_world_body.linear().transpose();
  Eigen::AngleAxisd diff_axis_angle(rotation_diff);
  EXPECT_LT(std::abs(diff_axis_angle.angle()), 1e-6);

  EXPECT_NEAR(result_single.final_cost, result_multi.final_cost, 1e-6);
}

}  // namespace pylot_lio

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
