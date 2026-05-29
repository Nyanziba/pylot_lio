// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#include <gtest/gtest.h>

#include <cmath>

#include <Eigen/Geometry>

#include "pylot_lio/map/voxel_map.hpp"
#include "pylot_lio/registration/pcl_ndt_registration.hpp"

namespace pylot_lio
{

namespace
{

Point makePoint(double x_value, double y_value, double z_value)
{
  Point point;
  point.x = static_cast<float>(x_value);
  point.y = static_cast<float>(y_value);
  point.z = static_cast<float>(z_value);
  point.intensity = 0.0f;
  return point;
}

// 3 軸方向に十分な構造を持つテスト用点群を生成する。
// 平面 1 枚だと NDT は法線方向以外の自由度を拘束できないため、 床 + 壁 2 枚の
// 「コーナー」 形状にして 6 自由度を観測可能にする。
PointCloud buildCornerCloud()
{
  PointCloud corner_cloud;
  for (double u = -3.0; u <= 3.0; u += 0.08) {
    for (double v = -3.0; v <= 3.0; v += 0.08) {
      // 床 z = 0
      corner_cloud.push_back(makePoint(u, v, 0.0));
      // 壁 x = 3
      corner_cloud.push_back(makePoint(3.0, u, v + 3.0));
      // 壁 y = 3
      corner_cloud.push_back(makePoint(u, 3.0, v + 3.0));
    }
  }
  return corner_cloud;
}

}  // namespace

TEST(PclNdtRegistration, AlignsIdentityWhenSourceMatchesMap)
{
  // 同一点群を target と source に渡せば結果は単位変換に近い。
  VoxelMap::Config map_config;
  // 注意: VoxelMap::toPointCloud() は各セルから「平均 1 点」 しか返さないため、
  // NDT の内部 VoxelGridCovariance (1 voxel あたり最低 6 点要求) が餓死する。
  // テストでは map の voxel を細かくして「実質的に全点を残す」 構成にする。
  map_config.voxel_size_m = 0.05;
  map_config.min_points_per_cell_for_covariance = 1;
  VoxelMap voxel_map(map_config);

  const PointCloud corner_cloud = buildCornerCloud();
  voxel_map.insertScan(corner_cloud, Eigen::Isometry3d::Identity());

  PclNdtRegistration::Config registration_config;
  registration_config.resolution_m = 2.0;
  registration_config.step_size_m = 0.2;
  registration_config.transformation_epsilon_m = 1e-5;
  registration_config.max_iterations = 100;
  PclNdtRegistration registration(registration_config);

  const auto result = registration.align(
    corner_cloud, voxel_map, Eigen::Isometry3d::Identity());

  EXPECT_LT(result.transform_world_body.translation().norm(), 0.1);
  const Eigen::AngleAxisd angle_axis(result.transform_world_body.rotation());
  EXPECT_LT(std::abs(angle_axis.angle()), 0.05);
}

TEST(PclNdtRegistration, RecoversSmallTranslationOffset)
{
  // map に対して source を既知の小オフセットでずらし、 NDT が打ち消す向きに
  // 収束することを確認する。
  VoxelMap::Config map_config;
  // 注意: VoxelMap::toPointCloud() は各セルから「平均 1 点」 しか返さないため、
  // NDT の内部 VoxelGridCovariance (1 voxel あたり最低 6 点要求) が餓死する。
  // テストでは map の voxel を細かくして「実質的に全点を残す」 構成にする。
  map_config.voxel_size_m = 0.05;
  map_config.min_points_per_cell_for_covariance = 1;
  VoxelMap voxel_map(map_config);

  const PointCloud reference_cloud = buildCornerCloud();
  voxel_map.insertScan(reference_cloud, Eigen::Isometry3d::Identity());

  // body フレームの source を構築: world での真の姿勢を
  // T_world_body_true = translation(0.3, -0.2, 0.1) と仮定し、 source 点群は
  // world 点を T_world_body_true^{-1} で body に持ってきたもの。
  const Eigen::Isometry3d true_transform_world_body = []() {
    Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
    transform.translation() = Eigen::Vector3d(0.15, -0.1, 0.05);
    return transform;
  }();
  PointCloud source_cloud_body;
  for (const Point & world_point : reference_cloud.points) {
    const Eigen::Vector3d point_world(world_point.x, world_point.y, world_point.z);
    const Eigen::Vector3d point_body = true_transform_world_body.inverse() * point_world;
    source_cloud_body.push_back(makePoint(point_body.x(), point_body.y(), point_body.z()));
  }

  PclNdtRegistration::Config registration_config;
  registration_config.resolution_m = 2.0;
  registration_config.step_size_m = 0.2;
  registration_config.transformation_epsilon_m = 1e-5;
  registration_config.max_iterations = 100;
  PclNdtRegistration registration(registration_config);

  const auto result = registration.align(
    source_cloud_body, voxel_map, Eigen::Isometry3d::Identity());

  const Eigen::Vector3d estimated_translation_error =
    result.transform_world_body.translation() - true_transform_world_body.translation();
  EXPECT_TRUE(result.converged);
  EXPECT_LT(estimated_translation_error.norm(), 0.05);
}

}  // namespace pylot_lio

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
