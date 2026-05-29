// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#include <gtest/gtest.h>

#include <cmath>

#include <Eigen/Geometry>

#include "pylot_lio/map/voxel_map.hpp"
#include "pylot_lio/registration/ndt_omp_lite_registration.hpp"

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

// 床 + 壁 2 枚のコーナー形状で 6 自由度を観測可能にする。
PointCloud buildCornerCloud()
{
  PointCloud corner_cloud;
  for (double u = -3.0; u <= 3.0; u += 0.08) {
    for (double v = -3.0; v <= 3.0; v += 0.08) {
      corner_cloud.push_back(makePoint(u, v, 0.0));
      corner_cloud.push_back(makePoint(3.0, u, v + 3.0));
      corner_cloud.push_back(makePoint(u, 3.0, v + 3.0));
    }
  }
  return corner_cloud;
}

VoxelMap buildDenseVoxelMap(const PointCloud & cloud)
{
  VoxelMap::Config map_config;
  // VoxelMap::toPointCloud() は各セルから 1 点しか返さないので、 NDT の内部
  // VoxelGridCovariance (1 voxel あたり最低 6 点要求) が餓死しないように、
  // セルを十分小さく取って実質的に全点を残す。
  map_config.voxel_size_m = 0.05;
  map_config.min_points_per_cell_for_covariance = 1;
  VoxelMap voxel_map(map_config);
  voxel_map.insertScan(cloud, Eigen::Isometry3d::Identity());
  return voxel_map;
}

}  // namespace

TEST(NdtOmpLiteRegistration, AlignsIdentityWhenSourceMatchesMap)
{
  const PointCloud corner_cloud = buildCornerCloud();
  VoxelMap voxel_map = buildDenseVoxelMap(corner_cloud);

  NdtOmpLiteRegistration::Config registration_config;
  registration_config.resolution_m = 1.5;
  registration_config.step_size = 1.0;
  registration_config.max_iterations = 50;
  registration_config.num_threads = 1;
  NdtOmpLiteRegistration registration(registration_config);

  const auto result = registration.align(
    corner_cloud, voxel_map, Eigen::Isometry3d::Identity());

  EXPECT_LT(result.transform_world_body.translation().norm(), 0.05);
  const Eigen::AngleAxisd angle_axis(result.transform_world_body.rotation());
  EXPECT_LT(std::abs(angle_axis.angle()), 0.02);
}

TEST(NdtOmpLiteRegistration, RecoversSmallTranslationOffset)
{
  const PointCloud reference_cloud = buildCornerCloud();
  VoxelMap voxel_map = buildDenseVoxelMap(reference_cloud);

  // 真の姿勢: 並進 (0.15, -0.1, 0.05) m。 source は world 点を逆変換した body 点群。
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

  NdtOmpLiteRegistration::Config registration_config;
  registration_config.resolution_m = 1.5;
  registration_config.step_size = 1.0;
  registration_config.transformation_epsilon_m = 1e-5;
  registration_config.rotation_epsilon_rad = 1e-5;
  registration_config.max_iterations = 100;
  registration_config.num_threads = 1;
  NdtOmpLiteRegistration registration(registration_config);

  const auto result = registration.align(
    source_cloud_body, voxel_map, Eigen::Isometry3d::Identity());

  const Eigen::Vector3d translation_error =
    result.transform_world_body.translation() - true_transform_world_body.translation();
  EXPECT_LT(translation_error.norm(), 0.05);
}

}  // namespace pylot_lio

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
