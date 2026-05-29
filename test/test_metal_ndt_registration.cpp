// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#include <gtest/gtest.h>

#include <cmath>

#include <Eigen/Geometry>

#include "pylot_lio/map/voxel_map.hpp"
#include "pylot_lio/registration/metal_ndt_registration.hpp"

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
  map_config.voxel_size_m = 0.05;
  map_config.min_points_per_cell_for_covariance = 1;
  VoxelMap voxel_map(map_config);
  voxel_map.insertScan(cloud, Eigen::Isometry3d::Identity());
  return voxel_map;
}

}  // namespace

// 注意: GPU 経路は gpu_min_points で source 点数が小さければ CPU 参照に自動切替する。
// このテストの corner cloud (約 17000 点 × 3 面) は gpu_min_points=0 にして GPU 経路を
// 強制したいが、 一部のビルドでは Metal が無効。 そこで gpu_min_points=0 にして
// 「GPU 利用可能なら GPU、 そうでなければ CPU 参照」 を通すように指定する。

TEST(MetalNdtRegistration, AlignsIdentityWhenSourceMatchesMap)
{
  const PointCloud corner_cloud = buildCornerCloud();
  VoxelMap voxel_map = buildDenseVoxelMap(corner_cloud);

  MetalNdtRegistration::Config registration_config;
  registration_config.resolution_m = 1.5;
  registration_config.step_size = 1.0;
  registration_config.max_iterations = 50;
  registration_config.gpu_min_points = 0;
  MetalNdtRegistration registration(registration_config);

  const auto result = registration.align(
    corner_cloud, voxel_map, Eigen::Isometry3d::Identity());

  EXPECT_LT(result.transform_world_body.translation().norm(), 0.05);
  const Eigen::AngleAxisd angle_axis(result.transform_world_body.rotation());
  EXPECT_LT(std::abs(angle_axis.angle()), 0.02);
}

TEST(MetalNdtRegistration, RecoversSmallTranslationOffset)
{
  const PointCloud reference_cloud = buildCornerCloud();
  VoxelMap voxel_map = buildDenseVoxelMap(reference_cloud);

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

  MetalNdtRegistration::Config registration_config;
  registration_config.resolution_m = 1.5;
  registration_config.step_size = 1.0;
  registration_config.convergence_translation_m = 1e-5;
  registration_config.convergence_rotation_rad = 1e-5;
  registration_config.max_iterations = 100;
  registration_config.gpu_min_points = 0;
  MetalNdtRegistration registration(registration_config);

  const auto result = registration.align(
    source_cloud_body, voxel_map, Eigen::Isometry3d::Identity());

  const Eigen::Vector3d translation_error =
    result.transform_world_body.translation() - true_transform_world_body.translation();
  // GPU 経路は fp32 のため CPU 参照より精度が落ちる。 0.05 を許容範囲とする。
  EXPECT_LT(translation_error.norm(), 0.05);
}

}  // namespace pylot_lio

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
