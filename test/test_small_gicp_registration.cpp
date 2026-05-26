// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#ifdef PYLOT_LIO_HAS_SMALL_GICP

#include <gtest/gtest.h>

#include "pylot_lio/map/voxel_map.hpp"
#include "pylot_lio/registration/small_gicp_registration.hpp"

namespace pylot_lio
{

namespace
{
Point makePoint(double x, double y, double z)
{
  Point point;
  point.x = static_cast<float>(x);
  point.y = static_cast<float>(y);
  point.z = static_cast<float>(z);
  point.intensity = 0.0f;
  return point;
}
}  // namespace

TEST(SmallGicpRegistration, AlignsIdentityWhenSourceMatchesMap)
{
  // 同じ点群をマップとソースに与えれば、最適化結果は単位変換に近い。
  VoxelMap::Config map_config;
  map_config.voxel_size_m = 0.5;
  map_config.min_points_per_cell_for_covariance = 1;
  VoxelMap voxel_map(map_config);

  PointCloud plane_cloud;
  for (double x = -3.0; x <= 3.0; x += 0.3) {
    for (double y = -3.0; y <= 3.0; y += 0.3) {
      plane_cloud.push_back(makePoint(x, y, 0.0));
    }
  }
  voxel_map.insertScan(plane_cloud, Eigen::Isometry3d::Identity());

  SmallGicpRegistration::Config registration_config;
  registration_config.variant = SmallGicpRegistration::Variant::VGICP;
  registration_config.num_threads = 1;
  SmallGicpRegistration registration(registration_config);

  const auto result = registration.align(
    plane_cloud, voxel_map, Eigen::Isometry3d::Identity());
  EXPECT_LT(result.transform_world_body.translation().norm(), 0.1);
}

}  // namespace pylot_lio

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

#else  // PYLOT_LIO_HAS_SMALL_GICP

// small_gicp が無い環境では空 main。CMake 側でも条件付き登録だが念のため。
int main(int /*argc*/, char ** /*argv*/) { return 0; }

#endif  // PYLOT_LIO_HAS_SMALL_GICP
