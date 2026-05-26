// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#include <gtest/gtest.h>

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

#include "pylot_lio/map/voxel_keyframe_submap_map.hpp"

namespace pylot_lio
{

namespace
{

// 1 キーフレーム分の点群を、指定位置中心の小さな立方体クラスタとして作る。
PointCloud makeKeyframeCloud(double center_x, double center_y, double center_z)
{
  PointCloud cloud;
  for (int dx = -1; dx <= 1; ++dx) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dz = -1; dz <= 1; ++dz) {
        Point sample;
        sample.x = static_cast<float>(center_x + 0.05 * dx);
        sample.y = static_cast<float>(center_y + 0.05 * dy);
        sample.z = static_cast<float>(center_z + 0.05 * dz);
        sample.intensity = 1.0f;
        cloud.points.push_back(sample);
      }
    }
  }
  cloud.width = static_cast<uint32_t>(cloud.points.size());
  cloud.height = 1;
  cloud.is_dense = true;
  return cloud;
}

Eigen::Isometry3d poseAt(double tx, double ty, double tz)
{
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.translation() = Eigen::Vector3d(tx, ty, tz);
  return pose;
}

VoxelKeyframeSubmapManager::Config smallConfig()
{
  VoxelKeyframeSubmapManager::Config config;
  config.voxel_size_m = 0.5;
  config.sliding_window_size = 3;
  config.finalize_size = 3;
  config.max_points_per_cell = 64;
  config.neighbor_search_radius_voxels = 1;
  config.min_points_per_cell_for_search = 1;
  return config;
}

}  // namespace

// 古いキーフレームは sliding_window_size を超えると grid から退場する。
// 退場直後、最古キーフレームの位置の近傍探索結果は無効か遠い点になる。
TEST(VoxelKeyframeSubmapManager, OldestKeyframeIsEvictedWhenWindowOverflows)
{
  VoxelKeyframeSubmapManager manager(smallConfig());

  // 互いに離れた 4 個のキーフレームを挿入。window=3 なので 4 個目挿入時に最古が落ちる。
  manager.insertScan(makeKeyframeCloud(0.0, 0.0, 0.0), poseAt(0.0, 0.0, 0.0));
  manager.insertScan(makeKeyframeCloud(50.0, 0.0, 0.0), poseAt(50.0, 0.0, 0.0));
  manager.insertScan(makeKeyframeCloud(100.0, 0.0, 0.0), poseAt(100.0, 0.0, 0.0));
  EXPECT_EQ(manager.activeKeyframeCount(), 3u);

  manager.insertScan(makeKeyframeCloud(150.0, 0.0, 0.0), poseAt(150.0, 0.0, 0.0));
  EXPECT_EQ(manager.activeKeyframeCount(), 3u);
  EXPECT_EQ(manager.totalKeyframesInserted(), 4u);

  // 退場した KF#0 の位置 (0,0,0) で検索すると、近傍 grid セルが消えているので無効。
  PointCorrespondence correspondence =
    manager.findNearestNeighbor(Eigen::Vector3d(0.0, 0.0, 0.0));
  EXPECT_FALSE(correspondence.valid);

  // 一方、最新 KF#3 の中心では有効。
  PointCorrespondence correspondence_recent =
    manager.findNearestNeighbor(Eigen::Vector3d(150.0, 0.0, 0.0));
  EXPECT_TRUE(correspondence_recent.valid);
}

// finalize_size 個ごとに finalized_submaps が 1 個ずつ増える。
TEST(VoxelKeyframeSubmapManager, FinalizeSnapshotEveryMKeyframes)
{
  VoxelKeyframeSubmapManager manager(smallConfig());

  manager.insertScan(makeKeyframeCloud(0.0, 0.0, 0.0), poseAt(0.0, 0.0, 0.0));
  EXPECT_EQ(manager.finalizedSubmapCount(), 0u);

  manager.insertScan(makeKeyframeCloud(1.0, 0.0, 0.0), poseAt(1.0, 0.0, 0.0));
  EXPECT_EQ(manager.finalizedSubmapCount(), 0u);

  manager.insertScan(makeKeyframeCloud(2.0, 0.0, 0.0), poseAt(2.0, 0.0, 0.0));
  EXPECT_EQ(manager.finalizedSubmapCount(), 1u);

  // 3 個ずつでもう 1 個確定するはず (合計 6 個挿入 → 2 個)
  manager.insertScan(makeKeyframeCloud(3.0, 0.0, 0.0), poseAt(3.0, 0.0, 0.0));
  manager.insertScan(makeKeyframeCloud(4.0, 0.0, 0.0), poseAt(4.0, 0.0, 0.0));
  manager.insertScan(makeKeyframeCloud(5.0, 0.0, 0.0), poseAt(5.0, 0.0, 0.0));
  EXPECT_EQ(manager.finalizedSubmapCount(), 2u);

  const auto & finalized = manager.finalizedSubmaps();
  ASSERT_EQ(finalized.size(), 2u);
  EXPECT_EQ(finalized[0].keyframe_ids.size(), 3u);
  EXPECT_EQ(finalized[0].keyframe_ids.front(), 0u);
  EXPECT_EQ(finalized[0].keyframe_ids.back(), 2u);
  EXPECT_EQ(finalized[1].keyframe_ids.front(), 3u);
  EXPECT_EQ(finalized[1].keyframe_ids.back(), 5u);
  // anchor は range 末尾の姿勢
  EXPECT_NEAR(finalized[1].anchor_pose_world_body.translation().x(), 5.0, 1e-9);
  // merged cloud は 3 keyframe 分の点群を結合 (27 points × 3)
  EXPECT_EQ(finalized[1].merged_cloud_world.points.size(), 27u * 3u);
}

// findNearestNeighbor は最新キーフレームの点群に対し、クエリ位置の最近傍を返す。
TEST(VoxelKeyframeSubmapManager, NearestNeighborReturnsValidCorrespondence)
{
  VoxelKeyframeSubmapManager manager(smallConfig());
  manager.insertScan(makeKeyframeCloud(0.0, 0.0, 0.0), poseAt(0.0, 0.0, 0.0));

  // 中心からほんの少しずれた位置からクエリ。最近傍はクラスタ内のいずれかの点。
  PointCorrespondence correspondence =
    manager.findNearestNeighbor(Eigen::Vector3d(0.02, 0.0, 0.0));
  ASSERT_TRUE(correspondence.valid);
  EXPECT_LT(correspondence.squared_distance, 0.01);
  // 共分散は対称・正定値であるべき (最小固有値 floor 適用後)
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(correspondence.target_covariance);
  EXPECT_GT(solver.eigenvalues().minCoeff(), 0.0);
}

}  // namespace pylot_lio
