// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#include <gtest/gtest.h>

#include <Eigen/Geometry>

#include "pylot_lio/loop/pose_graph_optimizer.hpp"

namespace pylot_lio
{

namespace
{

Eigen::Isometry3d makeTransform(double tx, double ty, double tz, double yaw_rad)
{
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.linear() =
    Eigen::AngleAxisd(yaw_rad, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  pose.translation() = Eigen::Vector3d(tx, ty, tz);
  return pose;
}

PoseGraphOptimizer::NoiseSigmas tightSigmas()
{
  PoseGraphOptimizer::NoiseSigmas sigmas;
  sigmas.rot_x_rad = 0.001;
  sigmas.rot_y_rad = 0.001;
  sigmas.rot_z_rad = 0.001;
  sigmas.trans_x_m = 0.001;
  sigmas.trans_y_m = 0.001;
  sigmas.trans_z_m = 0.001;
  return sigmas;
}

}  // namespace

// 一本鎖 (loop なし) の odometry だけなら、 最適化後の pose は初期推定とほぼ同じ。
TEST(PoseGraphOptimizer, SingleChainConvergesToInitialEstimates)
{
  PoseGraphOptimizer optimizer;

  optimizer.addKeyframePrior(0, makeTransform(0.0, 0.0, 0.0, 0.0));
  optimizer.addKeyframeInitialEstimate(1, makeTransform(1.0, 0.0, 0.0, 0.0));
  optimizer.addOdometryFactor(
    0, 1, makeTransform(1.0, 0.0, 0.0, 0.0), tightSigmas());
  optimizer.addKeyframeInitialEstimate(2, makeTransform(2.0, 0.0, 0.0, 0.0));
  optimizer.addOdometryFactor(
    1, 2, makeTransform(1.0, 0.0, 0.0, 0.0), tightSigmas());

  optimizer.optimize();

  const Eigen::Isometry3d pose1 = optimizer.optimizedPose(1);
  const Eigen::Isometry3d pose2 = optimizer.optimizedPose(2);
  EXPECT_NEAR(pose1.translation().x(), 1.0, 0.01);
  EXPECT_NEAR(pose2.translation().x(), 2.0, 0.01);
  EXPECT_NEAR(pose1.translation().y(), 0.0, 0.01);
  EXPECT_NEAR(pose2.translation().y(), 0.0, 0.01);
}

// odometry の累積ドリフトを loop closure factor で補正できる。
// シナリオ: ロボットが (0,0) → (1,0) → (2,0) → (3,0) と直進したが odometry には誤差 0.5m が
// 乗っていて、 odometry の累積で「初期推定」が (3.5, 0, 0) になっている状況。
// loop closure factor で「KF#3 は KF#0 の +3m の位置にある」 (= 直進で帰ってきた) と伝えると、
// KF#3 の pose は (3, 0, 0) に近づくはず。
TEST(PoseGraphOptimizer, LoopClosureCorrectsDrift)
{
  PoseGraphOptimizer optimizer;

  // 各 keyframe の初期推定 (drift 込み)
  optimizer.addKeyframePrior(0, makeTransform(0.0, 0.0, 0.0, 0.0));
  optimizer.addKeyframeInitialEstimate(1, makeTransform(1.1, 0.0, 0.0, 0.0));
  optimizer.addKeyframeInitialEstimate(2, makeTransform(2.2, 0.0, 0.0, 0.0));
  optimizer.addKeyframeInitialEstimate(3, makeTransform(3.5, 0.0, 0.0, 0.0));

  // odometry: 緩いノイズ (大きな sigma) で各セグメント
  PoseGraphOptimizer::NoiseSigmas loose;
  loose.rot_x_rad = 0.1; loose.rot_y_rad = 0.1; loose.rot_z_rad = 0.1;
  loose.trans_x_m = 0.5; loose.trans_y_m = 0.5; loose.trans_z_m = 0.5;
  optimizer.addOdometryFactor(0, 1, makeTransform(1.1, 0.0, 0.0, 0.0), loose);
  optimizer.addOdometryFactor(1, 2, makeTransform(1.1, 0.0, 0.0, 0.0), loose);
  optimizer.addOdometryFactor(2, 3, makeTransform(1.3, 0.0, 0.0, 0.0), loose);

  // loop closure: KF#3 から見て KF#0 はちょうど (-3, 0, 0) の位置 (= 直進してきた)
  // BetweenFactor(query=3, match=0, T_query_match) で T_world_3 · T_3_0 = T_world_0 が成り立つように。
  // T_3_0 = T_world_3^{-1} · T_world_0 = (-3, 0, 0)
  PoseGraphOptimizer::NoiseSigmas tight = tightSigmas();
  optimizer.addLoopFactor(3, 0, makeTransform(-3.0, 0.0, 0.0, 0.0), tight);

  optimizer.optimize();

  const Eigen::Isometry3d pose3 = optimizer.optimizedPose(3);
  // 補正後は KF#3 が (3, 0, 0) に近づくはず (loop factor の sigma が odometry より tight)
  EXPECT_NEAR(pose3.translation().x(), 3.0, 0.1);
  EXPECT_NEAR(pose3.translation().y(), 0.0, 0.1);
}

// allOptimizedPoses は全 keyframe を含む。 numFactors / numKeyframes も意味のある値。
TEST(PoseGraphOptimizer, ReportsAllOptimizedPoses)
{
  PoseGraphOptimizer optimizer;
  optimizer.addKeyframePrior(0, makeTransform(0.0, 0.0, 0.0, 0.0));
  optimizer.addKeyframeInitialEstimate(1, makeTransform(1.0, 0.0, 0.0, 0.0));
  optimizer.addOdometryFactor(0, 1, makeTransform(1.0, 0.0, 0.0, 0.0), tightSigmas());
  optimizer.optimize();

  const auto all_poses = optimizer.allOptimizedPoses();
  EXPECT_EQ(all_poses.size(), 2u);
  EXPECT_GE(optimizer.numFactors(), 2u);
  EXPECT_GE(optimizer.numKeyframes(), 2u);
}

}  // namespace pylot_lio
