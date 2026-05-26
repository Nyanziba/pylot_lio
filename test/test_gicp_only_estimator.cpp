// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#include <gtest/gtest.h>

#include <cstdint>

#include "pylot_lio/estimator/gicp_only_estimator.hpp"
#include "pylot_lio/map/voxel_map.hpp"
#include "pylot_lio/registration/i_registration.hpp"
#include "pylot_lio/registration/plain_gicp_registration.hpp"

namespace pylot_lio
{

namespace
{

// テスト専用の registration スタブ。 align 呼び出しごとに事前設定した戻り値を返す。
// これにより「収束した結果として X を返した」「未収束だった」を制御してテストできる。
class StubRegistration : public IRegistration
{
public:
  AlignResult align(
    const PointCloud & /*source_cloud_body*/,
    const IPointCloudMap & /*map_world*/,
    const Eigen::Isometry3d & initial_transform_world_body) override
  {
    last_initial_guess_ = initial_transform_world_body;
    AlignResult result;
    result.transform_world_body = return_pose_;
    result.iterations = 1;
    result.final_cost = 0.0;
    result.converged = return_converged_;
    return result;
  }

  std::string describe() const override { return "stub"; }

  Eigen::Isometry3d return_pose_ = Eigen::Isometry3d::Identity();
  bool return_converged_ = true;
  Eigen::Isometry3d last_initial_guess_ = Eigen::Isometry3d::Identity();
};

// 1 点だけ入った点群を作って map に挿入し、 size()>0 にする (空マップ判定を外す)。
void primeMapWithOnePoint(VoxelMap & map)
{
  PointCloud cloud;
  Point point;
  point.x = 0.0f;
  point.y = 0.0f;
  point.z = 0.0f;
  cloud.push_back(point);
  map.insertScan(cloud, Eigen::Isometry3d::Identity());
}

constexpr int64_t kOneHundredMillisecondsInNanoseconds = 100'000'000;

}  // namespace

TEST(GicpOnlyEstimator, InitializeAndGetState)
{
  GicpOnlyEstimator estimator;
  EXPECT_FALSE(estimator.isInitialized());
  RobotState initial;
  initial.pose_world_body.translation() = Eigen::Vector3d(0.5, 0.0, 0.0);
  estimator.initialize(initial);
  EXPECT_TRUE(estimator.isInitialized());
  EXPECT_NEAR(estimator.getState().pose_world_body.translation().x(), 0.5, 1e-9);
  EXPECT_NEAR(estimator.getState().velocity_world.norm(), 0.0, 1e-9);
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

TEST(GicpOnlyEstimator, EmptyMapKeepsInitialGuessAndSkipsAlign)
{
  // map が空のときは align をスキップし、 initial_guess (= 現在 pose) を維持する。
  GicpOnlyEstimator estimator;
  estimator.initialize(RobotState{});
  VoxelMap map(VoxelMap::Config{});
  PlainGicpRegistration registration(PlainGicpRegistration::Config{});
  PointCloud empty_cloud;
  estimator.updateWithScan(empty_cloud, map, registration, 0);
  EXPECT_NEAR(estimator.getState().pose_world_body.translation().norm(), 0.0, 1e-9);
  EXPECT_FALSE(estimator.getDiagnostics().converged);
}

TEST(GicpOnlyEstimator, VelocityIsUpdatedFromConsecutiveScanTimestamps)
{
  // 2 スキャン分の align を通すと、 (位置差 / dt) で velocity_world が更新される。
  GicpOnlyEstimator estimator;
  estimator.initialize(RobotState{});

  VoxelMap map(VoxelMap::Config{});
  primeMapWithOnePoint(map);

  StubRegistration registration;
  PointCloud empty_cloud;

  // 1 スキャン目: pose を (0,0,0) のままにする。
  registration.return_pose_ = Eigen::Isometry3d::Identity();
  estimator.updateWithScan(
    empty_cloud, map, registration, kOneHundredMillisecondsInNanoseconds);
  EXPECT_NEAR(estimator.getState().velocity_world.norm(), 0.0, 1e-9);

  // 2 スキャン目: pose を (0.1, 0, 0) に動かす。 dt = 0.1s なので速度は 1 m/s。
  Eigen::Isometry3d second_pose = Eigen::Isometry3d::Identity();
  second_pose.translation() = Eigen::Vector3d(0.1, 0.0, 0.0);
  registration.return_pose_ = second_pose;
  estimator.updateWithScan(
    empty_cloud, map, registration, kOneHundredMillisecondsInNanoseconds * 2);

  EXPECT_NEAR(estimator.getState().velocity_world.x(), 1.0, 1e-6);
  EXPECT_NEAR(estimator.getState().velocity_world.y(), 0.0, 1e-9);
  EXPECT_NEAR(estimator.getState().velocity_world.z(), 0.0, 1e-9);
}

TEST(GicpOnlyEstimator, NonConvergedAlignKeepsPreviousPose)
{
  // align が未収束だったら pose は据え置きで diagnostics.converged=false。
  GicpOnlyEstimator estimator;
  RobotState initial;
  initial.pose_world_body.translation() = Eigen::Vector3d(0.5, 0.0, 0.0);
  estimator.initialize(initial);

  VoxelMap map(VoxelMap::Config{});
  primeMapWithOnePoint(map);

  StubRegistration registration;
  PointCloud empty_cloud;
  Eigen::Isometry3d wrong_pose = Eigen::Isometry3d::Identity();
  wrong_pose.translation() = Eigen::Vector3d(99.0, 99.0, 99.0);  // 飛んだ値
  registration.return_pose_ = wrong_pose;
  registration.return_converged_ = false;

  estimator.updateWithScan(
    empty_cloud, map, registration, kOneHundredMillisecondsInNanoseconds);

  // pose は initialize 時の (0.5, 0, 0) のまま (飛んだ wrong_pose は採用されない)。
  EXPECT_NEAR(estimator.getState().pose_world_body.translation().x(), 0.5, 1e-9);
  EXPECT_NEAR(estimator.getState().pose_world_body.translation().y(), 0.0, 1e-9);
  EXPECT_FALSE(estimator.getDiagnostics().converged);
}

TEST(GicpOnlyEstimator, ExtrapolationDeltaIsClamped)
{
  // 1 スキャン目で大きく飛ばす (例: 5 m) と、 2 スキャン目の initial_guess は
  // 「5m + 等速度で更に 5m」となり 10 m まで暴れるところを、 max_translation_m で
  // クランプして 5m + 1m = 6m に抑える。
  GicpOnlyEstimator::Config config;
  config.max_extrapolation_translation_m = 1.0;
  config.max_extrapolation_rotation_rad = 0.5;
  GicpOnlyEstimator estimator(config);
  estimator.initialize(RobotState{});

  VoxelMap map(VoxelMap::Config{});
  primeMapWithOnePoint(map);

  StubRegistration registration;
  PointCloud empty_cloud;

  // 1 スキャン目: pose を (5, 0, 0) にジャンプ (誤対応相当)。
  Eigen::Isometry3d first_pose = Eigen::Isometry3d::Identity();
  first_pose.translation() = Eigen::Vector3d(5.0, 0.0, 0.0);
  registration.return_pose_ = first_pose;
  estimator.updateWithScan(empty_cloud, map, registration, kOneHundredMillisecondsInNanoseconds);

  // 2 スキャン目: registration は何でも良いが、 initial_guess がクランプされているかを確認。
  registration.return_pose_ = first_pose;  // align 結果は問わない
  estimator.updateWithScan(empty_cloud, map, registration, kOneHundredMillisecondsInNanoseconds * 2);

  // クランプ後の initial_guess は 5m + 1m = 6m になっているはず。
  EXPECT_NEAR(registration.last_initial_guess_.translation().x(), 6.0, 1e-6);
}

TEST(GicpOnlyEstimator, SetPoseResetsExtrapolationAndVelocity)
{
  // PGO ジャンプを setPose で表現。 直後の updateWithScan で等速度モデルが
  // 動かない (= initial_guess == current_pose) ことを確認する。
  GicpOnlyEstimator estimator;
  estimator.initialize(RobotState{});

  VoxelMap map(VoxelMap::Config{});
  primeMapWithOnePoint(map);

  StubRegistration registration;
  PointCloud empty_cloud;

  // 2 スキャン進めて has_previous_pose_=true 状態に。
  Eigen::Isometry3d step1 = Eigen::Isometry3d::Identity();
  step1.translation() = Eigen::Vector3d(0.1, 0.0, 0.0);
  registration.return_pose_ = step1;
  estimator.updateWithScan(empty_cloud, map, registration, kOneHundredMillisecondsInNanoseconds);
  Eigen::Isometry3d step2 = Eigen::Isometry3d::Identity();
  step2.translation() = Eigen::Vector3d(0.2, 0.0, 0.0);
  registration.return_pose_ = step2;
  estimator.updateWithScan(empty_cloud, map, registration, kOneHundredMillisecondsInNanoseconds * 2);

  // PGO 修正で pose を強制的にジャンプ。
  Eigen::Isometry3d corrected = Eigen::Isometry3d::Identity();
  corrected.translation() = Eigen::Vector3d(10.0, 0.0, 0.0);
  estimator.setPose(corrected);
  EXPECT_NEAR(estimator.getState().velocity_world.norm(), 0.0, 1e-9);

  // 次スキャン: initial_guess は corrected pose そのものになる (等速度外挿しない)。
  registration.return_pose_ = corrected;
  estimator.updateWithScan(empty_cloud, map, registration, kOneHundredMillisecondsInNanoseconds * 3);
  EXPECT_NEAR(registration.last_initial_guess_.translation().x(), 10.0, 1e-9);
}

}  // namespace pylot_lio

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
