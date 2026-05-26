// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#include "pylot_lio/estimator/gicp_only_estimator.hpp"

#include <chrono>
#include <cmath>
#include <sstream>

#include "pylot_lio/lie_algebra.hpp"

namespace pylot_lio
{

GicpOnlyEstimator::GicpOnlyEstimator()
: initialized_(false),
  has_previous_pose_(false),
  previous_pose_world_body_(Eigen::Isometry3d::Identity())
{
}

void GicpOnlyEstimator::initialize(const RobotState & initial_state)
{
  current_state_ = initial_state;
  has_previous_pose_ = false;
  previous_pose_world_body_ = initial_state.pose_world_body;
  initialized_ = true;
}

bool GicpOnlyEstimator::isInitialized() const
{
  return initialized_;
}

void GicpOnlyEstimator::setPose(const Eigen::Isometry3d & pose_world_body)
{
  // GICP-only は内部状態がほぼ姿勢のみ。 PGO ジャンプ後は previous_pose も更新して
  // 「次の等速度モデルが過去のジャンプ前 pose から外挿される」事故を防ぐ。
  current_state_.pose_world_body = pose_world_body;
  previous_pose_world_body_ = pose_world_body;
  has_previous_pose_ = false;  // 等速度モデルを 1 度リセット (次スキャンで現状から再開)
}

void GicpOnlyEstimator::predictWithImu(const ImuSample & /*imu_sample*/)
{
  // GICP-only は IMU を使わない。
}

void GicpOnlyEstimator::updateWithScan(
  const PointCloud & scan_cloud_body,
  IPointCloudMap & map_world,
  IRegistration & registration,
  int64_t /*scan_timestamp_ns*/)
{
  if (!initialized_) {
    return;
  }
  const auto start_time = std::chrono::steady_clock::now();

  // 等速度モデル: 前回→今回の運動をもう 1 ステップ外挿した姿勢を初期推定にする。
  // 初回 (has_previous_pose_ == false) のみ前回姿勢そのまま。
  Eigen::Isometry3d initial_guess = current_state_.pose_world_body;
  if (has_previous_pose_) {
    const Eigen::Isometry3d delta =
      previous_pose_world_body_.inverse() * current_state_.pose_world_body;
    initial_guess = current_state_.pose_world_body * delta;
    initial_guess.linear() = lie::normalizeRotation(initial_guess.linear());
  }

  const auto align_result =
    registration.align(scan_cloud_body, map_world, initial_guess);
  previous_pose_world_body_ = current_state_.pose_world_body;
  has_previous_pose_ = true;
  current_state_.pose_world_body = align_result.transform_world_body;
  current_state_.pose_world_body.linear() =
    lie::normalizeRotation(current_state_.pose_world_body.linear());

  // 注: scan を map に挿入する責務は lio_node (keyframe ロジック経由) に移譲したため、
  // ここでは何もしない。registration の初回は map が空なので align は initial_guess を返す。
  (void)map_world;  // 引数として保持 (IESKF/HGO は使う)。

  const auto end_time = std::chrono::steady_clock::now();
  last_diagnostics_.iterations = align_result.iterations;
  last_diagnostics_.cost = align_result.final_cost;
  last_diagnostics_.converged = align_result.converged;
  last_diagnostics_.processing_time_ms =
    std::chrono::duration<double, std::milli>(end_time - start_time).count();
}

RobotState GicpOnlyEstimator::getState() const
{
  return current_state_;
}

EstimatorDiagnostics GicpOnlyEstimator::getDiagnostics() const
{
  return last_diagnostics_;
}

std::string GicpOnlyEstimator::describe() const
{
  return "gicp_only";
}

}  // namespace pylot_lio
