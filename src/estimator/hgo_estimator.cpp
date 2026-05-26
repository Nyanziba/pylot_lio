// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#include "pylot_lio/estimator/hgo_estimator.hpp"

#include <chrono>
#include <cmath>
#include <sstream>

#include "pylot_lio/lie_algebra.hpp"

namespace pylot_lio
{

HgoEstimator::HgoEstimator(const Config & config)
: config_(config),
  initialized_(false),
  position_world_(Eigen::Vector3d::Zero()),
  velocity_world_(Eigen::Vector3d::Zero()),
  rotation_world_body_(Eigen::Matrix3d::Identity()),
  bias_acc_body_(Eigen::Vector3d::Zero()),
  bias_gyro_body_(Eigen::Vector3d::Zero()),
  gravity_estimation_remaining_(
    config.auto_estimate_gravity ? config.gravity_estimation_samples : 0),
  gravity_estimation_accumulator_(Eigen::Vector3d::Zero())
{
}

void HgoEstimator::initialize(const RobotState & initial_state)
{
  position_world_ = initial_state.pose_world_body.translation();
  rotation_world_body_ = initial_state.pose_world_body.linear();
  velocity_world_ = initial_state.velocity_world;
  bias_acc_body_.setZero();
  bias_gyro_body_.setZero();
  last_imu_timestamp_ns_.reset();
  gravity_estimation_remaining_ =
    config_.auto_estimate_gravity ? config_.gravity_estimation_samples : 0;
  gravity_estimation_accumulator_.setZero();
  initialized_ = true;
}

bool HgoEstimator::isInitialized() const
{
  return initialized_;
}

void HgoEstimator::setPose(const Eigen::Isometry3d & pose_world_body)
{
  // 共分散を持たない HGO は pose だけを上書きすれば十分。
  // バイアスはそのまま (PGO のジャンプはバイアスの問題ではない) 維持し、 速度も維持する
  // (急激な速度リセットは予測を壊す)。
  position_world_ = pose_world_body.translation();
  rotation_world_body_ = pose_world_body.linear();
}

void HgoEstimator::predictWithImu(const ImuSample & imu_sample)
{
  if (!initialized_) {
    return;
  }
  // 重力自動推定 (IESKF と同じ仕組み): 静止 N サンプルから gravity_world を確定。
  if (gravity_estimation_remaining_ > 0) {
    gravity_estimation_accumulator_ += imu_sample.linear_acceleration_mps2;
    --gravity_estimation_remaining_;
    if (gravity_estimation_remaining_ == 0) {
      const Eigen::Vector3d mean_acc =
        gravity_estimation_accumulator_ /
        static_cast<double>(config_.gravity_estimation_samples);
      if (mean_acc.norm() > 1.0) {
        config_.gravity_world = -mean_acc.normalized() * 9.81;
      }
    }
    last_imu_timestamp_ns_ = imu_sample.timestamp_ns;
    return;
  }
  if (!last_imu_timestamp_ns_.has_value()) {
    last_imu_timestamp_ns_ = imu_sample.timestamp_ns;
    return;
  }
  const int64_t delta_ns = imu_sample.timestamp_ns - *last_imu_timestamp_ns_;
  last_imu_timestamp_ns_ = imu_sample.timestamp_ns;
  if (delta_ns <= 0) {
    return;
  }
  const double delta_t_s = static_cast<double>(delta_ns) * 1e-9;
  if (delta_t_s > 0.5) {
    return;
  }

  const Eigen::Vector3d unbiased_acc_body =
    imu_sample.linear_acceleration_mps2 - bias_acc_body_;
  const Eigen::Vector3d unbiased_gyro_body =
    imu_sample.angular_velocity_rps - bias_gyro_body_;

  const Eigen::Vector3d acc_world =
    rotation_world_body_ * unbiased_acc_body + config_.gravity_world;
  position_world_ += velocity_world_ * delta_t_s
    + 0.5 * acc_world * delta_t_s * delta_t_s;
  velocity_world_ += acc_world * delta_t_s;
  rotation_world_body_ = lie::normalizeRotation(
    rotation_world_body_ * lie::expSO3(unbiased_gyro_body * delta_t_s));
}

void HgoEstimator::updateWithScan(
  const PointCloud & scan_cloud_body,
  IPointCloudMap & map_world,
  IRegistration & registration,
  int64_t /*scan_timestamp_ns*/)
{
  if (!initialized_) {
    return;
  }
  const auto start_time = std::chrono::steady_clock::now();

  Eigen::Isometry3d predicted_pose = Eigen::Isometry3d::Identity();
  predicted_pose.linear() = rotation_world_body_;
  predicted_pose.translation() = position_world_;

  const auto align_result = registration.align(scan_cloud_body, map_world, predicted_pose);

  // 観測 - 予測 を SE(3) 接ベクトルで取り、ゲイン K を掛けて反映する。
  const Eigen::Matrix3d delta_rotation_matrix =
    align_result.transform_world_body.linear() * predicted_pose.linear().transpose();
  const Eigen::Vector3d delta_rotation = lie::logSO3(delta_rotation_matrix);
  const Eigen::Vector3d delta_translation =
    align_result.transform_world_body.translation() - predicted_pose.translation();

  rotation_world_body_ = lie::normalizeRotation(
    lie::expSO3(config_.rotation_gain * delta_rotation) * rotation_world_body_);
  position_world_ += config_.translation_gain * delta_translation;
  velocity_world_ += config_.velocity_gain * delta_translation;  // 速度の粗い補正

  // バイアスの推定は省略 (ゲイン 0 でも動作確認はできる)。
  bias_acc_body_ *= (1.0 - config_.bias_acc_gain);
  bias_gyro_body_ *= (1.0 - config_.bias_gyro_gain);

  // map.insertScan は lio_node (keyframe 経由) に責務移譲。
  (void)map_world;

  const auto end_time = std::chrono::steady_clock::now();
  last_diagnostics_.iterations = align_result.iterations;
  last_diagnostics_.cost = align_result.final_cost;
  last_diagnostics_.converged = align_result.converged;
  last_diagnostics_.processing_time_ms =
    std::chrono::duration<double, std::milli>(end_time - start_time).count();
}

RobotState HgoEstimator::getState() const
{
  RobotState state;
  state.pose_world_body.linear() = rotation_world_body_;
  state.pose_world_body.translation() = position_world_;
  state.velocity_world = velocity_world_;
  return state;
}

EstimatorDiagnostics HgoEstimator::getDiagnostics() const
{
  return last_diagnostics_;
}

std::string HgoEstimator::describe() const
{
  std::ostringstream oss;
  oss << "hgo:kt=" << config_.translation_gain
      << ",kr=" << config_.rotation_gain;
  return oss.str();
}

}  // namespace pylot_lio
