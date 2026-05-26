// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#include "pylot_lio/estimator/gicp_only_estimator.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

#include "pylot_lio/lie_algebra.hpp"

namespace pylot_lio
{

namespace
{

// 相対変換 delta_pose を「並進ノルム / 回転角」の双方で上限以内にクランプする。
// 並進と回転で別々の縮小率を計算し、より厳しい方を採用してそれぞれ一様縮小する。
// SE(3) tangent vector を厳密に縮小するわけではないが、初期推定として十分な近似。
Eigen::Isometry3d clampRelativeTransform(
  const Eigen::Isometry3d & delta_pose,
  double max_translation_m,
  double max_rotation_rad)
{
  Eigen::Vector3d translation = delta_pose.translation();
  Eigen::Vector3d rotation_vector = lie::logSO3(delta_pose.linear());

  const double translation_norm = translation.norm();
  const double rotation_norm = rotation_vector.norm();

  double scale = 1.0;
  if (max_translation_m > 0.0 && translation_norm > max_translation_m) {
    scale = std::min(scale, max_translation_m / translation_norm);
  }
  if (max_rotation_rad > 0.0 && rotation_norm > max_rotation_rad) {
    scale = std::min(scale, max_rotation_rad / rotation_norm);
  }

  if (scale >= 1.0) {
    return delta_pose;
  }

  Eigen::Isometry3d clamped = Eigen::Isometry3d::Identity();
  clamped.linear() = lie::expSO3(rotation_vector * scale);
  clamped.translation() = translation * scale;
  return clamped;
}

// 相対変換 delta_pose を「tangent space で線形スケーリング」する。
// 等速度モデルで前回 dt とは異なる今回 dt に合わせて外挿量を補正するのに使う。
Eigen::Isometry3d scaleRelativeTransform(
  const Eigen::Isometry3d & delta_pose,
  double scale)
{
  Eigen::Isometry3d scaled = Eigen::Isometry3d::Identity();
  scaled.linear() = lie::expSO3(lie::logSO3(delta_pose.linear()) * scale);
  scaled.translation() = delta_pose.translation() * scale;
  return scaled;
}

constexpr double kNanosecondsPerSecond = 1.0e9;

}  // namespace

GicpOnlyEstimator::GicpOnlyEstimator(const Config & config)
: config_(config),
  initialized_(false),
  has_previous_pose_(false),
  previous_pose_world_body_(Eigen::Isometry3d::Identity()),
  previous_scan_timestamp_ns_(0),
  previous_dt_s_(0.0)
{
}

void GicpOnlyEstimator::initialize(const RobotState & initial_state)
{
  current_state_ = initial_state;
  current_state_.velocity_world.setZero();
  has_previous_pose_ = false;
  previous_pose_world_body_ = initial_state.pose_world_body;
  previous_scan_timestamp_ns_ = 0;
  previous_dt_s_ = 0.0;
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
  // 速度も連続性を失うのでゼロに戻す (IESKF/HGO ならここで共分散を膨らませる)。
  current_state_.pose_world_body = pose_world_body;
  current_state_.velocity_world.setZero();
  previous_pose_world_body_ = pose_world_body;
  has_previous_pose_ = false;  // 等速度モデルを 1 度リセット (次スキャンで現状から再開)
  previous_dt_s_ = 0.0;
}

void GicpOnlyEstimator::predictWithImu(const ImuSample & /*imu_sample*/)
{
  // GICP-only は IMU を使わない (usesImu() == false なので呼ばれない想定)。
}

void GicpOnlyEstimator::updateWithScan(
  const PointCloud & scan_cloud_body,
  IPointCloudMap & map_world,
  IRegistration & registration,
  int64_t scan_timestamp_ns)
{
  if (!initialized_) {
    return;
  }
  const auto start_time = std::chrono::steady_clock::now();

  // 今回スキャンの dt (前回スキャン取得時刻からの差)。 0 なら未確定として扱う。
  double current_dt_s = 0.0;
  if (previous_scan_timestamp_ns_ != 0 && scan_timestamp_ns != 0) {
    const int64_t diff_ns = scan_timestamp_ns - previous_scan_timestamp_ns_;
    if (diff_ns > 0) {
      current_dt_s = static_cast<double>(diff_ns) / kNanosecondsPerSecond;
    }
  }

  // 等速度モデル: 前回→今回の運動をもう 1 ステップ外挿した姿勢を初期推定にする。
  // 初回 (has_previous_pose_ == false) のみ前回姿勢そのまま。
  Eigen::Isometry3d initial_guess = current_state_.pose_world_body;
  if (has_previous_pose_) {
    Eigen::Isometry3d delta_pose =
      previous_pose_world_body_.inverse() * current_state_.pose_world_body;

    // dt 補正: 前回 dt と今回 dt が両方分かるなら、 delta を比率でスケールする。
    if (previous_dt_s_ > 0.0 && current_dt_s > 0.0) {
      delta_pose = scaleRelativeTransform(delta_pose, current_dt_s / previous_dt_s_);
    }

    // 上限クランプ: 前回スキャンで誤対応した分が遠方へ連鎖するのを防ぐ。
    delta_pose = clampRelativeTransform(
      delta_pose,
      config_.max_extrapolation_translation_m,
      config_.max_extrapolation_rotation_rad);

    initial_guess = current_state_.pose_world_body * delta_pose;
    initial_guess.linear() = lie::normalizeRotation(initial_guess.linear());
  }

  // map が空の初回スキャンでは align しても情報が得られないので明示的にスキップ。
  // (旧実装は registration の暗黙契約に依存していた。)
  const bool map_is_empty = (map_world.size() == 0);
  if (map_is_empty) {
    current_state_.pose_world_body = initial_guess;
    current_state_.velocity_world.setZero();
    // 等速度モデルの起点は作るが、 dt は連鎖しないので previous_dt_s_ はそのまま。
    previous_pose_world_body_ = current_state_.pose_world_body;
    has_previous_pose_ = false;  // map がある状態で再開する
    previous_scan_timestamp_ns_ = scan_timestamp_ns;
    last_diagnostics_.iterations = 0;
    last_diagnostics_.cost = 0.0;
    last_diagnostics_.converged = false;
    last_diagnostics_.processing_time_ms = 0.0;
    return;
  }

  const auto align_result =
    registration.align(scan_cloud_body, map_world, initial_guess);

  // 未収束フォールバック: pose を据え置きにする (initial_guess で書き換えない)。
  // has_previous_pose_=false とすることで「前回→今回」がゼロ delta になり次の
  // 外挿が静止仮定になる事故を避け、次スキャンでは current_pose から素直に再開する。
  if (!align_result.converged) {
    has_previous_pose_ = false;
    previous_dt_s_ = 0.0;
    previous_scan_timestamp_ns_ = scan_timestamp_ns;
    const auto end_time = std::chrono::steady_clock::now();
    last_diagnostics_.iterations = align_result.iterations;
    last_diagnostics_.cost = align_result.final_cost;
    last_diagnostics_.converged = false;
    last_diagnostics_.processing_time_ms =
      std::chrono::duration<double, std::milli>(end_time - start_time).count();
    return;
  }

  // 速度更新 (current_dt_s が確定している時のみ)。
  // pose 更新の前に「前回 align 結果 → 今回 align 結果」の並進差を取る。
  const Eigen::Isometry3d previous_aligned_pose = current_state_.pose_world_body;
  current_state_.pose_world_body = align_result.transform_world_body;
  current_state_.pose_world_body.linear() =
    lie::normalizeRotation(current_state_.pose_world_body.linear());
  if (current_dt_s > 0.0) {
    current_state_.velocity_world =
      (current_state_.pose_world_body.translation() - previous_aligned_pose.translation()) /
      current_dt_s;
  } else {
    current_state_.velocity_world.setZero();
  }

  previous_pose_world_body_ = previous_aligned_pose;
  has_previous_pose_ = true;
  previous_dt_s_ = current_dt_s;
  previous_scan_timestamp_ns_ = scan_timestamp_ns;

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
