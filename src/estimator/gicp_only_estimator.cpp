// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#include "pylot_lio/estimator/gicp_only_estimator.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

#include "pylot_lio/lie_algebra.hpp"

namespace pylot_lio
{

namespace
{

// 相対変換 delta_pose を「並進ノルム / 回転角」の双方で上限以内にクランプする。
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

// 相対変換を tangent space で線形スケーリング。
Eigen::Isometry3d scaleRelativeTransform(
  const Eigen::Isometry3d & delta_pose,
  double scale)
{
  Eigen::Isometry3d scaled = Eigen::Isometry3d::Identity();
  scaled.linear() = lie::expSO3(lie::logSO3(delta_pose.linear()) * scale);
  scaled.translation() = delta_pose.translation() * scale;
  return scaled;
}

// SE(3) 上で from から to に向かって「並進 alpha_translation / 回転 alpha_rotation」
// の比率で補間する。 alpha=1 で to 全採用、 0 で from 据え置き。
// 並進は線形、 回転は tangent space で interpolation して expSO3 で SO(3) に戻す。
Eigen::Isometry3d blendPoseTangent(
  const Eigen::Isometry3d & from_pose,
  const Eigen::Isometry3d & to_pose,
  double alpha_translation,
  double alpha_rotation)
{
  const double clamped_alpha_t = std::clamp(alpha_translation, 0.0, 1.0);
  const double clamped_alpha_r = std::clamp(alpha_rotation, 0.0, 1.0);

  // 並進: 線形補間。
  const Eigen::Vector3d blended_translation =
    (1.0 - clamped_alpha_t) * from_pose.translation() +
    clamped_alpha_t * to_pose.translation();

  // 回転: relative rotation を logSO3 で tangent vector 化してスケール。
  const Eigen::Matrix3d relative_rotation =
    from_pose.linear().transpose() * to_pose.linear();
  const Eigen::Vector3d relative_log = lie::logSO3(relative_rotation);
  const Eigen::Matrix3d blended_rotation =
    from_pose.linear() * lie::expSO3(relative_log * clamped_alpha_r);

  Eigen::Isometry3d blended = Eigen::Isometry3d::Identity();
  blended.linear() = lie::normalizeRotation(blended_rotation);
  blended.translation() = blended_translation;
  return blended;
}

// SE(3) delta の「ノルム」を並進と回転に分けて返す。
struct SE3DeltaNorm
{
  double translation_m;
  double rotation_rad;
};

SE3DeltaNorm computeDeltaNorm(const Eigen::Isometry3d & delta_pose)
{
  return {
    delta_pose.translation().norm(),
    lie::logSO3(delta_pose.linear()).norm()};
}

constexpr double kNanosecondsPerSecond = 1.0e9;

}  // namespace

GicpOnlyEstimator::GicpOnlyEstimator(const Config & config)
: config_(config),
  initialized_(false),
  has_previous_pose_(false),
  previous_pose_world_body_(Eigen::Isometry3d::Identity()),
  previous_scan_timestamp_ns_(0),
  previous_dt_s_(0.0),
  has_older_pose_(false),
  older_pose_world_body_(Eigen::Isometry3d::Identity()),
  stationary_streak_count_(0)
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
  has_older_pose_ = false;
  older_pose_world_body_ = Eigen::Isometry3d::Identity();
  stationary_streak_count_ = 0;
  initialized_ = true;
}

bool GicpOnlyEstimator::isInitialized() const
{
  return initialized_;
}

void GicpOnlyEstimator::setPose(const Eigen::Isometry3d & pose_world_body)
{
  // PGO ジャンプ後は previous_pose / velocity / 静止カウンタ / dt をリセットして
  // 「ジャンプ前 pose から外挿される」事故を防ぐ。
  current_state_.pose_world_body = pose_world_body;
  current_state_.velocity_world.setZero();
  previous_pose_world_body_ = pose_world_body;
  has_previous_pose_ = false;
  previous_dt_s_ = 0.0;
  has_older_pose_ = false;
  older_pose_world_body_ = Eigen::Isometry3d::Identity();
  stationary_streak_count_ = 0;
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

  // ---- dt 計測 ----
  double current_dt_s = 0.0;
  if (previous_scan_timestamp_ns_ != 0 && scan_timestamp_ns != 0) {
    const int64_t diff_ns = scan_timestamp_ns - previous_scan_timestamp_ns_;
    if (diff_ns > 0) {
      current_dt_s = static_cast<double>(diff_ns) / kNanosecondsPerSecond;
    }
  }

  // ---- C. 静止検出: 直前 1 スキャン分の運動が閾値以下のスキャンが連続したら
  //                     align をスキップして pose 据え置きにする。 ----
  if (has_previous_pose_ &&
      config_.stationary_translation_threshold_m > 0.0 &&
      config_.stationary_rotation_threshold_rad > 0.0)
  {
    const Eigen::Isometry3d previous_step_delta =
      previous_pose_world_body_.inverse() * current_state_.pose_world_body;
    const auto previous_step_norm = computeDeltaNorm(previous_step_delta);
    if (previous_step_norm.translation_m <= config_.stationary_translation_threshold_m &&
        previous_step_norm.rotation_rad <= config_.stationary_rotation_threshold_rad)
    {
      ++stationary_streak_count_;
    } else {
      stationary_streak_count_ = 0;
    }
  } else {
    stationary_streak_count_ = 0;
  }

  const bool is_stationary =
    config_.stationary_streak_required > 0 &&
    stationary_streak_count_ >= config_.stationary_streak_required;

  if (is_stationary) {
    // pose / previous_pose を据え置き、 velocity をゼロに。 next-extrapolation は
    // 「前回→今回」がゼロ delta なので静止仮定で次スキャンに突入する。
    current_state_.velocity_world.setZero();
    previous_pose_world_body_ = current_state_.pose_world_body;
    has_previous_pose_ = true;
    previous_dt_s_ = current_dt_s;
    previous_scan_timestamp_ns_ = scan_timestamp_ns;
    last_diagnostics_.iterations = 0;
    last_diagnostics_.cost = 0.0;
    last_diagnostics_.converged = true;
    const auto end_time = std::chrono::steady_clock::now();
    last_diagnostics_.processing_time_ms =
      std::chrono::duration<double, std::milli>(end_time - start_time).count();
    return;
  }

  // ---- 等速度モデルによる initial_guess (主候補) ----
  Eigen::Isometry3d velocity_initial_guess = current_state_.pose_world_body;
  if (has_previous_pose_) {
    Eigen::Isometry3d delta_pose =
      previous_pose_world_body_.inverse() * current_state_.pose_world_body;
    if (previous_dt_s_ > 0.0 && current_dt_s > 0.0) {
      delta_pose = scaleRelativeTransform(delta_pose, current_dt_s / previous_dt_s_);
    }
    delta_pose = clampRelativeTransform(
      delta_pose,
      config_.max_extrapolation_translation_m,
      config_.max_extrapolation_rotation_rad);
    velocity_initial_guess = current_state_.pose_world_body * delta_pose;
    velocity_initial_guess.linear() = lie::normalizeRotation(velocity_initial_guess.linear());
  }

  // ---- map 空の初回スキャン: align をスキップ ----
  const bool map_is_empty = (map_world.size() == 0);
  if (map_is_empty) {
    current_state_.pose_world_body = velocity_initial_guess;
    current_state_.velocity_world.setZero();
    previous_pose_world_body_ = current_state_.pose_world_body;
    has_previous_pose_ = false;
    previous_scan_timestamp_ns_ = scan_timestamp_ns;
    last_diagnostics_.iterations = 0;
    last_diagnostics_.cost = 0.0;
    last_diagnostics_.converged = false;
    last_diagnostics_.processing_time_ms = 0.0;
    return;
  }

  // ---- A. 多候補 initial guess ----
  // 候補 1: 等速度外挿。 候補 2 (optional): 静止仮定 (= 現状 pose そのまま)。
  // それぞれ align を叩いて converged かつ final_cost が小さい方を採用。
  IRegistration::AlignResult best_result;
  Eigen::Isometry3d best_initial_guess = velocity_initial_guess;
  best_result.converged = false;
  best_result.final_cost = std::numeric_limits<double>::infinity();

  auto try_candidate = [&](const Eigen::Isometry3d & candidate_guess) {
    const auto result = registration.align(scan_cloud_body, map_world, candidate_guess);
    // converged を優先しつつ、 cost 最小を採用する。
    // 「converged な候補があるなら未収束候補は採らない」「converged 同士なら cost 最小」
    // 「両方未収束なら cost 最小」のルール。
    const bool result_is_better =
      (result.converged && !best_result.converged) ||
      (result.converged == best_result.converged && result.final_cost < best_result.final_cost);
    if (result_is_better) {
      best_result = result;
      best_initial_guess = candidate_guess;
    }
  };

  try_candidate(velocity_initial_guess);
  if (config_.enable_static_candidate && has_previous_pose_) {
    // 静止候補が等速度候補と (ほぼ) 同じなら 2 回目を省略 (無駄計算回避)。
    const Eigen::Isometry3d static_candidate = current_state_.pose_world_body;
    const Eigen::Isometry3d candidate_diff = static_candidate.inverse() * velocity_initial_guess;
    const auto diff_norm = computeDeltaNorm(candidate_diff);
    if (diff_norm.translation_m > 1e-6 || diff_norm.rotation_rad > 1e-6) {
      try_candidate(static_candidate);
    }
  }
  // E. 等加速度候補: 過去 3 つの pose (N-2, N-1, N) から jerk を計算し、
  //                    等速度外挿を 1 ステップ分さらに延ばす。
  //   older_step   = pose_{N-2}⁻¹ * pose_{N-1}
  //   current_step = pose_{N-1}⁻¹ * pose_N
  //   jerk_delta   = older_step⁻¹ * current_step   (1 ステップ前との step 差)
  //   acc_candidate = pose_N * current_step * jerk_delta
  // dt 比は省略 (E は 'もう 1 ステップ持続' の素朴モデル)。 jerk は max_extrapolation_*
  // で clamp して暴走を防ぐ。
  if (config_.enable_acceleration_candidate && has_previous_pose_ && has_older_pose_) {
    const Eigen::Isometry3d older_step =
      older_pose_world_body_.inverse() * previous_pose_world_body_;
    const Eigen::Isometry3d current_step =
      previous_pose_world_body_.inverse() * current_state_.pose_world_body;
    const Eigen::Isometry3d jerk_delta = older_step.inverse() * current_step;
    const Eigen::Isometry3d clamped_jerk = clampRelativeTransform(
      jerk_delta,
      config_.max_extrapolation_translation_m,
      config_.max_extrapolation_rotation_rad);
    Eigen::Isometry3d acc_candidate =
      current_state_.pose_world_body * current_step * clamped_jerk;
    acc_candidate.linear() = lie::normalizeRotation(acc_candidate.linear());
    const Eigen::Isometry3d candidate_diff =
      acc_candidate.inverse() * velocity_initial_guess;
    const auto diff_norm = computeDeltaNorm(candidate_diff);
    if (diff_norm.translation_m > 1e-6 || diff_norm.rotation_rad > 1e-6) {
      try_candidate(acc_candidate);
    }
  }

  // ---- 未収束 → A2 フォールバック ----
  if (!best_result.converged) {
    has_previous_pose_ = false;
    previous_dt_s_ = 0.0;
    previous_scan_timestamp_ns_ = scan_timestamp_ns;
    const auto end_time = std::chrono::steady_clock::now();
    last_diagnostics_.iterations = best_result.iterations;
    last_diagnostics_.cost = best_result.final_cost;
    last_diagnostics_.converged = false;
    last_diagnostics_.processing_time_ms =
      std::chrono::duration<double, std::milli>(end_time - start_time).count();
    return;
  }

  // ---- B. サニティチェック: initial_guess と align 結果の SE(3) 差が大きすぎたら拒否 ----
  Eigen::Isometry3d accepted_pose = best_result.transform_world_body;
  accepted_pose.linear() = lie::normalizeRotation(accepted_pose.linear());

  if (config_.max_correction_translation_m > 0.0 ||
      config_.max_correction_rotation_rad > 0.0)
  {
    const Eigen::Isometry3d correction = best_initial_guess.inverse() * accepted_pose;
    const auto correction_norm = computeDeltaNorm(correction);
    const bool translation_violated =
      config_.max_correction_translation_m > 0.0 &&
      correction_norm.translation_m > config_.max_correction_translation_m;
    const bool rotation_violated =
      config_.max_correction_rotation_rad > 0.0 &&
      correction_norm.rotation_rad > config_.max_correction_rotation_rad;
    if (translation_violated || rotation_violated) {
      // 飛び値とみなして拒否。 initial_guess を pose として据え置き、 align 結果は捨てる。
      accepted_pose = best_initial_guess;
      best_result.converged = false;  // 診断値上は「品質悪」として伝える
    }
  }

  // ---- B'. jerk サニティチェック: 「前回 step (N-1 → N)」 と
  //          「今回 align で得られた step (N → align)」の差が大きすぎたら拒否。
  //          高速回転中の正当な大きな補正と「急変動による誤対応」を区別したい場合に使う。 ----
  if ((config_.max_jerk_translation_m > 0.0 || config_.max_jerk_rotation_rad > 0.0) &&
      has_previous_pose_ && best_result.converged)
  {
    const Eigen::Isometry3d previous_step =
      previous_pose_world_body_.inverse() * current_state_.pose_world_body;
    const Eigen::Isometry3d new_step =
      current_state_.pose_world_body.inverse() * accepted_pose;
    const Eigen::Isometry3d jerk_delta = previous_step.inverse() * new_step;
    const auto jerk_norm = computeDeltaNorm(jerk_delta);
    const bool translation_violated =
      config_.max_jerk_translation_m > 0.0 &&
      jerk_norm.translation_m > config_.max_jerk_translation_m;
    const bool rotation_violated =
      config_.max_jerk_rotation_rad > 0.0 &&
      jerk_norm.rotation_rad > config_.max_jerk_rotation_rad;
    if (translation_violated || rotation_violated) {
      // jerk 過大 → 急変動を疑って等速度外挿を採用。
      accepted_pose = velocity_initial_guess;
      best_result.converged = false;
    }
  }

  // ---- D. EMA 平滑化 ----
  Eigen::Isometry3d filtered_pose = accepted_pose;
  if (has_previous_pose_ &&
      (config_.ema_alpha_translation < 1.0 || config_.ema_alpha_rotation < 1.0))
  {
    filtered_pose = blendPoseTangent(
      current_state_.pose_world_body,
      accepted_pose,
      config_.ema_alpha_translation,
      config_.ema_alpha_rotation);
  }

  // ---- velocity 更新 (filtered_pose 基準) ----
  const Eigen::Isometry3d previous_aligned_pose = current_state_.pose_world_body;
  current_state_.pose_world_body = filtered_pose;
  if (current_dt_s > 0.0) {
    current_state_.velocity_world =
      (current_state_.pose_world_body.translation() - previous_aligned_pose.translation()) /
      current_dt_s;
  } else {
    current_state_.velocity_world.setZero();
  }

  // older_pose 更新: 次スキャンで E (等加速度候補) を生成するために pose 履歴を 1 つ古くする。
  // ただし、 B / B' で拒否された場合 (converged=false) は信頼できない pose 進行なので、
  // 次回 jerk 計算を抑制するため has_older_pose_ をリセットする。
  if (best_result.converged && has_previous_pose_) {
    older_pose_world_body_ = previous_pose_world_body_;
    has_older_pose_ = true;
  } else {
    has_older_pose_ = false;
  }

  previous_pose_world_body_ = previous_aligned_pose;
  has_previous_pose_ = true;
  previous_dt_s_ = current_dt_s;
  previous_scan_timestamp_ns_ = scan_timestamp_ns;

  const auto end_time = std::chrono::steady_clock::now();
  last_diagnostics_.iterations = best_result.iterations;
  last_diagnostics_.cost = best_result.final_cost;
  last_diagnostics_.converged = best_result.converged;
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
