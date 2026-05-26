// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
//
// IESKF (Fast-LIO2 / Super-LIO 流) リライト版。
//
// 旧実装との主な違い:
//   - 重力: gravity_norm を固定値として持ち、 起動時 IMU 平均から方向だけ取って
//     gravity_world = -unit(mean_acc) * gravity_norm を採用。 g の大きさに IMU
//     ノイズ/単位誤差が乗らなくなる (Super-LIO lio.sensor.gravity_norm=9.7946 と整合)。
//   - LiDAR ↔ IMU extrinsic を Config で受け、 update 内で LiDAR 点を IMU 中心系へ
//     持ち上げてから world 投影する。 nominal_position_world_ は IMU 中心位置を表す。
//   - 走行中の重力更新はしない (Super-LIO 流)。
//
// 観測モデル: point-to-distribution (GICP 流)。 各 LiDAR 点について
//   r = target - R · (R_il · p_lidar + t_il) - p
// を最小化する Gauss-Newton 反復を SE(3) 左摂動で実装。
#include "pylot_lio/estimator/ieskf_estimator.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <sstream>

#include <Eigen/Eigenvalues>

#include "pylot_lio/lie_algebra.hpp"

namespace pylot_lio
{

IeskfEstimator::IeskfEstimator(const Config & config)
: config_(config),
  initialized_(false),
  gravity_estimation_remaining_(
    config.auto_estimate_gravity ? config.gravity_estimation_samples : 0),
  gravity_estimation_accumulator_(Eigen::Vector3d::Zero()),
  gravity_world_active_(config.gravity_world_init),
  nominal_position_world_(Eigen::Vector3d::Zero()),
  nominal_velocity_world_(Eigen::Vector3d::Zero()),
  nominal_rotation_world_body_(Eigen::Matrix3d::Identity()),
  bias_acc_body_(Eigen::Vector3d::Zero()),
  bias_gyro_body_(Eigen::Vector3d::Zero()),
  error_covariance_(Eigen::Matrix<double, kStateDim, kStateDim>::Identity() * 1e-3)
{
}

void IeskfEstimator::initialize(const RobotState & initial_state)
{
  nominal_position_world_ = initial_state.pose_world_body.translation();
  nominal_rotation_world_body_ = initial_state.pose_world_body.linear();
  nominal_velocity_world_ = initial_state.velocity_world;
  bias_acc_body_.setZero();
  bias_gyro_body_.setZero();
  error_covariance_.setIdentity();
  error_covariance_ *= 1e-3;
  last_imu_timestamp_ns_.reset();
  gravity_estimation_remaining_ =
    config_.auto_estimate_gravity ? config_.gravity_estimation_samples : 0;
  gravity_estimation_accumulator_.setZero();
  gravity_world_active_ = config_.gravity_world_init;
  initialized_ = true;
}

bool IeskfEstimator::isInitialized() const
{
  return initialized_;
}

void IeskfEstimator::setPose(const Eigen::Isometry3d & pose_world_body)
{
  nominal_position_world_ = pose_world_body.translation();
  nominal_rotation_world_body_ = pose_world_body.linear();
  error_covariance_.setIdentity();
  error_covariance_ *= 1e-3;
}

void IeskfEstimator::predictWithImu(const ImuSample & imu_sample)
{
  if (!initialized_) {
    return;
  }

  // 重力推定: N サンプル静止平均から「方向」だけ取り、 大きさは config_.gravity_norm 固定。
  //   gravity_world = -unit(mean_acc) * gravity_norm
  // 起動時 IMU の生 acc の単位や符号誤差が重力大きさに影響しなくなるのが要点。
  // (旧実装は -unit(mean_acc) * 9.81 で「ハードコード 9.81」だった。)
  if (gravity_estimation_remaining_ > 0) {
    gravity_estimation_accumulator_ += imu_sample.linear_acceleration_mps2;
    --gravity_estimation_remaining_;
    if (gravity_estimation_remaining_ == 0) {
      const Eigen::Vector3d mean_acc =
        gravity_estimation_accumulator_ /
        static_cast<double>(config_.gravity_estimation_samples);
      const double acc_norm = mean_acc.norm();
      // 異常に小さい acc は静止と見なせない (例: bag 録画時に既に動いていた)。
      // この場合は config_.gravity_world_init を初期値として残す。
      if (acc_norm > 1.0) {
        gravity_world_active_ = -mean_acc.normalized() * config_.gravity_norm;
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
    // IMU が長時間止まっていた場合は予測ステップをスキップ (異常値防止)。
    return;
  }

  const Eigen::Vector3d unbiased_acc_body =
    imu_sample.linear_acceleration_mps2 - bias_acc_body_;
  const Eigen::Vector3d unbiased_gyro_body =
    imu_sample.angular_velocity_rps - bias_gyro_body_;

  // 名目状態の積分 (Euler 法; 必要なら midpoint に拡張可能)。
  const Eigen::Vector3d acc_world =
    nominal_rotation_world_body_ * unbiased_acc_body + gravity_world_active_;
  const Eigen::Vector3d new_position =
    nominal_position_world_
    + nominal_velocity_world_ * delta_t_s
    + 0.5 * acc_world * delta_t_s * delta_t_s;
  const Eigen::Vector3d new_velocity =
    nominal_velocity_world_ + acc_world * delta_t_s;
  const Eigen::Matrix3d new_rotation =
    nominal_rotation_world_body_ * lie::expSO3(unbiased_gyro_body * delta_t_s);

  // 誤差状態の遷移行列 F (連続時間モデルの離散化、1次近似)。
  Eigen::Matrix<double, kStateDim, kStateDim> transition_matrix =
    Eigen::Matrix<double, kStateDim, kStateDim>::Identity();
  transition_matrix.block<3, 3>(kIdxPos, kIdxVel) =
    Eigen::Matrix3d::Identity() * delta_t_s;
  transition_matrix.block<3, 3>(kIdxVel, kIdxOri) =
    -nominal_rotation_world_body_ * lie::skew(unbiased_acc_body) * delta_t_s;
  transition_matrix.block<3, 3>(kIdxVel, kIdxBiasAcc) =
    -nominal_rotation_world_body_ * delta_t_s;
  transition_matrix.block<3, 3>(kIdxOri, kIdxOri) =
    lie::expSO3(-unbiased_gyro_body * delta_t_s);
  transition_matrix.block<3, 3>(kIdxOri, kIdxBiasGyro) =
    -Eigen::Matrix3d::Identity() * delta_t_s;

  Eigen::Matrix<double, kStateDim, kStateDim> process_noise =
    Eigen::Matrix<double, kStateDim, kStateDim>::Zero();
  const double sigma_acc_squared =
    config_.imu_acc_noise_density * config_.imu_acc_noise_density * delta_t_s;
  const double sigma_gyro_squared =
    config_.imu_gyro_noise_density * config_.imu_gyro_noise_density * delta_t_s;
  const double sigma_acc_bias_squared =
    config_.imu_acc_bias_random_walk * config_.imu_acc_bias_random_walk * delta_t_s;
  const double sigma_gyro_bias_squared =
    config_.imu_gyro_bias_random_walk * config_.imu_gyro_bias_random_walk * delta_t_s;

  process_noise.block<3, 3>(kIdxVel, kIdxVel) =
    Eigen::Matrix3d::Identity() * sigma_acc_squared;
  process_noise.block<3, 3>(kIdxOri, kIdxOri) =
    Eigen::Matrix3d::Identity() * sigma_gyro_squared;
  process_noise.block<3, 3>(kIdxBiasAcc, kIdxBiasAcc) =
    Eigen::Matrix3d::Identity() * sigma_acc_bias_squared;
  process_noise.block<3, 3>(kIdxBiasGyro, kIdxBiasGyro) =
    Eigen::Matrix3d::Identity() * sigma_gyro_bias_squared;

  error_covariance_ = transition_matrix * error_covariance_ * transition_matrix.transpose()
    + process_noise;

  nominal_position_world_ = new_position;
  nominal_velocity_world_ = new_velocity;
  nominal_rotation_world_body_ = lie::normalizeRotation(new_rotation);
}

void IeskfEstimator::updateWithScan(
  const PointCloud & scan_cloud_body,
  IPointCloudMap & map_world,
  IRegistration & /*registration*/,
  int64_t /*scan_timestamp_ns*/)
{
  // Full 15D IESKF update (Fast-LIO2 流):
  //   1) Gauss-Newton で scan による pose 観測 (δθ_obs, δt_obs) を求める
  //   2) これを「pose 観測値」として全 15 次元の error state を Kalman update
  //   3) δx を nominal state に inject (R は exp(δθ) 左乗、 他は加算)
  //   4) P を Joseph 形式で事後更新 → bias の不確実性を維持しつつ pose の確信度は上げる
  //
  // この設計の肝は (2)。 P の predict step で生まれる pose と bias のクロス項
  // (P_pos_bg, P_ori_ba 等) を介して、 pose の補正が bias にも伝播する。
  // ここを実装しないと bias_acc / bias_gyro が永遠に 0 のままで、 IMU の重力誤差が
  // 走行中ずっと z 方向に積分される (= 観測症状)。
  if (!initialized_) {
    return;
  }

  const auto start_time = std::chrono::steady_clock::now();
  const double max_correspondence_distance_squared =
    config_.max_correspondence_distance_m * config_.max_correspondence_distance_m;

  const Eigen::Matrix3d & rotation_imu_from_lidar =
    config_.extrinsic_rotation_imu_from_lidar;
  const Eigen::Vector3d & translation_imu_from_lidar =
    config_.extrinsic_translation_imu_from_lidar;

  Eigen::Vector3d position_estimate = nominal_position_world_;
  Eigen::Matrix3d rotation_estimate = nominal_rotation_world_body_;

  int iteration_index = 0;
  double final_cost = 0.0;
  bool converged = false;
  int last_valid_correspondences = 0;

  for (; iteration_index < config_.max_iteration_per_scan; ++iteration_index) {
    Eigen::Matrix<double, 6, 6> hessian_pose = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> gradient_pose = Eigen::Matrix<double, 6, 1>::Zero();
    final_cost = 0.0;
    last_valid_correspondences = 0;

    for (const Point & source_point_lidar : scan_cloud_body.points) {
      if (!std::isfinite(source_point_lidar.x) ||
          !std::isfinite(source_point_lidar.y) ||
          !std::isfinite(source_point_lidar.z))
      {
        continue;
      }
      // LiDAR frame → IMU frame
      const Eigen::Vector3d source_in_lidar(
        source_point_lidar.x, source_point_lidar.y, source_point_lidar.z);
      const Eigen::Vector3d source_in_imu =
        rotation_imu_from_lidar * source_in_lidar + translation_imu_from_lidar;
      // IMU frame → world frame (nominal_position_world_ は IMU 中心の世界座標)
      const Eigen::Vector3d source_in_world =
        rotation_estimate * source_in_imu + position_estimate;

      const PointCorrespondence correspondence =
        map_world.findNearestNeighbor(source_in_world);
      if (!correspondence.valid ||
          correspondence.squared_distance > max_correspondence_distance_squared)
      {
        continue;
      }

      Eigen::Matrix3d information_matrix;
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(correspondence.target_covariance);
      Eigen::Vector3d eigenvalues = solver.eigenvalues();
      for (int axis = 0; axis < 3; ++axis) {
        eigenvalues(axis) = 1.0 / std::max(eigenvalues(axis), 1e-9);
      }
      information_matrix = solver.eigenvectors() * eigenvalues.asDiagonal()
        * solver.eigenvectors().transpose();

      const Eigen::Vector3d residual_world =
        correspondence.target_point_world - source_in_world;

      // SE(3) 左摂動の Jacobian。 source_in_world 全体で skew を取る。
      Eigen::Matrix<double, 3, 6> jacobian_matrix;
      jacobian_matrix.block<3, 3>(0, 0) = lie::skew(source_in_world);
      jacobian_matrix.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();

      hessian_pose.noalias() +=
        jacobian_matrix.transpose() * information_matrix * jacobian_matrix;
      gradient_pose.noalias() -=
        jacobian_matrix.transpose() * information_matrix * residual_world;
      final_cost += residual_world.transpose() * information_matrix * residual_world;
      ++last_valid_correspondences;
    }

    if (last_valid_correspondences < 6) {
      break;
    }

    // Pose 共分散 (6x6) を error_covariance から [orientation, translation] の順で
    // 抽出する。 Jacobian の列順 ([δθ; δt]) と合わせるためで、 ここを間違えると
    // 「速度の共分散を pose prior として使う」典型バグになり、 scan update が IMU の
    // 重力誤差を全く補正できず z drift が単調に積もる。
    Eigen::Matrix<double, 6, 6> pose_covariance;
    pose_covariance.block<3, 3>(0, 0) =
      error_covariance_.block<3, 3>(kIdxOri, kIdxOri);
    pose_covariance.block<3, 3>(0, 3) =
      error_covariance_.block<3, 3>(kIdxOri, kIdxPos);
    pose_covariance.block<3, 3>(3, 0) =
      error_covariance_.block<3, 3>(kIdxPos, kIdxOri);
    pose_covariance.block<3, 3>(3, 3) =
      error_covariance_.block<3, 3>(kIdxPos, kIdxPos);
    const Eigen::Matrix<double, 6, 6> prior_information = pose_covariance.inverse();
    hessian_pose += prior_information;

    const Eigen::Matrix<double, 6, 1> delta_pose =
      hessian_pose.ldlt().solve(gradient_pose);

    const Eigen::Vector3d delta_rotation = delta_pose.head<3>();
    const Eigen::Vector3d delta_translation = delta_pose.tail<3>();

    // SE(3) 左摂動の更新: T_new = delta_T * T_old
    const Eigen::Matrix3d delta_rotation_matrix = lie::expSO3(delta_rotation);
    position_estimate = delta_rotation_matrix * position_estimate + delta_translation;
    rotation_estimate = lie::normalizeRotation(delta_rotation_matrix * rotation_estimate);

    if (delta_translation.norm() < config_.convergence_translation_m &&
        delta_rotation.norm() < config_.convergence_rotation_rad)
    {
      converged = true;
      ++iteration_index;
      break;
    }
  }

  // ============================================================
  // ここから Full 15D Kalman update。
  // Gauss-Newton で求めた (position_estimate, rotation_estimate) を「scan 観測」とし、
  // 観測残差 y を nominal state に対する補正量として表現する:
  //   y_rot = log(R_obs · R_nom^T)
  //   y_pos = position_estimate - R_correction · nominal_position_world_
  //          ( = SE(3) 左摂動 ΔT = T_obs * T_nom^{-1} の並進成分)
  // 観測モデル H (6x15): δθ ← state.kIdxOri, δt ← state.kIdxPos の I_3。
  // 観測ノイズ R = diag(σ_rot², σ_t²) で、 scan の信頼度を表す。
  // ============================================================
  if (last_valid_correspondences >= 6) {
    // 観測残差 y は error state δx と同じ parameterization で作る必要がある。
    // ここでの error state 定義:
    //   p_new = p_nom + δp        (並進は単純加算)
    //   R_new = exp(δθ) · R_nom   (回転は左摂動 tangent space)
    // よって観測残差も同じ:
    //   y_rot = log(R_obs · R_nom^T)
    //   y_pos = p_obs - p_nom              ← 単純差分 (SE(3) 左摂動 δt ではない)
    //
    // 旧実装は y_pos = p_obs - R_correction · p_nom と SE(3) 左摂動の δt を入れて
    // いたため、 nominal_position が大きくなると (exp(δθ)-I)·p_nom の項が観測残差
    // に偽の lever arm として混入し、 斜め drift を引き起こしていた。
    const Eigen::Matrix3d rotation_correction =
      rotation_estimate * nominal_rotation_world_body_.transpose();
    Eigen::Matrix<double, 6, 1> observation_residual;
    observation_residual.head<3>() = lie::logSO3(rotation_correction);
    observation_residual.tail<3>() = position_estimate - nominal_position_world_;

    // H: rows = (δθ_obs, δt_obs)、 cols = 15D state
    Eigen::Matrix<double, 6, kStateDim> observation_jacobian =
      Eigen::Matrix<double, 6, kStateDim>::Zero();
    observation_jacobian.block<3, 3>(0, kIdxOri) = Eigen::Matrix3d::Identity();
    observation_jacobian.block<3, 3>(3, kIdxPos) = Eigen::Matrix3d::Identity();

    // R: 観測ノイズ共分散 (対角)
    Eigen::Matrix<double, 6, 6> observation_noise =
      Eigen::Matrix<double, 6, 6>::Zero();
    const double sigma_rot_squared =
      config_.scan_observation_noise_rotation_rad *
      config_.scan_observation_noise_rotation_rad;
    const double sigma_trans_squared =
      config_.scan_observation_noise_translation_m *
      config_.scan_observation_noise_translation_m;
    observation_noise.block<3, 3>(0, 0) =
      Eigen::Matrix3d::Identity() * sigma_rot_squared;
    observation_noise.block<3, 3>(3, 3) =
      Eigen::Matrix3d::Identity() * sigma_trans_squared;

    // S = H P H^T + R
    const Eigen::Matrix<double, 6, 6> innovation_covariance =
      observation_jacobian * error_covariance_ * observation_jacobian.transpose() +
      observation_noise;

    // K = P H^T S^{-1}
    const Eigen::Matrix<double, kStateDim, 6> kalman_gain =
      error_covariance_ * observation_jacobian.transpose() *
      innovation_covariance.inverse();

    // δx = K * y (15D 補正量)
    const Eigen::Matrix<double, kStateDim, 1> delta_state =
      kalman_gain * observation_residual;

    // nominal state へ inject。 順序: kIdxPos, kIdxVel, kIdxOri, kIdxBiasAcc, kIdxBiasGyro
    nominal_position_world_ += delta_state.segment<3>(kIdxPos);
    nominal_velocity_world_ += delta_state.segment<3>(kIdxVel);
    const Eigen::Vector3d delta_orientation = delta_state.segment<3>(kIdxOri);
    nominal_rotation_world_body_ = lie::normalizeRotation(
      lie::expSO3(delta_orientation) * nominal_rotation_world_body_);
    bias_acc_body_ += delta_state.segment<3>(kIdxBiasAcc);
    bias_gyro_body_ += delta_state.segment<3>(kIdxBiasGyro);

    // P の事後更新 (Joseph 形式)。 数値安定で、 共分散行列が対称・正定値を維持する。
    //   P = (I - K H) P (I - K H)^T + K R K^T
    const Eigen::Matrix<double, kStateDim, kStateDim> identity_minus_kh =
      Eigen::Matrix<double, kStateDim, kStateDim>::Identity() -
      kalman_gain * observation_jacobian;
    error_covariance_ =
      identity_minus_kh * error_covariance_ * identity_minus_kh.transpose() +
      kalman_gain * observation_noise * kalman_gain.transpose();
  } else {
    // Scan 観測が取れなかった (廊下端や障害物面が極端に少ない場合) は state を触らず
    // P もそのまま (predict で増えた状態を保つ)。
  }

  (void)map_world;

  const auto end_time = std::chrono::steady_clock::now();
  last_diagnostics_.iterations = iteration_index;
  last_diagnostics_.cost = final_cost;
  last_diagnostics_.converged = converged;
  last_diagnostics_.processing_time_ms =
    std::chrono::duration<double, std::milli>(end_time - start_time).count();
}

RobotState IeskfEstimator::getState() const
{
  RobotState state;
  state.pose_world_body.linear() = nominal_rotation_world_body_;
  state.pose_world_body.translation() = nominal_position_world_;
  state.velocity_world = nominal_velocity_world_;
  return state;
}

EstimatorDiagnostics IeskfEstimator::getDiagnostics() const
{
  return last_diagnostics_;
}

std::string IeskfEstimator::describe() const
{
  std::ostringstream oss;
  oss << "ieskf:max_iter=" << config_.max_iteration_per_scan
      << ",gravity_norm=" << config_.gravity_norm
      << ",extrinsic_t=("
      << config_.extrinsic_translation_imu_from_lidar.x() << ","
      << config_.extrinsic_translation_imu_from_lidar.y() << ","
      << config_.extrinsic_translation_imu_from_lidar.z() << ")";
  return oss.str();
}

}  // namespace pylot_lio
