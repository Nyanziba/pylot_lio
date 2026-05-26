// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#ifndef PYLOT_LIO__ESTIMATOR__IESKF_ESTIMATOR_HPP_
#define PYLOT_LIO__ESTIMATOR__IESKF_ESTIMATOR_HPP_

#include <optional>
#include <string>

#include <Eigen/Core>

#include "pylot_lio/estimator/i_state_estimator.hpp"

namespace pylot_lio
{

// Iterated Error-State Kalman Filter (FAST-LIO2 系) の自前実装。
// 状態ベクトル (15D): [p; v; theta; bias_acc; bias_gyro]
// 誤差は SO(3) 上で扱い (theta は tangent space)、姿勢自体は回転行列で保持。
// IMU prediction → 反復スキャン更新の標準パターン。
class IeskfEstimator : public IStateEstimator
{
public:
  struct Config
  {
    // Super-LIO 流: gravity_norm を固定値として持ち、 起動時 IMU 平均から方向だけ取る。
    // この設計により「IMU の生 acc 大きさが m/s² なのか g 単位なのか」のスケール誤差が
    // gravity 推定に乗らず、 z drift の主因を消せる。
    // 既定値 9.7946 は中緯度の標準重力 (Super-LIO/livox_360.yaml と整合)。
    double gravity_norm = 9.7946;
    // true なら起動直後の静止 IMU N サンプル平均から方向を取り、 g_world =
    // -unit(mean_acc) * gravity_norm を採用する。 false なら gravity_world_init を使う。
    bool auto_estimate_gravity = true;
    int gravity_estimation_samples = 200;
    // auto_estimate_gravity=false のときに使う重力 (world 系)
    Eigen::Vector3d gravity_world_init = Eigen::Vector3d(0.0, 0.0, -9.7946);

    // LiDAR ↔ IMU extrinsic (Super-LIO lio.extrinsic.lidar_imu と同じ定義)。
    // 「LiDAR 点を IMU frame に持ってくる」変換 T_imu_lidar = (R, t)。
    // Mid-360 内蔵 IMU では R ≈ identity、 t ≈ (-0.011, -0.023, 0.044) m。
    Eigen::Matrix3d extrinsic_rotation_imu_from_lidar = Eigen::Matrix3d::Identity();
    Eigen::Vector3d extrinsic_translation_imu_from_lidar = Eigen::Vector3d::Zero();

    // IMU process noise
    double imu_acc_noise_density = 0.1;          // [m/s^2/sqrt(Hz)] Super-LIO imu_na
    double imu_gyro_noise_density = 0.1;         // [rad/s/sqrt(Hz)] Super-LIO imu_ng
    double imu_acc_bias_random_walk = 1e-4;      // [m/s^2 * sqrt(Hz)] Super-LIO imu_nba
    double imu_gyro_bias_random_walk = 1e-4;     // [rad/s * sqrt(Hz)] Super-LIO imu_nbg

    // Scan 観測 (registration が返す T を観測値として fusion)
    double scan_observation_noise_translation_m = 0.05;
    double scan_observation_noise_rotation_rad = 0.05;
    int max_iteration_per_scan = 4;
    double convergence_translation_m = 1e-3;
    double convergence_rotation_rad = 1e-3;
    double max_correspondence_distance_m = 2.0;
  };

  explicit IeskfEstimator(const Config & config);

  void initialize(const RobotState & initial_state) override;
  bool isInitialized() const override;

  void predictWithImu(const ImuSample & imu_sample) override;
  void updateWithScan(
    const PointCloud & scan_cloud_body,
    IPointCloudMap & map_world,
    IRegistration & registration,
    int64_t scan_timestamp_ns) override;

  RobotState getState() const override;
  EstimatorDiagnostics getDiagnostics() const override;
  std::string describe() const override;

  void setPose(const Eigen::Isometry3d & pose_world_body) override;

private:
  // 名前付きインデックス。可読性のため。
  static constexpr int kStateDim = 15;
  static constexpr int kIdxPos = 0;
  static constexpr int kIdxVel = 3;
  static constexpr int kIdxOri = 6;
  static constexpr int kIdxBiasAcc = 9;
  static constexpr int kIdxBiasGyro = 12;

  Config config_;
  bool initialized_;
  std::optional<int64_t> last_imu_timestamp_ns_;

  // 重力自動推定: 起動直後の N サンプル間は predict をスキップし、加速度の平均で重力を確定。
  int gravity_estimation_remaining_;
  Eigen::Vector3d gravity_estimation_accumulator_;
  // 確定した重力ベクトル (world 系)。 大きさは config_.gravity_norm 固定、 方向は auto/init から。
  Eigen::Vector3d gravity_world_active_;

  // 名目状態。
  Eigen::Vector3d nominal_position_world_;
  Eigen::Vector3d nominal_velocity_world_;
  Eigen::Matrix3d nominal_rotation_world_body_;
  Eigen::Vector3d bias_acc_body_;
  Eigen::Vector3d bias_gyro_body_;

  Eigen::Matrix<double, kStateDim, kStateDim> error_covariance_;

  EstimatorDiagnostics last_diagnostics_;
};

}  // namespace pylot_lio

#endif  // PYLOT_LIO__ESTIMATOR__IESKF_ESTIMATOR_HPP_
