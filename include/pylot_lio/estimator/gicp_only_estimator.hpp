// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#ifndef PYLOT_LIO__ESTIMATOR__GICP_ONLY_ESTIMATOR_HPP_
#define PYLOT_LIO__ESTIMATOR__GICP_ONLY_ESTIMATOR_HPP_

#include <cstdint>
#include <string>

#include "pylot_lio/estimator/i_state_estimator.hpp"

namespace pylot_lio
{

// IMU を一切使わず、毎スキャンの registration 結果をそのまま姿勢推定値とする
// 最小実装。比較ベースラインとして使う。
class GicpOnlyEstimator : public IStateEstimator
{
public:
  struct Config
  {
    // 等速度モデルで外挿する 1 スキャン分の相対変換に課す上限。
    // 上限を超えた相対変換は initial_guess としては危険 (前回の誤対応や物理的な
    // 急加速で実態と乖離している可能性が高い) なので、上限にクランプして
    // GICP の収束範囲を超えた飛びを抑える。
    double max_extrapolation_translation_m = 1.0;
    double max_extrapolation_rotation_rad = 0.5;  // ~28.6°
  };

  GicpOnlyEstimator() : GicpOnlyEstimator(Config{}) {}
  explicit GicpOnlyEstimator(const Config & config);

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
  bool usesImu() const override { return false; }

  void setPose(const Eigen::Isometry3d & pose_world_body) override;

private:
  Config config_;
  bool initialized_;
  RobotState current_state_;
  // 等速度モデル用: 前回スキャン時の姿勢を覚えておき、(前回→今回) の相対変換を
  // もう 1 ステップ進めた値を次の初期推定にする。
  bool has_previous_pose_;
  Eigen::Isometry3d previous_pose_world_body_;
  // 速度推定 / dt 補正用。 0 は「タイムスタンプ未取得」を意味する。
  int64_t previous_scan_timestamp_ns_;
  double previous_dt_s_;  // 前回スキャンと前々回スキャンの間の dt。 0 で「未確定」。
  EstimatorDiagnostics last_diagnostics_;
};

}  // namespace pylot_lio

#endif  // PYLOT_LIO__ESTIMATOR__GICP_ONLY_ESTIMATOR_HPP_
