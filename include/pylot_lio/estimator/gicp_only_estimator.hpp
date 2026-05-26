// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#ifndef PYLOT_LIO__ESTIMATOR__GICP_ONLY_ESTIMATOR_HPP_
#define PYLOT_LIO__ESTIMATOR__GICP_ONLY_ESTIMATOR_HPP_

#include <string>

#include "pylot_lio/estimator/i_state_estimator.hpp"

namespace pylot_lio
{

// IMU を一切使わず、毎スキャンの registration 結果をそのまま姿勢推定値とする
// 最小実装。比較ベースラインとして使う。
class GicpOnlyEstimator : public IStateEstimator
{
public:
  GicpOnlyEstimator();

  void initialize(const RobotState & initial_state) override;
  bool isInitialized() const override;

  void predictWithImu(const ImuSample & imu_sample) override;
  void updateWithScan(
    const PointCloud & scan_cloud_body,
    IPointCloudMap & map_world,
    IRegistration & registration) override;

  RobotState getState() const override;
  EstimatorDiagnostics getDiagnostics() const override;
  std::string describe() const override;
  bool usesImu() const override { return false; }

  void setPose(const Eigen::Isometry3d & pose_world_body) override;

private:
  bool initialized_;
  RobotState current_state_;
  // 等速度モデル用: 前回スキャン時の姿勢を覚えておき、(前回→今回) の相対変換を
  // もう 1 ステップ進めた値を次の初期推定にする。
  bool has_previous_pose_;
  Eigen::Isometry3d previous_pose_world_body_;
  EstimatorDiagnostics last_diagnostics_;
};

}  // namespace pylot_lio

#endif  // PYLOT_LIO__ESTIMATOR__GICP_ONLY_ESTIMATOR_HPP_
