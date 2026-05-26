// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#ifndef PYLOT_LIO__ESTIMATOR__HGO_ESTIMATOR_HPP_
#define PYLOT_LIO__ESTIMATOR__HGO_ESTIMATOR_HPP_

#include <optional>
#include <string>

#include <Eigen/Core>

#include "pylot_lio/estimator/i_state_estimator.hpp"

namespace pylot_lio
{

// Hierarchical Geometric Observer 風の自前実装。
// 厳密な論文実装ではなく、SE(3) 上の比例観測器として簡略化したもの:
//   - IMU で姿勢・速度を高速予測
//   - スキャン GICP 結果と予測の差を「観測誤差」とし、ゲイン K で滑らかに補正
// 共分散行列は持たないため軽量だが、退化方向に弱い。
class HgoEstimator : public IStateEstimator
{
public:
  struct Config
  {
    Eigen::Vector3d gravity_world = Eigen::Vector3d(0.0, 0.0, -9.81);
    bool auto_estimate_gravity = true;
    int gravity_estimation_samples = 100;
    double translation_gain = 0.5;       // 0..1
    double rotation_gain = 0.5;          // 0..1
    double velocity_gain = 0.3;
    double bias_acc_gain = 1e-3;
    double bias_gyro_gain = 1e-4;
  };

  explicit HgoEstimator(const Config & config);

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

  void setPose(const Eigen::Isometry3d & pose_world_body) override;

private:
  Config config_;
  bool initialized_;
  std::optional<int64_t> last_imu_timestamp_ns_;
  int gravity_estimation_remaining_;
  Eigen::Vector3d gravity_estimation_accumulator_;

  Eigen::Vector3d position_world_;
  Eigen::Vector3d velocity_world_;
  Eigen::Matrix3d rotation_world_body_;
  Eigen::Vector3d bias_acc_body_;
  Eigen::Vector3d bias_gyro_body_;

  EstimatorDiagnostics last_diagnostics_;
};

}  // namespace pylot_lio

#endif  // PYLOT_LIO__ESTIMATOR__HGO_ESTIMATOR_HPP_
