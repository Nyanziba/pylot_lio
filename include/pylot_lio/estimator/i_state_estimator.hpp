// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#ifndef PYLOT_LIO__ESTIMATOR__I_STATE_ESTIMATOR_HPP_
#define PYLOT_LIO__ESTIMATOR__I_STATE_ESTIMATOR_HPP_

#include <memory>
#include <string>

#include "pylot_lio/map/i_point_cloud_map.hpp"
#include "pylot_lio/registration/i_registration.hpp"
#include "pylot_lio/types.hpp"

namespace pylot_lio
{

// 「状態推定器」のフラットな抽象。内部の状態ベクトル次元 (6D / 15D / その他) は
// 実装に隠蔽し、外には RobotState (姿勢 + 速度) と診断値だけを露出する。
//
// 1 スキャン処理の流れ:
//   1) 前回スキャンから今スキャンまでの IMU サンプルを predictWithImu で取り込む
//   2) スキャンを updateWithScan で観測としてフィードバック
//   3) getState() で外に出す
//
// IMU を使わない実装 (GICP-only) は predictWithImu を no-op にする。
class IStateEstimator
{
public:
  virtual ~IStateEstimator() = default;

  virtual void initialize(const RobotState & initial_state) = 0;
  virtual bool isInitialized() const = 0;

  virtual void predictWithImu(const ImuSample & imu_sample) = 0;

  virtual void updateWithScan(
    const PointCloud & scan_cloud_body,
    IPointCloudMap & map_world,
    IRegistration & registration) = 0;

  virtual RobotState getState() const = 0;
  virtual EstimatorDiagnostics getDiagnostics() const = 0;

  virtual std::string describe() const = 0;

  // この実装が IMU を必要とするか。false なら lio_node 側は IMU を購読しない。
  virtual bool usesImu() const { return true; }

  // PGO による pose 修正後の上書き。
  // ieskf/hgo は内部に共分散 (P) と nominal state を持っているため、 setPose は
  // pose のみを上書きしつつ共分散を「再初期化」してジャンプ後の再収束を促す責務がある。
  // 各実装が必要に応じて override する (デフォルトは no-op)。
  virtual void setPose(const Eigen::Isometry3d & /*pose_world_body*/) {}
};

using IStateEstimatorPtr = std::unique_ptr<IStateEstimator>;

}  // namespace pylot_lio

#endif  // PYLOT_LIO__ESTIMATOR__I_STATE_ESTIMATOR_HPP_
