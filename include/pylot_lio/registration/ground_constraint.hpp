// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#ifndef PYLOT_LIO__REGISTRATION__GROUND_CONSTRAINT_HPP_
#define PYLOT_LIO__REGISTRATION__GROUND_CONSTRAINT_HPP_

#include <optional>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace pylot_lio
{

// 地面平面 leveling 拘束 (IMU 重力が使えないときのピッチドリフト対策) の純データ + 純関数。
// ROS 非依存・GoogleTest 対象。
//
// アイデア:
//   各スキャンで「支配的地面」 の法線を body フレームで推定し、 world frame で
//   (0, 0, 1) に合わせる soft 拘束を VGICP の Gauss-Newton 正規方程式に加算する。
//   重力ベクトル無しでも遠方地面の「お椀化」 (ピッチドリフト) を抑制できる。
//
// 制約:
//   平地走行が前提 (坂・段差では max_tilt_deg で除外、 max_correction_per_frame で damping)。
struct GroundConstraintConfig
{
  // 最低 world z からこの高さ [m] までを地面候補として法線推定する。
  double band_m = 0.5;
  // 推定法線が (0,0,1) からこの角度 [deg] を超えて傾く scan は地面とみなさない (壁・坂を弾く)。
  double max_tilt_deg = 30.0;
};

// source 点群 (body frame) と現在姿勢から「支配的地面の法線 (body)」 を推定する。
// 平面性 / 傾きゲートを満たさなければ nullopt (= この scan では拘束を使わない)。
// percentile-based floor band 抽出 + 共分散の最小固有ベクトル ≈ 法線。
std::optional<Eigen::Vector3d> estimateGroundNormalBody(
  const std::vector<Eigen::Vector3d> & source_points_body,
  const Eigen::Isometry3d & pose_world_body,
  const GroundConstraintConfig & config);

// VGICP の 6x6 正規方程式 (H, b) に「地面法線を world-up に合わせる」 1 自由度の
// soft 拘束を加算する。 回転ヘッシアン (上 3x3) と回転勾配 (上 3) に効く。
//
// 引数:
//   hessian, gradient            in-place 更新される (左摂動 [rotation, translation] 順)
//   current_pose_world_body      現在の推定姿勢 (法線を world に持って行くため)
//   ground_normal_body           推定済みの地面法線 (body)
//   weight                       拘束強度 (内部で回転剛性スケール済を想定)
//   max_correction_rad           damping: 1 align あたりの leveling 補正上限 [rad]。
//                                0 以下なら無制限 (hard 拘束、 振動リスクあり)。
void addGroundLevelingConstraint(
  Eigen::Matrix<double, 6, 6> & hessian,
  Eigen::Matrix<double, 6, 1> & gradient,
  const Eigen::Isometry3d & current_pose_world_body,
  const Eigen::Vector3d & ground_normal_body,
  double weight,
  double max_correction_rad);

}  // namespace pylot_lio

#endif  // PYLOT_LIO__REGISTRATION__GROUND_CONSTRAINT_HPP_
