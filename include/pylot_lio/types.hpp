// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#ifndef PYLOT_LIO__TYPES_HPP_
#define PYLOT_LIO__TYPES_HPP_

#include <cstdint>
#include <memory>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pylot_lio
{

// 1点 = 位置 (x,y,z) + 強度。Mid-360 の生点群を直接扱う型として使う。
using Point = pcl::PointXYZI;
using PointCloud = pcl::PointCloud<Point>;
using PointCloudPtr = PointCloud::Ptr;
using PointCloudConstPtr = PointCloud::ConstPtr;

// IMU 1サンプル。t は IMU 計時のナノ秒タイムスタンプ (rclcpp::Time から）。
struct ImuSample
{
  int64_t timestamp_ns;
  Eigen::Vector3d linear_acceleration_mps2;  // body frame
  Eigen::Vector3d angular_velocity_rps;      // body frame
};

// 状態推定器の公開状態。
// 内部に IMU バイアスを持つかどうかは実装に隠蔽し、外には pose と velocity だけを露出する。
struct RobotState
{
  // T_world_body : ロボット body フレームを world フレームに移す変換
  Eigen::Isometry3d pose_world_body = Eigen::Isometry3d::Identity();
  // world フレーム上の並進速度
  Eigen::Vector3d velocity_world = Eigen::Vector3d::Zero();
};

// 状態推定器の診断値。比較ダッシュボードで使う。
struct EstimatorDiagnostics
{
  int iterations = 0;             // 反復回数 (収束まで)
  double cost = 0.0;              // 最終コスト関数値
  double processing_time_ms = 0.0;
  bool converged = false;
};

// マップ近傍探索の結果 1 件。
struct PointCorrespondence
{
  Eigen::Vector3d source_point_world;  // 変換後 (world)
  Eigen::Vector3d target_point_world;  // マップ上の最近傍点 (world)
  Eigen::Matrix3d target_covariance;   // 近傍点の局所共分散 (法線方向情報を含む)
  double squared_distance;
  bool valid = true;
};

}  // namespace pylot_lio

#endif  // PYLOT_LIO__TYPES_HPP_
