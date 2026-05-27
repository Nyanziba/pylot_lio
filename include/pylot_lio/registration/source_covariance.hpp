// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#ifndef PYLOT_LIO__REGISTRATION__SOURCE_COVARIANCE_HPP_
#define PYLOT_LIO__REGISTRATION__SOURCE_COVARIANCE_HPP_

#include <vector>

#include <Eigen/Core>

#include "pylot_lio/types.hpp"

namespace pylot_lio
{

// GICP の distribution-to-distribution 化に使う source 点群の局所共分散 C_s を計算する。
// 各点について PCL KdTree で近傍 num_neighbors 点を取り、 その (母) 共分散を求め、
// 平面性正則化 (Segal 2009): 固有値を (plane_epsilon, 1, 1) に置換する
// (最小固有値方向 = 法線方向を plane_epsilon、 平面に沿う 2 方向を 1)。
//
// 戻り値は source_cloud_body.points と同じ並び順・長さ。 近傍が 3 点未満で共分散を
// 組めない点や NaN 点は plane_epsilon * I (ほぼ等方、 重みを効かせない) を入れる。
//
// plain_gicp と metal_vgicp の双方から使う共通実装 (DRY)。
std::vector<Eigen::Matrix3d> computeSourceCovariances(
  const PointCloud & source_cloud_body,
  int num_neighbors,
  double plane_epsilon);

}  // namespace pylot_lio

#endif  // PYLOT_LIO__REGISTRATION__SOURCE_COVARIANCE_HPP_
