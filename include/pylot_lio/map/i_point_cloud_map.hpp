// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#ifndef PYLOT_LIO__MAP__I_POINT_CLOUD_MAP_HPP_
#define PYLOT_LIO__MAP__I_POINT_CLOUD_MAP_HPP_

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "pylot_lio/types.hpp"

namespace pylot_lio
{

// 走行中に蓄積していく world 座標系のローカルマップ。マッチングの「相手」になる。
// 実装によって表現が違う:
//   - VoxelMap     : 一定ボクセル内の点と局所共分散
//   - NormalMap    : 法線つき点群
//   - KdTreeMap    : ナイーブな KdTree。リファレンス実装の役割。
class IPointCloudMap
{
public:
  virtual ~IPointCloudMap() = default;

  // world 座標系の点群をマップに追加する。
  // pose_world_body は現在ロボット姿勢で、ノードが渡す。中身はマップ実装ごとに判断する。
  virtual void insertScan(
    const PointCloud & scan_in_world,
    const Eigen::Isometry3d & pose_world_body) = 0;

  // query_world に最も近い点を 1 件返す。法線方向の情報は target_covariance に詰める。
  // 該当なしの場合は correspondence.valid = false で返す。
  virtual PointCorrespondence findNearestNeighbor(
    const Eigen::Vector3d & query_world) const = 0;

  // 現在マップに含まれる点数 (代表点数; ボクセル中心など)。
  virtual std::size_t size() const = 0;

  // 現在マップを 1 個の point cloud として吐き出す (デバッグ/可視化用)。
  virtual PointCloudPtr toPointCloud() const = 0;

  virtual std::string describe() const = 0;
};

using IPointCloudMapPtr = std::unique_ptr<IPointCloudMap>;

}  // namespace pylot_lio

#endif  // PYLOT_LIO__MAP__I_POINT_CLOUD_MAP_HPP_
