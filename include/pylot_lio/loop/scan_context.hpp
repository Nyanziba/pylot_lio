// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#ifndef PYLOT_LIO__LOOP__SCAN_CONTEXT_HPP_
#define PYLOT_LIO__LOOP__SCAN_CONTEXT_HPP_

#include <cstddef>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "pylot_lio/types.hpp"

namespace pylot_lio
{

// Kim & Kim, "Scan Context: Egocentric Spatial Descriptor for Place Recognition
// within 3D Point Cloud Map" (IROS 2018) の自前実装。
//
// 概要:
//   - 入力: 1 キーフレームの body フレーム点群 (LiDAR 中心を原点とする)
//   - 出力: (num_rings × num_sectors) の行列。 cell 値 = その cell に入った点の最大 z (高さ)
//   - ring key: 各 ring に対する非ゼロ cell の占有率 (yaw 不変量)。 粗フィルタに使う。
//   - マッチング: 列方向 (= sector 軸 = 方位角) のシフトを全探索し、最小列ごとの
//     cosine 距離 (1 - cos类似度) の平均を取る。 シフト量 k → yaw = 2π·k/num_sectors。
//
// Scan Context は body 系で計算するからこそ yaw 推定が意味を持つ。 world 系で計算
// すると 2 つの descriptor の角度軸がそれぞれ別の世界回転を含んでしまい、 column-shift
// だけで yaw を取り出せなくなる点に注意。
struct ScanContextDescriptor
{
  // (ring index, sector index) → 高さ。 行優先 (num_rings 行 × num_sectors 列)。
  Eigen::MatrixXd matrix;
  // ring ごとの占有率 (= non-zero cell 数 / num_sectors)。 yaw 不変なので粗フィルタ用。
  Eigen::VectorXd ring_key;
  int num_rings = 0;
  int num_sectors = 0;

  bool valid() const { return num_rings > 0 && num_sectors > 0; }
};

struct ScanContextConfig
{
  int num_rings = 60;
  int num_sectors = 20;
  double max_radius_m = 80.0;
  // ring 0 の内径。 これ未満の点は描画対象外 (LiDAR 自身の dead zone)。
  double min_radius_m = 0.5;
};

// 点群から descriptor を計算する。 NaN / 範囲外の点は単に無視される。
ScanContextDescriptor computeScanContext(
  const PointCloud & cloud_body, const ScanContextConfig & config);

// 2 descriptor 間の最小距離と最良 yaw 推定値を計算する。
// 内部で column-shift 0..num_sectors-1 全探索。 計算量 O(num_sectors² × num_rings)。
//
// 戻り値: { distance, best_shift_columns }
struct ScanContextMatchResult
{
  double distance = 1.0;     // 0 が完全一致、 1 が完全不一致 (sector ごとの 1 - cos の平均)
  int best_shift_columns = 0;  // column を何個右シフトすると best になるか
  double estimated_yaw_rad = 0.0;
};

ScanContextMatchResult matchScanContexts(
  const ScanContextDescriptor & query,
  const ScanContextDescriptor & target);

// ring key 同士の L2 距離 (粗フィルタ用)。
double ringKeyDistance(
  const Eigen::VectorXd & query_ring_key,
  const Eigen::VectorXd & target_ring_key);

}  // namespace pylot_lio

#endif  // PYLOT_LIO__LOOP__SCAN_CONTEXT_HPP_
