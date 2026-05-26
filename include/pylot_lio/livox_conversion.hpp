// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#ifndef PYLOT_LIO__LIVOX_CONVERSION_HPP_
#define PYLOT_LIO__LIVOX_CONVERSION_HPP_

#include <cstdint>
#include <vector>

#include "pylot_lio/types.hpp"

namespace pylot_lio
{

// Livox CustomMsg -> 内部 PCL 点群への変換。
//
// CustomMsg は Livox ROS driver が出す点ごと構造体の配列で、 各点は
//   { x, y, z, reflectivity, tag, line, offset_time }
// を持つ。 x/y/z は float (m)、 reflectivity は uint8、 offset_time はパッケージ
// header.stamp からの ns オフセット (パッケージ先頭点が 0)。
//
// このモジュールでは「CustomMsg の生 raw 値の配列」を入力として受け、 lio_node 内部の
// pcl::PointXYZI に変換した PointCloud を返す。 livox_ros_driver2 のヘッダに直接依存
// せずにテストできるよう、 構造体は最小限の素な型 (LivoxRawPoint) を経由する。 ROS
// 側のコードは livox_ros_driver2::msg::CustomMsg::points を LivoxRawPoint vector に
// 移し替えてからこの関数に渡す。
struct LivoxRawPoint
{
  float x;
  float y;
  float z;
  uint8_t reflectivity;
  // offset_time は本変換では使わないが、 motion undistortion を後段に入れたいとき
  // 必要なので保持しておく。 単位は ns。
  uint32_t offset_time_ns;
};

// CustomMsg 流の生点列を pcl::PointXYZI 点群に変換する。
//   - reflectivity (0..255) を intensity (float) にそのまま入れる
//   - NaN / 範囲外フィルタは行わない (後段の前処理 / マッチングに任せる)
//   - 出力 PointCloud の width = points.size(), height = 1, is_dense = true
PointCloud convertLivoxRawPoints(const std::vector<LivoxRawPoint> & raw_points);

}  // namespace pylot_lio

#endif  // PYLOT_LIO__LIVOX_CONVERSION_HPP_
