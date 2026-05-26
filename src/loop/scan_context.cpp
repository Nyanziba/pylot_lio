// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#include "pylot_lio/loop/scan_context.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace pylot_lio
{

namespace
{

// 1 点の (x, y, z) から (ring index, sector index) を計算する。
// 範囲外なら ring_index < 0 を返す。 sector は [0, num_sectors) の範囲に正規化される。
struct PolarBin
{
  int ring_index;
  int sector_index;
};

PolarBin pointToPolarBin(
  double point_x_body, double point_y_body, const ScanContextConfig & config)
{
  const double radius = std::sqrt(point_x_body * point_x_body + point_y_body * point_y_body);
  if (radius < config.min_radius_m || radius >= config.max_radius_m) {
    return PolarBin{-1, 0};
  }
  // ring index は [min_radius, max_radius) を num_rings 等分。
  const double ring_width =
    (config.max_radius_m - config.min_radius_m) / static_cast<double>(config.num_rings);
  int ring_index = static_cast<int>((radius - config.min_radius_m) / ring_width);
  if (ring_index < 0) {
    ring_index = 0;
  }
  if (ring_index >= config.num_rings) {
    ring_index = config.num_rings - 1;
  }
  // sector: atan2 の [-π, π] を [0, 2π) に変換し、 num_sectors 等分。
  const double two_pi = 2.0 * M_PI;
  double angle = std::atan2(point_y_body, point_x_body);
  if (angle < 0.0) {
    angle += two_pi;
  }
  int sector_index = static_cast<int>(
    angle / two_pi * static_cast<double>(config.num_sectors));
  if (sector_index < 0) {
    sector_index = 0;
  }
  if (sector_index >= config.num_sectors) {
    sector_index = config.num_sectors - 1;
  }
  return PolarBin{ring_index, sector_index};
}

}  // namespace

ScanContextDescriptor computeScanContext(
  const PointCloud & cloud_body, const ScanContextConfig & config)
{
  ScanContextDescriptor descriptor;
  descriptor.num_rings = config.num_rings;
  descriptor.num_sectors = config.num_sectors;
  // 高さは「該当 cell に入った最大 z」。 未入力 cell は 0 と区別したいが、 論文では未入力も 0
  // として扱うのでそれに従う (地面付近の cell と未入力 cell の区別は ring key が担う)。
  descriptor.matrix = Eigen::MatrixXd::Zero(config.num_rings, config.num_sectors);

  for (const Point & input_point : cloud_body.points) {
    if (!std::isfinite(input_point.x) || !std::isfinite(input_point.y) ||
        !std::isfinite(input_point.z))
    {
      continue;
    }
    const PolarBin bin = pointToPolarBin(input_point.x, input_point.y, config);
    if (bin.ring_index < 0) {
      continue;
    }
    const double current_value =
      descriptor.matrix(bin.ring_index, bin.sector_index);
    if (static_cast<double>(input_point.z) > current_value) {
      descriptor.matrix(bin.ring_index, bin.sector_index) =
        static_cast<double>(input_point.z);
    }
  }

  // ring key: 各 ring の非ゼロ cell 比率。 yaw 不変。
  descriptor.ring_key = Eigen::VectorXd::Zero(config.num_rings);
  for (int ring_index = 0; ring_index < config.num_rings; ++ring_index) {
    int non_zero_count = 0;
    for (int sector_index = 0; sector_index < config.num_sectors; ++sector_index) {
      if (descriptor.matrix(ring_index, sector_index) != 0.0) {
        ++non_zero_count;
      }
    }
    descriptor.ring_key(ring_index) =
      static_cast<double>(non_zero_count) / static_cast<double>(config.num_sectors);
  }
  return descriptor;
}

ScanContextMatchResult matchScanContexts(
  const ScanContextDescriptor & query, const ScanContextDescriptor & target)
{
  ScanContextMatchResult result;
  if (!query.valid() || !target.valid()) {
    return result;
  }
  if (query.num_rings != target.num_rings || query.num_sectors != target.num_sectors) {
    return result;
  }

  const int num_sectors = query.num_sectors;
  const int num_rings = query.num_rings;
  double best_distance = std::numeric_limits<double>::infinity();
  int best_shift = 0;

  for (int shift = 0; shift < num_sectors; ++shift) {
    // shift 列だけ target を右にローテートしたもの (= query をその逆だけ回転したもの) と
    // sector ごとの cosine 距離を足し合わせる。
    double sector_distance_sum = 0.0;
    int counted_sectors = 0;
    for (int sector_index = 0; sector_index < num_sectors; ++sector_index) {
      const int target_sector = (sector_index + shift) % num_sectors;
      const Eigen::VectorXd query_column = query.matrix.col(sector_index);
      const Eigen::VectorXd target_column = target.matrix.col(target_sector);
      const double query_norm = query_column.norm();
      const double target_norm = target_column.norm();
      if (query_norm < 1e-9 || target_norm < 1e-9) {
        // どちらかの列が完全に空ならその sector はスキップする (空 cell ばかり数えると 1
        // が積もって false negative になる)。
        continue;
      }
      const double cosine_similarity =
        query_column.dot(target_column) / (query_norm * target_norm);
      sector_distance_sum += (1.0 - cosine_similarity);
      ++counted_sectors;
    }
    if (counted_sectors == 0) {
      continue;
    }
    // 論文 Kim&Kim 2018 式に合わせ、 空 sector も「全不一致」としてペナルティ込みで
    // num_sectors で割る。 counted_sectors で割ってしまうと「shift=0 で 1 sector
    // しか overlap していなくてもそこが完全一致なら averaged_distance=0 になる」
    // という偽の最小値が生まれ、 yaw 推定が偏る (例: 常に 0 を返す) 原因になる。
    const double averaged_distance =
      sector_distance_sum / static_cast<double>(num_sectors);
    if (averaged_distance < best_distance) {
      best_distance = averaged_distance;
      best_shift = shift;
    }
  }
  (void)num_rings;

  if (!std::isfinite(best_distance)) {
    return result;
  }

  result.distance = best_distance;
  result.best_shift_columns = best_shift;
  // shift 列分回転は angle = 2π·shift/num_sectors に対応。
  // target を query に合わせるための yaw 推定値: query = R(yaw) · target。
  result.estimated_yaw_rad =
    2.0 * M_PI * static_cast<double>(best_shift) / static_cast<double>(num_sectors);
  if (result.estimated_yaw_rad > M_PI) {
    result.estimated_yaw_rad -= 2.0 * M_PI;
  }
  return result;
}

double ringKeyDistance(
  const Eigen::VectorXd & query_ring_key, const Eigen::VectorXd & target_ring_key)
{
  if (query_ring_key.size() != target_ring_key.size()) {
    return std::numeric_limits<double>::infinity();
  }
  return (query_ring_key - target_ring_key).norm();
}

}  // namespace pylot_lio
