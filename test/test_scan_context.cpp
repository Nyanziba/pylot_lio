// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#include <gtest/gtest.h>

#include <cmath>

#include <Eigen/Geometry>

#include "pylot_lio/loop/scan_context.hpp"
#include "pylot_lio/loop/scan_context_loop_detector.hpp"

namespace pylot_lio
{

namespace
{

// 同心円状にランダム-ish な点群を作る (再現性のため決定的シード)。
// 「壁」のように半径方向に伸びるストライプ模様を作って yaw 推定をテストできるようにする。
PointCloud makeStripedCloud()
{
  PointCloud cloud;
  // 8 本のストライプを 45 度ごとに置く。 各ストライプは半径方向に伸びる点列。
  for (int stripe_index = 0; stripe_index < 8; ++stripe_index) {
    const double base_angle =
      static_cast<double>(stripe_index) * (2.0 * M_PI / 8.0);
    for (int radius_step = 1; radius_step <= 20; ++radius_step) {
      const double radius = static_cast<double>(radius_step);
      Point sample;
      sample.x = static_cast<float>(radius * std::cos(base_angle));
      sample.y = static_cast<float>(radius * std::sin(base_angle));
      // ストライプごとに違う高さを与えると、 descriptor が yaw 識別を持つようになる。
      sample.z = static_cast<float>(0.1 * stripe_index + 0.5);
      sample.intensity = 1.0f;
      cloud.points.push_back(sample);
    }
  }
  cloud.width = static_cast<uint32_t>(cloud.points.size());
  cloud.height = 1;
  cloud.is_dense = true;
  return cloud;
}

PointCloud rotateCloudYaw(const PointCloud & input, double yaw_rad)
{
  PointCloud rotated;
  rotated.points.reserve(input.points.size());
  const double cos_yaw = std::cos(yaw_rad);
  const double sin_yaw = std::sin(yaw_rad);
  for (const Point & input_point : input.points) {
    Point rotated_point;
    rotated_point.x =
      static_cast<float>(cos_yaw * input_point.x - sin_yaw * input_point.y);
    rotated_point.y =
      static_cast<float>(sin_yaw * input_point.x + cos_yaw * input_point.y);
    rotated_point.z = input_point.z;
    rotated_point.intensity = input_point.intensity;
    rotated.points.push_back(rotated_point);
  }
  rotated.width = static_cast<uint32_t>(rotated.points.size());
  rotated.height = 1;
  rotated.is_dense = true;
  return rotated;
}

ScanContextConfig smallConfig()
{
  ScanContextConfig config;
  config.num_rings = 20;
  config.num_sectors = 8;
  config.max_radius_m = 25.0;
  config.min_radius_m = 0.5;
  return config;
}

Eigen::Isometry3d poseAt(double tx, double ty)
{
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.translation() = Eigen::Vector3d(tx, ty, 0.0);
  return pose;
}

}  // namespace

TEST(ScanContext, IdenticalCloudHasZeroDistance)
{
  const PointCloud cloud = makeStripedCloud();
  const ScanContextDescriptor a = computeScanContext(cloud, smallConfig());
  const ScanContextDescriptor b = computeScanContext(cloud, smallConfig());
  const ScanContextMatchResult result = matchScanContexts(a, b);
  EXPECT_NEAR(result.distance, 0.0, 1e-9);
  EXPECT_EQ(result.best_shift_columns, 0);
}

TEST(ScanContext, YawRotationProducesLowDistance)
{
  // num_sectors=8 で 1 sector ぶん (=45度) 回した cloud と、 元の cloud は
  // ベストマッチで小さな距離になるはず。 shift の整数値はストライプ高さ分布の
  // 形状にも依存するためここでは距離のみを検証する (具体的な shift 値の検証は
  // Y2 のような明瞭なテストでないと不安定になる)。
  const PointCloud cloud = makeStripedCloud();
  const double applied_yaw = 2.0 * M_PI / 8.0;
  const PointCloud rotated = rotateCloudYaw(cloud, applied_yaw);

  const ScanContextDescriptor original = computeScanContext(cloud, smallConfig());
  const ScanContextDescriptor turned = computeScanContext(rotated, smallConfig());
  const ScanContextMatchResult result = matchScanContexts(original, turned);

  EXPECT_LT(result.distance, 0.1);
}

TEST(ScanContext, ColumnShiftDirectlyRecoversYaw)
{
  // 単一のストライプ (sector 0) だけ存在する cloud を作る。 これを 1 sector ぶん
  // 回した cloud との column-shift マッチは確定的に shift=1 を返すはず。
  PointCloud single_stripe;
  for (int radius_step = 1; radius_step <= 15; ++radius_step) {
    Point sample;
    sample.x = static_cast<float>(radius_step);
    sample.y = 0.0f;
    sample.z = 1.0f;  // 一定高さの方が cosine の振る舞いが素直
    sample.intensity = 1.0f;
    single_stripe.points.push_back(sample);
  }
  single_stripe.width = static_cast<uint32_t>(single_stripe.points.size());
  single_stripe.height = 1;
  single_stripe.is_dense = true;

  const double applied_yaw = 2.0 * M_PI / 8.0;
  const PointCloud rotated_stripe = rotateCloudYaw(single_stripe, applied_yaw);

  const ScanContextDescriptor a = computeScanContext(single_stripe, smallConfig());
  const ScanContextDescriptor b = computeScanContext(rotated_stripe, smallConfig());
  const ScanContextMatchResult result = matchScanContexts(a, b);

  EXPECT_NEAR(result.distance, 0.0, 1e-6);
  // shift は 1 sector (= 45 度) または等価な 7 sector シフト
  EXPECT_TRUE(result.best_shift_columns == 1 || result.best_shift_columns == 7);
}

TEST(ScanContext, DifferentCloudsHaveLargeDistance)
{
  PointCloud cloud_a = makeStripedCloud();
  // 全く別の cloud: 単一クラスタを 1 点だけ置く
  PointCloud cloud_b;
  for (int point_index = 0; point_index < 5; ++point_index) {
    Point sample;
    sample.x = static_cast<float>(0.5 + 0.1 * point_index);
    sample.y = 0.0f;
    sample.z = 0.0f;
    sample.intensity = 1.0f;
    cloud_b.points.push_back(sample);
  }
  cloud_b.width = static_cast<uint32_t>(cloud_b.points.size());
  cloud_b.height = 1;
  cloud_b.is_dense = true;
  const ScanContextDescriptor a = computeScanContext(cloud_a, smallConfig());
  const ScanContextDescriptor b = computeScanContext(cloud_b, smallConfig());
  const ScanContextMatchResult result = matchScanContexts(a, b);
  // 違う点群同士は距離が遠い。 厳密値は実装依存だが、 0.05 より十分大きいはず。
  EXPECT_GT(result.distance, 0.1);
}

TEST(ScanContextLoopDetector, RecentKeyframesAreExcluded)
{
  ScanContextLoopDetector::Config config;
  config.descriptor = smallConfig();
  config.ring_key_top_k = 4;
  config.score_threshold = 0.05;
  config.exclude_recent_kf = 5;  // 直近 5 個は検索対象外
  ScanContextLoopDetector detector(config);

  const PointCloud cloud = makeStripedCloud();

  // 同じ cloud を 4 keyframe 連続で入れる: どれも recent 範囲 (差<5) なので検出しない
  for (uint32_t kf_id = 0; kf_id < 4; ++kf_id) {
    auto result = detector.addKeyframeAndQuery(
      kf_id, cloud, poseAt(static_cast<double>(kf_id), 0.0));
    EXPECT_FALSE(result.detected) << "kf=" << kf_id;
  }

  // 5 個目 (id=4) も exclude_recent=5 なので未検出 (id=0 との差 4 < 5)
  auto result_4 = detector.addKeyframeAndQuery(4, cloud, poseAt(4.0, 0.0));
  EXPECT_FALSE(result_4.detected);

  // 6 個目 (id=5): id=0 との差 5 >= 5 で検索対象に入り、 同じ cloud なので検出される
  auto result_5 = detector.addKeyframeAndQuery(5, cloud, poseAt(5.0, 0.0));
  EXPECT_TRUE(result_5.detected);
  EXPECT_EQ(result_5.candidate.query_kf_id, 5u);
  EXPECT_EQ(result_5.candidate.match_kf_id, 0u);
  EXPECT_LT(result_5.candidate.scan_context_distance, 0.05);
}

TEST(ScanContextLoopDetector, UnrelatedKeyframesAreNotDetected)
{
  ScanContextLoopDetector::Config config;
  config.descriptor = smallConfig();
  config.ring_key_top_k = 4;
  config.score_threshold = 0.05;  // 厳しめ
  config.exclude_recent_kf = 0;
  ScanContextLoopDetector detector(config);

  const PointCloud cloud_a = makeStripedCloud();
  // 構造的に違う cloud_b: 単一の小さな点群クラスタだけを置く。 ring 占有率も大きく変わる。
  PointCloud cloud_b;
  for (int point_index = 0; point_index < 10; ++point_index) {
    Point sample;
    sample.x = static_cast<float>(2.0 + 0.1 * point_index);
    sample.y = 0.0f;
    sample.z = 0.0f;
    sample.intensity = 1.0f;
    cloud_b.points.push_back(sample);
  }
  cloud_b.width = static_cast<uint32_t>(cloud_b.points.size());
  cloud_b.height = 1;
  cloud_b.is_dense = true;

  detector.addKeyframeAndQuery(0, cloud_a, poseAt(0.0, 0.0));
  auto result = detector.addKeyframeAndQuery(1, cloud_b, poseAt(10.0, 10.0));
  // 構造が大きく違うので、 score_threshold=0.05 では検出されないはず
  EXPECT_FALSE(result.detected);
}

}  // namespace pylot_lio
