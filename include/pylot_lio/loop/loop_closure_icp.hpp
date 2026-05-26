// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#ifndef PYLOT_LIO__LOOP__LOOP_CLOSURE_ICP_HPP_
#define PYLOT_LIO__LOOP__LOOP_CLOSURE_ICP_HPP_

#include <string>

#include <Eigen/Geometry>

#include "pylot_lio/types.hpp"

namespace pylot_lio
{

// Loop closure factor のための SE(3) 拘束を、 2 つの keyframe 点群から ICP で推定する。
//
// Scan Context は place recognition と yaw 推定までしか提供しないため、 PGO に投入する
// loop factor の 6DoF 相対 pose は別途精密化する必要がある。 ここでは small_gicp
// (GICP) を用いて query keyframe の cloud を match keyframe の cloud に合わせて
// SE(3) を取り直す。
//
// 入出力:
//   - 入力: query_cloud_body, match_cloud_body, 初期推定 T_match_query
//   - 出力: 精密化された T_match_query (= 「query 系の点を match 系に持っていく変換」)
//
// small_gicp が利用できない環境ではフォールバックとして初期推定をそのまま返す
// (loop closure の効果は薄れるが、 ビルドは通る)。
struct LoopClosureIcpResult
{
  Eigen::Isometry3d transform_match_from_query = Eigen::Isometry3d::Identity();
  double fitness_score = 0.0;     // 小さいほど良い (small_gicp の error)
  bool converged = false;
  std::string backend_name;
};

struct LoopClosureIcpConfig
{
  double voxel_resolution_m = 0.5;
  double max_correspondence_distance_m = 5.0;
  int max_iterations = 50;
  int num_threads = 4;
};

LoopClosureIcpResult refineLoopTransform(
  const PointCloud & query_cloud_body,
  const PointCloud & match_cloud_body,
  const Eigen::Isometry3d & initial_transform_match_from_query,
  const LoopClosureIcpConfig & config);

}  // namespace pylot_lio

#endif  // PYLOT_LIO__LOOP__LOOP_CLOSURE_ICP_HPP_
