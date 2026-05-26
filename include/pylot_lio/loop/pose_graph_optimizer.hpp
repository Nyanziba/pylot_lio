// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#ifndef PYLOT_LIO__LOOP__POSE_GRAPH_OPTIMIZER_HPP_
#define PYLOT_LIO__LOOP__POSE_GRAPH_OPTIMIZER_HPP_

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Geometry>

namespace pylot_lio
{

// GTSAM ベースの SE(3) pose graph 最適化。
//
// API 想定:
//   - addKeyframePrior(0, T_world_kf0): 最初の keyframe は world に固定 (prior)
//   - addOdometryFactor(prev_id, curr_id, T_prev_curr, sigmas): odometry 拘束
//   - addLoopFactor(query_id, match_id, T_match_query, sigmas): loop 拘束
//   - addKeyframeInitialEstimate(kf_id, T_world_kf): 新キーフレーム値を初期推定として投入
//   - optimize(): ISAM2 を 1 回回し、 全 keyframe の最適化済 pose を更新
//   - optimizedPose(kf_id): 最新最適化結果
//
// 失敗時は std::runtime_error。 失敗ケースは「factor を追加せず optimize()」
// のような pathological な場合のみで、 通常運用では発生しない想定。
class PoseGraphOptimizer
{
public:
  struct NoiseSigmas
  {
    // SE(3) は 6DoF: [rx, ry, rz, tx, ty, tz] の順 (GTSAM Pose3 と整合)
    double rot_x_rad = 0.01;
    double rot_y_rad = 0.01;
    double rot_z_rad = 0.01;
    double trans_x_m = 0.05;
    double trans_y_m = 0.05;
    double trans_z_m = 0.05;
  };

  PoseGraphOptimizer();
  ~PoseGraphOptimizer();
  PoseGraphOptimizer(const PoseGraphOptimizer &) = delete;
  PoseGraphOptimizer & operator=(const PoseGraphOptimizer &) = delete;

  // 「最初の keyframe を world frame にピン留めする」prior factor。
  // 起動時に必ず 1 回だけ呼ぶ。 重い sigma で固定すること (各軸 1e-6)。
  void addKeyframePrior(
    uint32_t keyframe_id, const Eigen::Isometry3d & pose_world_keyframe);

  // odometry 拘束: prev_keyframe_id → curr_keyframe_id の相対 SE(3)。
  // T_prev_curr = pose_world_prev^{-1} · pose_world_curr。 LIO 出力から計算したものを渡す。
  void addOdometryFactor(
    uint32_t prev_keyframe_id, uint32_t curr_keyframe_id,
    const Eigen::Isometry3d & relative_transform_prev_to_curr,
    const NoiseSigmas & sigmas);

  // loop closure 拘束: ICP で精密化した T_match_query。
  // 「query 系の点を match 系に持っていく変換」。
  void addLoopFactor(
    uint32_t query_keyframe_id, uint32_t match_keyframe_id,
    const Eigen::Isometry3d & relative_transform_query_to_match,
    const NoiseSigmas & sigmas);

  // 新規 keyframe の初期推定値を投入。 まだ optimize() を呼んでいない場合は
  // 「現時点の推定値」として使われる。
  void addKeyframeInitialEstimate(
    uint32_t keyframe_id, const Eigen::Isometry3d & pose_world_keyframe);

  // ISAM2 を 1 step 回す。 内部に積もった factor / new values を flush し、 全 keyframe
  // の最適化済 pose を更新する。
  void optimize();

  // 最後に optimize() した時点の最適化済 pose。 存在しなければ identity を返す
  // (まだ追加されていない / optimize 未実行のとき)。
  Eigen::Isometry3d optimizedPose(uint32_t keyframe_id) const;

  // 全ての最新最適化済 pose を取り出す。
  std::vector<std::pair<uint32_t, Eigen::Isometry3d>> allOptimizedPoses() const;

  std::size_t numFactors() const;
  std::size_t numKeyframes() const;

  std::string describe() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace pylot_lio

#endif  // PYLOT_LIO__LOOP__POSE_GRAPH_OPTIMIZER_HPP_
