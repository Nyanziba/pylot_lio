// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#ifndef PYLOT_LIO__LOOP__SCAN_CONTEXT_LOOP_DETECTOR_HPP_
#define PYLOT_LIO__LOOP__SCAN_CONTEXT_LOOP_DETECTOR_HPP_

#include <cstdint>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "pylot_lio/loop/scan_context.hpp"
#include "pylot_lio/types.hpp"

namespace pylot_lio
{

// Scan Context ベースのループ検出器。
//
// 使い方:
//   1. keyframe 採択ごとに addKeyframe(kf_id, scan_body, pose_world_body) を呼ぶ。
//      内部で descriptor を計算し、 同時に query として現在 DB に対し検索する。
//   2. 検索結果が score < threshold なら LoopCandidate として記録 + 返す。
//   3. exclude_recent_kf 個以内の最新キーフレームは検索対象から除外する
//      (隣接 keyframe を「同じ場所」と誤検出しないため)。
//
// loop closure の補正適用 (estimator setPose / pose graph) は本クラスの責務外。
// PR3 で別途実装する想定。
class ScanContextLoopDetector
{
public:
  struct Config
  {
    ScanContextConfig descriptor;
    // 検索時、 ring key L2 距離で上位何個に絞ってから column-shift マッチを取るか
    int ring_key_top_k = 10;
    // 「同じ場所」と判定するための Scan Context 距離 (0=完全一致, 1=完全不一致)
    double score_threshold = 0.2;
    // この個数より新しい keyframe は検索対象から除外する (近接 keyframe との誤検出防止)
    int exclude_recent_kf = 50;
  };

  struct LoopCandidate
  {
    uint32_t query_kf_id;
    uint32_t match_kf_id;
    double scan_context_distance;
    double estimated_yaw_rad;
    Eigen::Isometry3d query_pose_world_body;
    Eigen::Isometry3d match_pose_world_body;
  };

  explicit ScanContextLoopDetector(const Config & config);

  // 新しい keyframe を DB に追加し、 同時に過去 DB に対して loop 検索を行う。
  // ヒットしなければ detected=false で返すが、 「なぜヒットしなかったか」を
  // best_distance / best_match_kf_id_if_any / num_candidates_after_filter で
  // 観察できるよう診断情報を返す (lio_node 側で throttle ログに使える)。
  struct AddAndQueryResult
  {
    bool detected = false;
    LoopCandidate candidate;
    // ---- 診断 ----
    // ring-key 粗フィルタ + recent_kf 除外を通過した候補数。 0 なら「DB が空 / 全部
    // recent_kf に弾かれた」。
    int num_candidates_after_filter = 0;
    // 候補があったとき、 ベストマッチの Scan Context 距離 (低いほど類似)。
    // 0 候補のときは std::numeric_limits<double>::infinity()。
    double best_distance = 0.0;
    // ベストマッチのキーフレーム ID (best_distance に対応)。 候補ゼロのとき未定義。
    uint32_t best_match_kf_id = 0;
    // 検索後に DB に積まれているキーフレーム総数。
    std::size_t database_size_after = 0;
  };
  AddAndQueryResult addKeyframeAndQuery(
    uint32_t keyframe_id,
    const PointCloud & scan_body,
    const Eigen::Isometry3d & pose_world_body);

  // これまで検出したすべての loop 候補 (古い順)。
  const std::vector<LoopCandidate> & detectedCandidates() const { return detected_candidates_; }
  std::size_t keyframeCount() const { return keyframe_database_.size(); }

  std::string describe() const;

private:
  struct DatabaseEntry
  {
    uint32_t keyframe_id;
    ScanContextDescriptor descriptor;
    Eigen::Isometry3d pose_world_body;
  };

  Config config_;
  std::vector<DatabaseEntry> keyframe_database_;
  std::vector<LoopCandidate> detected_candidates_;
};

}  // namespace pylot_lio

#endif  // PYLOT_LIO__LOOP__SCAN_CONTEXT_LOOP_DETECTOR_HPP_
