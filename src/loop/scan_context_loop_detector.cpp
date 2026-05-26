// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#include "pylot_lio/loop/scan_context_loop_detector.hpp"

#include <algorithm>
#include <limits>
#include <sstream>
#include <utility>

namespace pylot_lio
{

ScanContextLoopDetector::ScanContextLoopDetector(const Config & config)
: config_(config)
{
}

ScanContextLoopDetector::AddAndQueryResult
ScanContextLoopDetector::addKeyframeAndQuery(
  uint32_t keyframe_id,
  const PointCloud & scan_body,
  const Eigen::Isometry3d & pose_world_body)
{
  AddAndQueryResult query_result;

  // 1) descriptor を計算して DB エントリ作成 (検索は DB に挿入する前に行うのが鉄則)
  DatabaseEntry new_entry;
  new_entry.keyframe_id = keyframe_id;
  new_entry.descriptor = computeScanContext(scan_body, config_.descriptor);
  new_entry.pose_world_body = pose_world_body;

  // 2) 検索: 自分より exclude_recent_kf 以上古いキーフレームのみが対象
  // DB は挿入順 = keyframe 順なので、 後ろから走査する
  const int recent_skip = std::max(0, config_.exclude_recent_kf);

  // ring key L2 距離で粗フィルタ → 上位 K 個に対し column-shift マッチ
  std::vector<std::pair<double, std::size_t>> ring_key_ranking;
  ring_key_ranking.reserve(keyframe_database_.size());
  for (std::size_t database_index = 0; database_index < keyframe_database_.size();
       ++database_index)
  {
    const DatabaseEntry & candidate_entry = keyframe_database_[database_index];
    // exclude_recent: 「現 keyframe id - 候補 id」が recent_skip 未満なら除外
    if (keyframe_id < static_cast<uint32_t>(recent_skip) ||
        candidate_entry.keyframe_id >
        keyframe_id - static_cast<uint32_t>(recent_skip))
    {
      continue;
    }
    const double distance =
      ringKeyDistance(new_entry.descriptor.ring_key, candidate_entry.descriptor.ring_key);
    ring_key_ranking.emplace_back(distance, database_index);
  }

  const int top_k = std::max(1, config_.ring_key_top_k);
  if (static_cast<int>(ring_key_ranking.size()) > top_k) {
    std::nth_element(
      ring_key_ranking.begin(), ring_key_ranking.begin() + top_k,
      ring_key_ranking.end(),
      [](const auto & lhs, const auto & rhs) { return lhs.first < rhs.first; });
    ring_key_ranking.resize(top_k);
  }

  query_result.num_candidates_after_filter =
    static_cast<int>(ring_key_ranking.size());

  double best_distance = std::numeric_limits<double>::infinity();
  std::size_t best_database_index = 0;
  double best_yaw_rad = 0.0;
  for (const auto & [_ring_distance, database_index] : ring_key_ranking) {
    (void)_ring_distance;
    const DatabaseEntry & candidate_entry = keyframe_database_[database_index];
    const ScanContextMatchResult match =
      matchScanContexts(new_entry.descriptor, candidate_entry.descriptor);
    if (match.distance < best_distance) {
      best_distance = match.distance;
      best_database_index = database_index;
      best_yaw_rad = match.estimated_yaw_rad;
    }
  }

  // 3) しきい値判定 → 候補確定
  if (std::isfinite(best_distance) && best_distance < config_.score_threshold) {
    LoopCandidate candidate;
    candidate.query_kf_id = keyframe_id;
    candidate.match_kf_id = keyframe_database_[best_database_index].keyframe_id;
    candidate.scan_context_distance = best_distance;
    candidate.estimated_yaw_rad = best_yaw_rad;
    candidate.query_pose_world_body = pose_world_body;
    candidate.match_pose_world_body =
      keyframe_database_[best_database_index].pose_world_body;
    detected_candidates_.push_back(candidate);
    query_result.detected = true;
    query_result.candidate = candidate;
  }

  // 診断情報: ベスト距離とベスト match を格納 (detected==false でも参照できる)
  query_result.best_distance = best_distance;
  if (std::isfinite(best_distance) && !ring_key_ranking.empty()) {
    query_result.best_match_kf_id =
      keyframe_database_[best_database_index].keyframe_id;
  }

  // 4) DB に挿入 (検索後に行うことで自己マッチを防ぐ)
  keyframe_database_.push_back(std::move(new_entry));
  query_result.database_size_after = keyframe_database_.size();
  return query_result;
}

std::string ScanContextLoopDetector::describe() const
{
  std::ostringstream oss;
  oss << "scan_context_loop_detector:"
      << "rings=" << config_.descriptor.num_rings
      << ",sectors=" << config_.descriptor.num_sectors
      << ",max_radius=" << config_.descriptor.max_radius_m
      << ",threshold=" << config_.score_threshold
      << ",exclude_recent=" << config_.exclude_recent_kf
      << ",db=" << keyframe_database_.size()
      << ",detected=" << detected_candidates_.size();
  return oss.str();
}

}  // namespace pylot_lio
