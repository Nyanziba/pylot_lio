// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#ifndef PYLOT_LIO__MAP__VOXEL_KEYFRAME_SUBMAP_MAP_HPP_
#define PYLOT_LIO__MAP__VOXEL_KEYFRAME_SUBMAP_MAP_HPP_

#include <cstddef>
#include <cstdint>
#include <deque>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "pylot_lio/map/i_point_cloud_map.hpp"

namespace pylot_lio
{

// キーフレーム単位で点群を管理する submap マップ。
//
// 大局構造:
//   1. スライディングウィンドウ: 直近 N 個のキーフレームの点群だけを registration target
//      として保持する。新しいキーフレームが入ったら最古を退場させ、grid から該当
//      キーフレーム ID をもつエントリだけを削除する。
//      → これにより registration target が常にロボット周辺の最新ジオメトリだけになる。
//   2. 確定 submap: M 個キーフレームが溜まるたびに、その M 個を 1 個の Submap として
//      内部 std::vector に追加する。これは将来 loop closure / pose graph に渡す単位。
//      今回は publish/save しない (将来 PR で loop_closure 検出を追加する想定)。
//
// IPointCloudMap::insertScan は「新しいキーフレーム 1 個」を受け取るセマンティクスにする。
// lio_node は keyframe_selector が採択したスキャンに対してだけ insertScan を呼ぶ慣習に
// なっており、本クラスは「呼ばれた回数 = キーフレーム数」を前提に kf_id を発番する。
class VoxelKeyframeSubmapManager : public IPointCloudMap
{
public:
  struct VoxelKey
  {
    int64_t x;
    int64_t y;
    int64_t z;
    bool operator==(const VoxelKey & other) const noexcept
    {
      return x == other.x && y == other.y && z == other.z;
    }
  };

  struct VoxelKeyHash
  {
    std::size_t operator()(const VoxelKey & key) const noexcept;
  };

  // grid セル 1 個に貯まる「点 + どのキーフレーム由来か」
  struct PointEntry
  {
    Eigen::Vector3d point_world;
    uint32_t keyframe_id;
  };

  // 確定 submap のスナップショット。将来の loop closure 用。
  struct Submap
  {
    uint32_t submap_id;
    std::vector<uint32_t> keyframe_ids;
    Eigen::Isometry3d anchor_pose_world_body;  // 最後のキーフレーム時点の姿勢
    PointCloud merged_cloud_world;             // M 個のキーフレーム点群を world 系で連結
  };

  // 各キーフレームの記録 (loop closure 用に anchor pose を持っておく)
  struct KeyframeRecord
  {
    uint32_t keyframe_id;
    Eigen::Isometry3d pose_world_body;
  };

  // セル内の点の保持戦略。
  //   KeepUpToN : insertScan で push_back し続け max_points_per_cell まで貯める (既定、 後方互換)
  //   RandomOne : セルに 1 点だけ保持し、 新点が来たら reservoir sampling で確率的置換
  //               → メモリ/PCD サイズが激減、 ICP の対応点も「実点」を保つ (重心化しない)
  enum class CellSamplingMode
  {
    KeepUpToN,
    RandomOne,
  };

  struct Config
  {
    double voxel_size_m = 0.5;
    int min_points_per_cell_for_search = 1;
    // セル内の点に上限を設けるとメモリは抑えられるが、近傍探索の質が落ちる。
    // 0 以下なら無制限。 cell_sampling_mode=RandomOne では事実上 1。
    int max_points_per_cell = 32;
    int neighbor_search_radius_voxels = 1;
    // スライディングウィンドウ: 直近何個のキーフレームを registration target に含めるか
    int sliding_window_size = 20;
    // 確定 submap: 何個キーフレームを束ねて 1 つの確定 submap にするか
    int finalize_size = 20;
    // 共分散の最小固有値の床 (退化方向の暴走防止)
    double covariance_eigen_floor = 1e-3;
    // 共分散推定に使う最小近傍点数
    int k_nearest_for_covariance = 8;
    // セル内の点保持戦略
    CellSamplingMode cell_sampling_mode = CellSamplingMode::KeepUpToN;
    // RandomOne モードで使う乱数 seed (0 でマシン乱数)
    uint32_t random_seed = 12345u;
  };

  explicit VoxelKeyframeSubmapManager(const Config & config);

  // IPointCloudMap: 「新しいキーフレーム 1 個」として呼ばれる。
  // kf_id をインクリメントして付与し、grid にエントリを足す。
  void insertScan(
    const PointCloud & scan_in_world,
    const Eigen::Isometry3d & pose_world_body) override;

  PointCorrespondence findNearestNeighbor(
    const Eigen::Vector3d & query_world) const override;

  // size() は IPointCloudMap の意味に合わせて「現在 grid にあるセル数」を返す。
  std::size_t size() const override;

  // 現在 grid にあるセル中心点を吐き出す (可視化用)。
  PointCloudPtr toPointCloud() const override;

  std::string describe() const override;

  // ---- submap 拡張 API (将来 loop closure / pose graph 用) ----

  // active submap (sliding window) に現在含まれるキーフレーム ID の列。
  std::vector<uint32_t> activeKeyframeIds() const;

  // 確定済 submap 一覧 (古い順)。
  const std::vector<Submap> & finalizedSubmaps() const;

  // 「形成したマップ全体」の点群。 finalized submap 群 (各 submap の merged_cloud_world)
  // と active sliding window の grid を 1 つの cloud に連結したもの。
  //
  // 注意:
  //   - finalized 部分は finalize 時点の pose で固定 (PGO 後の updatePoses では更新されない)
  //   - active 部分は最新 pose に追従 (updatePoses で grid 再構築される)
  //   - サイズが大きくなるので呼び出し頻度には注意 (lio_node 側で 5 秒タイマーで制限)
  PointCloudPtr toFullMapPointCloud() const;

  // PGO 結果による keyframe pose の事後修正。
  //   updated_poses: { keyframe_id -> 新しい pose_world_body }
  //
  // 影響範囲:
  //   - keyframe_records_ の pose を更新
  //   - active sliding window 内の keyframe については、 grid 内の world 座標点を
  //     T_new · T_old^{-1} で剛体変換し直して再投入する (grid を作り直す)
  //   - 既に finalize 済みの submap (merged_cloud / anchor_pose) は今回は触らない
  //     (finalize 時点でスナップショットされたものとして扱う)
  void updatePoses(
    const std::vector<std::pair<uint32_t, Eigen::Isometry3d>> & updated_poses);

  std::size_t activeKeyframeCount() const { return active_keyframe_ids_.size(); }
  std::size_t finalizedSubmapCount() const { return finalized_submaps_.size(); }
  std::size_t totalKeyframesInserted() const { return next_keyframe_id_; }

private:
  VoxelKey computeKey(const Eigen::Vector3d & position_world) const;

  // 古いキーフレームを退場させ、grid から該当 keyframe_id のエントリを除去する。
  void evictKeyframeFromGrid(uint32_t keyframe_id_to_evict);

  // 直近 finalize_size 個のキーフレームを 1 個の Submap にまとめて保存する。
  void maybeFinalizeSubmap();

  Config config_;
  std::unordered_map<VoxelKey, std::vector<PointEntry>, VoxelKeyHash> grid_;

  // sliding window 用: 直近 N 個の id のみを保持し、N を超えたら front から退場
  std::deque<uint32_t> active_keyframe_ids_;

  // 全キーフレームの記録 (確定 submap 構築のために pose と cloud が必要)
  std::vector<KeyframeRecord> keyframe_records_;
  // 各キーフレームの世界座標点群 (pose は keyframe_records_ 側に持つ)
  std::vector<PointCloud> keyframe_clouds_world_;

  std::vector<Submap> finalized_submaps_;

  uint32_t next_keyframe_id_ = 0;
  uint32_t next_submap_id_ = 0;
  // 最後に finalize したキーフレーム数 (これと現キーフレーム総数の差が finalize_size に到達したら確定)
  uint32_t keyframes_at_last_finalize_ = 0;

  // RandomOne モードで使う乱数エンジン (insertScan / updatePoses で reservoir sampling)。
  // mutable で findNearestNeighbor の const もそのまま維持 (ここでは使わない)。
  mutable std::mt19937 random_engine_;
};

}  // namespace pylot_lio

#endif  // PYLOT_LIO__MAP__VOXEL_KEYFRAME_SUBMAP_MAP_HPP_
