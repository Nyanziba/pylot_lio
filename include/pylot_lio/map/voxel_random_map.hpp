// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#ifndef PYLOT_LIO__MAP__VOXEL_RANDOM_MAP_HPP_
#define PYLOT_LIO__MAP__VOXEL_RANDOM_MAP_HPP_

#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "pylot_lio/map/i_point_cloud_map.hpp"

namespace pylot_lio
{

// 各ボクセルに最大 K 個の「実測点」をリザーバサンプリング (Vitter Algorithm R) で保持するマップ。
//
// VoxelMap が平均 + 共分散 1 個でガウス近似するのに対し、こちらは生の点群を保つので
//   - 1 ボクセルに 2 枚の面が偶然入った場合でも両方の情報が残る
//   - メモリは max_points_per_voxel * cells_ で有界
//   - findNearestNeighbor では「クエリ近傍ボクセル群の点」から最近傍 k 個を取り出して
//     その場で平均 + 共分散を計算する (KdTreeMap のローカル近傍 PCA と同じ考え方)
// という第 4 の表現を提供する。
//
// 参考: 標準的なリザーバサンプリング (Vitter, "Random sampling with a reservoir", 1985) の
// アルゴリズム R を 0-indexed で書き起こした自前実装。
class VoxelRandomMap : public IPointCloudMap
{
public:
  struct Config
  {
    double voxel_size_m = 0.5;
    int max_points_per_voxel = 8;
    int neighbor_search_radius_voxels = 1;
    int k_nearest_for_covariance = 8;
    double covariance_eigen_floor = 1e-3;
    std::size_t max_total_cells = 200000;
    uint64_t random_seed = 0x9E3779B97F4A7C15ULL;
  };

  explicit VoxelRandomMap(const Config & config);

  void insertScan(
    const PointCloud & scan_in_world,
    const Eigen::Isometry3d & pose_world_body) override;

  PointCorrespondence findNearestNeighbor(
    const Eigen::Vector3d & query_world) const override;

  std::size_t size() const override;
  PointCloudPtr toPointCloud() const override;
  std::string describe() const override;

private:
  struct Reservoir
  {
    std::vector<Eigen::Vector3d> retained_points;
    uint64_t observed_count = 0;
  };

  struct Key
  {
    int64_t x;
    int64_t y;
    int64_t z;
    bool operator==(const Key & other) const noexcept
    {
      return x == other.x && y == other.y && z == other.z;
    }
  };

  struct KeyHash
  {
    std::size_t operator()(const Key & key) const noexcept;
  };

  Key computeKey(const Eigen::Vector3d & position_world) const;

  // Vitter Algorithm R を 0-indexed で書き直したリザーバ更新。
  void offerPointToReservoir(Reservoir & reservoir, const Eigen::Vector3d & new_point) const;

  void evictOldCellsIfNeeded();

  Config config_;
  std::unordered_map<Key, Reservoir, KeyHash> cells_;
  mutable std::mt19937_64 random_engine_;
};

}  // namespace pylot_lio

#endif  // PYLOT_LIO__MAP__VOXEL_RANDOM_MAP_HPP_
