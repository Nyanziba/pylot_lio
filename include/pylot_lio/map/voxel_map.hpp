// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#ifndef PYLOT_LIO__MAP__VOXEL_MAP_HPP_
#define PYLOT_LIO__MAP__VOXEL_MAP_HPP_

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>

#include <Eigen/Core>

#include "pylot_lio/map/gaussian_voxel.hpp"
#include "pylot_lio/map/i_point_cloud_map.hpp"

namespace pylot_lio
{

// 一定ボクセル単位で点群統計 (平均、共分散、サンプル数) をオンライン更新するマップ。
// 各ボクセルが 1 つの「ガウス分布」を表現するため、GICP 系のマッチングと相性が良い。
// 共分散の最小固有値を上から clamp することで、平面方向の情報を強調する。
class VoxelMap : public IPointCloudMap
{
public:
  struct GaussianCell
  {
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    Eigen::Matrix3d covariance = Eigen::Matrix3d::Identity();
    int sample_count = 0;
  };

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

  struct Config
  {
    double voxel_size_m = 0.5;
    // 1 にしておけば「1 点でも入ったボクセル」が近傍探索の対象になる。
    // サンプルが 1 個だと共分散はゼロ行列になるが covariance_eigen_floor で持ち上げる。
    // 大きくすると統計が安定する代わりに、初期スキャン群で対応点が極端に減り、
    // gicp_only や IESKF が初期推定そのままを返して姿勢が原点に張り付く症状になる。
    int min_points_per_cell_for_covariance = 1;
    int max_points_per_cell = 50;
    double covariance_eigen_floor = 1e-3;
    int neighbor_search_radius_voxels = 1;
    std::size_t max_total_cells = 200000;
  };

  explicit VoxelMap(const Config & config);

  void insertScan(
    const PointCloud & scan_in_world,
    const Eigen::Isometry3d & pose_world_body) override;

  PointCorrespondence findNearestNeighbor(
    const Eigen::Vector3d & query_world) const override;

  std::size_t size() const override;

  PointCloudPtr toPointCloud() const override;

  std::string describe() const override;

  // テスト用に内部統計へアクセスするための補助 (read-only)。
  const std::unordered_map<VoxelKey, GaussianCell, VoxelKeyHash> & cells() const;

  // voxel_size / covariance_eigen_floor 等を read-only で参照する (GPU VGICP の
  // voxel 表構築や、 外部から map のボクセル幾何を知りたい用途)。
  const Config & config() const { return config_; }

  // 各ボクセルの保存済みガウス分布 (mean/covariance/count) を中間表現で返す。
  // metal_vgicp が「マップ非依存で target を受け取る」 ために使う。 共分散はオンライン
  // 更新で蓄積したものをそのまま返す (toPointCloud() は 1 ボクセル 1 点なので不可)。
  // sample_count が min_points_per_cell_for_covariance 未満のセルは除外する。
  std::vector<GaussianVoxel> toGaussianVoxels() const;

private:
  VoxelKey computeKey(const Eigen::Vector3d & position_world) const;

  void updateCellOnline(GaussianCell & cell, const Eigen::Vector3d & new_point);

  // ボクセルが上限を超えたら最も古いセルを落とす単純な FIFO 風削除。
  // 実装の単純さを優先し O(N) スキャンで済ませる (運用上 cells 数は十数万程度を想定)。
  void evictOldCellsIfNeeded();

  Config config_;
  std::unordered_map<VoxelKey, GaussianCell, VoxelKeyHash> cells_;
};

}  // namespace pylot_lio

#endif  // PYLOT_LIO__MAP__VOXEL_MAP_HPP_
