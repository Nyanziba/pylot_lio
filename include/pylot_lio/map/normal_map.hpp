// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#ifndef PYLOT_LIO__MAP__NORMAL_MAP_HPP_
#define PYLOT_LIO__MAP__NORMAL_MAP_HPP_

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "pylot_lio/map/gaussian_voxel.hpp"
#include "pylot_lio/map/i_point_cloud_map.hpp"

namespace pylot_lio
{

// 各点に「法線方向の異方性」を共分散として埋め込んだマップ。
// 内部は VoxelMap と同じガウス分布を保持するが、findNearestNeighbor は
// 法線方向以外を縮小した「平面寄り」共分散を返すため点対面 (point-to-plane) 風になる。
class NormalMap : public IPointCloudMap
{
public:
  struct Config
  {
    double voxel_size_m = 0.5;
    // 同じ理由で 1 が安全 (voxel_map.hpp の同名フィールドのコメント参照)。
    int min_points_per_cell = 1;
    double tangent_covariance_floor = 1e-2;  // 平面方向の最小固有値
    double normal_covariance_ceiling = 1e-3; // 法線方向の最大固有値
    int neighbor_search_radius_voxels = 1;
    std::size_t max_total_cells = 200000;
  };

  explicit NormalMap(const Config & config);

  void insertScan(
    const PointCloud & scan_in_world,
    const Eigen::Isometry3d & pose_world_body) override;

  PointCorrespondence findNearestNeighbor(
    const Eigen::Vector3d & query_world) const override;

  std::size_t size() const override;
  PointCloudPtr toPointCloud() const override;
  std::string describe() const override;

  // 各ボクセルの保存済みガウス分布 (point-to-plane 整形済み共分散) を中間表現で返す。
  // metal_vgicp が target として受け取るのに使う (toPointCloud() は 1 ボクセル 1 点
  // なので共分散が復元できない)。 sample_count が min_points_per_cell 未満は除外。
  std::vector<GaussianVoxel> toGaussianVoxels() const;

private:
  struct Cell
  {
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    Eigen::Matrix3d covariance = Eigen::Matrix3d::Identity();
    int sample_count = 0;
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
  Eigen::Matrix3d shapeAsPointToPlane(const Eigen::Matrix3d & raw_covariance) const;

  Config config_;
  std::unordered_map<Key, Cell, KeyHash> cells_;
};

}  // namespace pylot_lio

#endif  // PYLOT_LIO__MAP__NORMAL_MAP_HPP_
