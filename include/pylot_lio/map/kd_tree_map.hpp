// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#ifndef PYLOT_LIO__MAP__KD_TREE_MAP_HPP_
#define PYLOT_LIO__MAP__KD_TREE_MAP_HPP_

#include <memory>
#include <string>
#include <vector>

#include <pcl/kdtree/kdtree_flann.h>

#include "pylot_lio/map/i_point_cloud_map.hpp"

namespace pylot_lio
{

// PCL の FLANN KdTree を 1 個持ち、scan 追加のたびに点群を蓄積して再構築する素朴な実装。
// 共分散は近傍 k 点から PCA で計算する。VoxelMap や NormalMap に比べてマップ更新コストが高いが、
// 「アルゴリズム比較における素朴ベースライン」としての位置付け。
class KdTreeMap : public IPointCloudMap
{
public:
  struct Config
  {
    int neighbor_count_for_covariance = 8;
    double covariance_eigen_floor = 1e-3;
    int rebuild_every_n_insert = 1;        // 1 = 毎フレーム再構築 (簡易実装)
    std::size_t max_total_points = 200000;
    double subsample_voxel_size_m = 0.2;   // 入力 scan を内部でダウンサンプル
  };

  explicit KdTreeMap(const Config & config);

  void insertScan(
    const PointCloud & scan_in_world,
    const Eigen::Isometry3d & pose_world_body) override;

  PointCorrespondence findNearestNeighbor(
    const Eigen::Vector3d & query_world) const override;

  std::size_t size() const override;
  PointCloudPtr toPointCloud() const override;
  std::string describe() const override;

private:
  void rebuildKdTree();

  Config config_;
  PointCloudPtr accumulated_cloud_;
  std::unique_ptr<pcl::KdTreeFLANN<Point>> kd_tree_;
  int inserts_since_rebuild_;
};

}  // namespace pylot_lio

#endif  // PYLOT_LIO__MAP__KD_TREE_MAP_HPP_
