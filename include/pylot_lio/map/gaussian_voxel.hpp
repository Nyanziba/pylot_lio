// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#ifndef PYLOT_LIO__MAP__GAUSSIAN_VOXEL_HPP_
#define PYLOT_LIO__MAP__GAUSSIAN_VOXEL_HPP_

#include <vector>

#include <Eigen/Core>

#include "pylot_lio/types.hpp"

namespace pylot_lio
{

// ボクセル単位のガウス分布 (VGICP の target 分布)。 マップ実装非依存の中間表現で、
// metal_vgicp が「どのマップからでも同じ形で target を受け取る」 ために使う。
struct GaussianVoxel
{
  Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  Eigen::Matrix3d covariance = Eigen::Matrix3d::Identity();
  // そのボクセルに寄与した点数 (多重解像度の平行軸定理マージで重みに使う)。
  double count = 0.0;
};

// 生点群を一辺 voxel_size のボクセルに振り分け、 各ボクセルの平均・標本共分散を計算して
// GaussianVoxel 列を返す。 共分散は固有値を eigen_floor 以上にクランプして退化を防ぐ
// (VoxelMap が内部で行うのと同じ正則化)。 点数が min_points 未満のボクセルは捨てる。
//
// voxel_map / normal_map のように「保存済みガウス分布」 を直接出せるマップでは使わない
// (それらの toPointCloud() は 1 ボクセル 1 点なので再ボクセル化すると共分散が退化する)。
// voxel_random_map / voxel_keyframe_submap のように「ボクセルあたり複数の生点」 を返す
// マップの toPointCloud() をボクセル化してガウス分布を復元するために使う。
std::vector<GaussianVoxel> voxelizePointsToGaussians(
  const PointCloud & cloud_world,
  double voxel_size_m,
  double covariance_eigen_floor,
  int min_points_per_voxel);

}  // namespace pylot_lio

#endif  // PYLOT_LIO__MAP__GAUSSIAN_VOXEL_HPP_
