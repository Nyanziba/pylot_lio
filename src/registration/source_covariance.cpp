// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#include "pylot_lio/registration/source_covariance.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

#include <Eigen/Eigenvalues>

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pylot_lio
{

std::vector<Eigen::Matrix3d> computeSourceCovariances(
  const PointCloud & source_cloud_body,
  int num_neighbors,
  double plane_epsilon)
{
  const int num_points = static_cast<int>(source_cloud_body.points.size());

  // 近傍が足りない点の既定値: 等方 (plane_epsilon * I)。 重みをほぼ効かせない。
  std::vector<Eigen::Matrix3d> source_covariances(
    num_points, Eigen::Matrix3d::Identity() * plane_epsilon);

  // 近傍点数 k は最低 3 点ないと平面方向 2 + 法線 1 を分離できない。
  if (num_points < 3 || num_neighbors < 3) {
    return source_covariances;
  }

  // PCL KdTree を source 点群上に構築。 NaN は除外しつつ、 元 index への
  // 対応を保つため pcl_index -> original_index の写像を持つ。
  pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>());
  pcl_cloud->points.reserve(num_points);
  std::vector<int> original_index_of;
  original_index_of.reserve(num_points);
  for (int point_index = 0; point_index < num_points; ++point_index) {
    const Point & point = source_cloud_body.points[point_index];
    if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
      continue;
    }
    pcl_cloud->points.emplace_back(point.x, point.y, point.z);
    original_index_of.push_back(point_index);
  }
  pcl_cloud->width = static_cast<std::uint32_t>(pcl_cloud->points.size());
  pcl_cloud->height = 1;

  if (pcl_cloud->points.size() < 3) {
    return source_covariances;
  }

  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
  kdtree.setInputCloud(pcl_cloud);

  const int effective_k =
    std::min(num_neighbors, static_cast<int>(pcl_cloud->points.size()));

  for (std::size_t pcl_index = 0; pcl_index < pcl_cloud->points.size(); ++pcl_index) {
    std::vector<int> neighbor_indices(effective_k);
    std::vector<float> neighbor_squared_distances(effective_k);
    const int found = kdtree.nearestKSearch(
      pcl_cloud->points[pcl_index], effective_k, neighbor_indices,
      neighbor_squared_distances);
    if (found < 3) {
      continue;  // 既定の plane_epsilon * I のまま。
    }

    // 近傍点の平均と (母) 共分散を計算。
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (int neighbor = 0; neighbor < found; ++neighbor) {
      const auto & neighbor_point = pcl_cloud->points[neighbor_indices[neighbor]];
      mean += Eigen::Vector3d(neighbor_point.x, neighbor_point.y, neighbor_point.z);
    }
    mean /= static_cast<double>(found);

    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    for (int neighbor = 0; neighbor < found; ++neighbor) {
      const auto & neighbor_point = pcl_cloud->points[neighbor_indices[neighbor]];
      const Eigen::Vector3d deviation =
        Eigen::Vector3d(neighbor_point.x, neighbor_point.y, neighbor_point.z) - mean;
      covariance.noalias() += deviation * deviation.transpose();
    }
    covariance /= static_cast<double>(found);

    // 平面性正則化 (Segal 2009 の plane-to-plane): 固有値を (epsilon, 1, 1) に置換。
    // SelfAdjointEigenSolver の固有値は昇順なので index 0 = 最小 = 法線方向。
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
    if (solver.info() != Eigen::Success) {
      continue;
    }
    Eigen::Vector3d regularized_eigenvalues(plane_epsilon, 1.0, 1.0);
    const Eigen::Matrix3d eigenvectors = solver.eigenvectors();
    const Eigen::Matrix3d regularized_covariance =
      eigenvectors * regularized_eigenvalues.asDiagonal() * eigenvectors.transpose();

    source_covariances[original_index_of[pcl_index]] = regularized_covariance;
  }

  return source_covariances;
}

}  // namespace pylot_lio
