// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#include "pylot_lio/map/gaussian_voxel.hpp"

#include <cmath>
#include <cstdint>
#include <tuple>
#include <unordered_map>

#include <Eigen/Eigenvalues>

namespace pylot_lio
{

namespace
{

// ボクセル整数座標 → 64bit パックキー (unordered_map 用)。 各軸 21bit。
struct VoxelKey
{
  std::int64_t x;
  std::int64_t y;
  std::int64_t z;
  bool operator==(const VoxelKey & other) const noexcept
  {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct VoxelKeyHash
{
  std::size_t operator()(const VoxelKey & key) const noexcept
  {
    const auto h1 = std::hash<std::int64_t>{}(key.x);
    const auto h2 = std::hash<std::int64_t>{}(key.y);
    const auto h3 = std::hash<std::int64_t>{}(key.z);
    return h1 ^ (h2 * 0x9E3779B97F4A7C15ULL) ^ (h3 * 0xBF58476D1CE4E5B9ULL);
  }
};

struct Accumulator
{
  Eigen::Vector3d sum = Eigen::Vector3d::Zero();
  Eigen::Matrix3d sum_outer = Eigen::Matrix3d::Zero();  // Σ p pᵀ
  int count = 0;
};

}  // namespace

std::vector<GaussianVoxel> voxelizePointsToGaussians(
  const PointCloud & cloud_world,
  double voxel_size_m,
  double covariance_eigen_floor,
  int min_points_per_voxel)
{
  std::vector<GaussianVoxel> voxels;
  if (cloud_world.points.empty() || voxel_size_m <= 0.0) {
    return voxels;
  }

  std::unordered_map<VoxelKey, Accumulator, VoxelKeyHash> bins;
  bins.reserve(cloud_world.points.size() / 4 + 1);
  const double inverse_voxel_size = 1.0 / voxel_size_m;

  for (const Point & point : cloud_world.points) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
      continue;
    }
    const Eigen::Vector3d position(point.x, point.y, point.z);
    const VoxelKey key{
      static_cast<std::int64_t>(std::floor(position.x() * inverse_voxel_size)),
      static_cast<std::int64_t>(std::floor(position.y() * inverse_voxel_size)),
      static_cast<std::int64_t>(std::floor(position.z() * inverse_voxel_size))
    };
    Accumulator & accumulator = bins[key];
    accumulator.sum += position;
    accumulator.sum_outer.noalias() += position * position.transpose();
    accumulator.count += 1;
  }

  voxels.reserve(bins.size());
  for (const auto & [key, accumulator] : bins) {
    (void)key;
    if (accumulator.count < min_points_per_voxel) {
      continue;
    }
    const double count = static_cast<double>(accumulator.count);
    const Eigen::Vector3d mean = accumulator.sum / count;
    // 標本 (母) 共分散: E[p pᵀ] - μ μᵀ。
    Eigen::Matrix3d covariance = accumulator.sum_outer / count - mean * mean.transpose();

    // 固有値を eigen_floor 以上にクランプして退化 (1〜2 点や同一平面) を防ぐ。
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
    if (solver.info() == Eigen::Success) {
      Eigen::Vector3d eigenvalues = solver.eigenvalues();
      for (int i = 0; i < 3; ++i) {
        if (eigenvalues(i) < covariance_eigen_floor) {
          eigenvalues(i) = covariance_eigen_floor;
        }
      }
      const Eigen::Matrix3d eigenvectors = solver.eigenvectors();
      covariance = eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();
    } else {
      covariance = Eigen::Matrix3d::Identity() * covariance_eigen_floor;
    }

    GaussianVoxel voxel;
    voxel.mean = mean;
    voxel.covariance = covariance;
    voxel.count = count;
    voxels.push_back(voxel);
  }
  return voxels;
}

}  // namespace pylot_lio
