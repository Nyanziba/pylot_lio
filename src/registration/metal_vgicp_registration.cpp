// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#include "pylot_lio/registration/metal_vgicp_registration.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <sstream>
#include <tuple>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "pylot_lio/gpu/metal_vgicp_linearizer.hpp"
#include "pylot_lio/lie_algebra.hpp"
#include "pylot_lio/map/voxel_map.hpp"

namespace pylot_lio
{

namespace
{

// source 点群の各点について body フレームの局所共分散 C_s を推定する。
// plain_gicp の computeSourceCovariances と同方式 (近傍 k 点の共分散を平面正則化)。
// NOTE: plain_gicp と重複している。 将来 gicp_source_covariance として共通化候補。
std::vector<Eigen::Matrix3d> computeSourceCovariances(
  const PointCloud & source_cloud_body, int num_neighbors, double plane_epsilon)
{
  const int num_points = static_cast<int>(source_cloud_body.points.size());
  std::vector<Eigen::Matrix3d> covariances(
    num_points, Eigen::Matrix3d::Identity() * plane_epsilon);
  if (num_points < 3 || num_neighbors < 3) {
    return covariances;
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>());
  pcl_cloud->points.reserve(num_points);
  std::vector<int> original_index_of;
  original_index_of.reserve(num_points);
  for (int i = 0; i < num_points; ++i) {
    const Point & p = source_cloud_body.points[i];
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
      continue;
    }
    pcl_cloud->points.emplace_back(p.x, p.y, p.z);
    original_index_of.push_back(i);
  }
  pcl_cloud->width = static_cast<std::uint32_t>(pcl_cloud->points.size());
  pcl_cloud->height = 1;
  if (pcl_cloud->points.size() < 3) {
    return covariances;
  }

  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
  kdtree.setInputCloud(pcl_cloud);
  const int k = std::min(num_neighbors, static_cast<int>(pcl_cloud->points.size()));

  std::vector<int> neighbor_indices(k);
  std::vector<float> neighbor_sq_dists(k);
  for (std::size_t query = 0; query < pcl_cloud->points.size(); ++query) {
    if (kdtree.nearestKSearch(
        pcl_cloud->points[query], k, neighbor_indices, neighbor_sq_dists) < 3)
    {
      continue;
    }
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (int n = 0; n < k; ++n) {
      const auto & pt = pcl_cloud->points[neighbor_indices[n]];
      mean += Eigen::Vector3d(pt.x, pt.y, pt.z);
    }
    mean /= static_cast<double>(k);
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (int n = 0; n < k; ++n) {
      const auto & pt = pcl_cloud->points[neighbor_indices[n]];
      const Eigen::Vector3d diff = Eigen::Vector3d(pt.x, pt.y, pt.z) - mean;
      cov += diff * diff.transpose();
    }
    cov /= static_cast<double>(k);

    // 平面正則化 (Segal 2009): 固有値を (epsilon, 1, 1) に置換。
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
    Eigen::Matrix3d eigenvectors = solver.eigenvectors();
    Eigen::Vector3d replaced(plane_epsilon, 1.0, 1.0);  // 昇順固有値: 最小=法線方向
    Eigen::Matrix3d regularized =
      eigenvectors * replaced.asDiagonal() * eigenvectors.transpose();
    covariances[original_index_of[query]] = regularized;
  }
  return covariances;
}

// 指定解像度 level_voxel_size で VoxelMap のセル群を再ビン (平行軸定理でマージ) し、
// VgicpVoxelTable を作る。 各セルの 2 次モーメント (C_i + μ_i μ_iᵀ) を count 重みで
// 合算し、 粗ボクセルの分布 (μ, C) を得る。 level_voxel_size = base のときは各セルが
// 自分のボクセルに一意に落ちるので実質そのまま (= 最細レベル)。
gpu::VgicpVoxelTable buildLevelTable(
  const VoxelMap & voxel_map, double level_voxel_size)
{
  const double eigen_floor = voxel_map.config().covariance_eigen_floor;
  const int min_points = voxel_map.config().min_points_per_cell_for_covariance;
  const double inv_level = 1.0 / level_voxel_size;

  struct Accumulator
  {
    Eigen::Vector3i coord;
    double total_count = 0.0;
    Eigen::Vector3d sum_weighted_mean = Eigen::Vector3d::Zero();      // Σ n_i μ_i
    Eigen::Matrix3d sum_weighted_2nd_moment = Eigen::Matrix3d::Zero();  // Σ n_i (C_i + μ_iμ_iᵀ)
  };
  // 粗ボクセル座標 → accumulator。 std::map で決定的順序にする (テスト再現性)。
  std::map<std::tuple<int, int, int>, Accumulator> bins;

  for (const auto & [key, cell] : voxel_map.cells()) {
    if (cell.sample_count < min_points) {
      continue;
    }
    const int cx = static_cast<int>(std::floor(cell.mean.x() * inv_level));
    const int cy = static_cast<int>(std::floor(cell.mean.y() * inv_level));
    const int cz = static_cast<int>(std::floor(cell.mean.z() * inv_level));
    Accumulator & acc = bins[std::make_tuple(cx, cy, cz)];
    acc.coord = Eigen::Vector3i(cx, cy, cz);
    const double n = static_cast<double>(cell.sample_count);
    acc.total_count += n;
    acc.sum_weighted_mean += n * cell.mean;
    acc.sum_weighted_2nd_moment +=
      n * (cell.covariance + cell.mean * cell.mean.transpose());
  }

  std::vector<Eigen::Vector3i> coords;
  std::vector<Eigen::Vector3d> means;
  std::vector<Eigen::Matrix3d> covs;
  coords.reserve(bins.size());
  means.reserve(bins.size());
  covs.reserve(bins.size());

  for (const auto & [bin_key, acc] : bins) {
    if (acc.total_count <= 0.0) {
      continue;
    }
    const Eigen::Vector3d mean = acc.sum_weighted_mean / acc.total_count;
    // 平行軸定理: C = (Σ n_i (C_i + μ_iμ_iᵀ)) / N - μ μᵀ。
    Eigen::Matrix3d cov =
      acc.sum_weighted_2nd_moment / acc.total_count - mean * mean.transpose();
    // 対称化 (数値誤差で非対称になりうる) してから固有値 floor。
    cov = 0.5 * (cov + cov.transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
    Eigen::Vector3d eigenvalues = solver.eigenvalues();
    for (int axis = 0; axis < 3; ++axis) {
      if (eigenvalues(axis) < eigen_floor) {
        eigenvalues(axis) = eigen_floor;
      }
    }
    coords.push_back(acc.coord);
    means.push_back(mean);
    covs.push_back(
      solver.eigenvectors() * eigenvalues.asDiagonal() * solver.eigenvectors().transpose());
  }

  return gpu::VgicpVoxelTable::build(
    static_cast<float>(level_voxel_size), coords, means, covs);
}

// 多重解像度テーブルを「粗→細」の順で構築する。 level i の voxel_size は
// base * scaling^(levels-1-i) なので、 先頭が最も粗く末尾 (= base) が最も細かい。
std::vector<gpu::VgicpVoxelTable> buildMultiResolutionTables(
  const VoxelMap & voxel_map, int levels, double scaling_factor)
{
  const double base_voxel_size = voxel_map.config().voxel_size_m;
  const int clamped_levels = std::max(1, levels);
  std::vector<gpu::VgicpVoxelTable> tables;
  tables.reserve(clamped_levels);
  for (int level = clamped_levels - 1; level >= 0; --level) {
    const double level_voxel_size =
      base_voxel_size * std::pow(scaling_factor, static_cast<double>(level));
    tables.push_back(buildLevelTable(voxel_map, level_voxel_size));
  }
  return tables;
}

}  // namespace

MetalVgicpRegistration::MetalVgicpRegistration(const Config & config)
: config_(config), engine_(std::make_shared<gpu::MetalVgicpEngine>())
{
  // engine_ の ctor で device/PSO を 1 回構築する (Metal 無効ビルドでは isValid()=false)。
}

bool MetalVgicpRegistration::isAvailable()
{
#ifdef PYLOT_LIO_HAS_METAL
  return true;
#else
  return false;
#endif
}

MetalVgicpRegistration::AlignResult MetalVgicpRegistration::align(
  const PointCloud & source_cloud_body,
  const IPointCloudMap & map_world,
  const Eigen::Isometry3d & initial_transform_world_body)
{
  AlignResult result;
  result.transform_world_body = initial_transform_world_body;

  // target マップは VoxelMap のみ対応 (Step2 制約)。
  const VoxelMap * voxel_map = dynamic_cast<const VoxelMap *>(&map_world);
  if (voxel_map == nullptr) {
    result.converged = false;
    return result;
  }

  // source 点を Vector3d 配列に変換 (NaN 点も含めて並びを保つ。 linearizer 側で除外)。
  std::vector<Eigen::Vector3d> source_points;
  source_points.reserve(source_cloud_body.points.size());
  for (const Point & p : source_cloud_body.points) {
    source_points.emplace_back(p.x, p.y, p.z);
  }
  const std::vector<Eigen::Matrix3d> source_covariances = computeSourceCovariances(
    source_cloud_body, config_.source_covariance_num_neighbors,
    config_.source_covariance_plane_epsilon);

  // engine 無効 (Metal 無しビルド / デバイス無) なら未収束で返す。 これは factory が
  // plain_gicp へフォールバックする領域 (metal_vgicp はそもそも使われない想定)。
  if (!engine_ || !engine_->isValid()) {
    result.converged = false;
    return result;
  }

  // 多重解像度テーブルを「粗→細」で構築 (levels=1 なら単一解像度)。
  const std::vector<gpu::VgicpVoxelTable> level_tables =
    buildMultiResolutionTables(
      *voxel_map, config_.voxelmap_levels, config_.voxelmap_scaling_factor);
  if (level_tables.empty() || level_tables.back().capacity <= 0) {
    result.converged = false;
    return result;
  }

  // backend を align ごとに事前選択する (engine は有効):
  //   - 点数 >= gpu_min_points → GPU (大規模で GPU が勝つ)
  //   - それ未満 → CPU VGICP (起動オーバヘッドで GPU が不利な小規模)
  const bool use_gpu =
    static_cast<int>(source_points.size()) >= config_.gpu_min_points;
  if (use_gpu) {
    // source は全レベル共通なので 1 回だけアップロード。 target はレベルごとに差し替える。
    engine_->setSource(source_points, source_covariances);
  }

  const double base_voxel_size = voxel_map->config().voxel_size_m;

  Eigen::Isometry3d current = initial_transform_world_body;
  // 粗→細の各レベルで GN を回し、 transform を次レベルに引き継ぐ。
  for (std::size_t level = 0; level < level_tables.size(); ++level) {
    const gpu::VgicpVoxelTable & table = level_tables[level];
    if (table.capacity <= 0) {
      continue;
    }
    if (use_gpu) {
      engine_->setTarget(table);
    }

    gpu::VgicpLinearizeConfig lin_config;
    lin_config.huber_threshold = static_cast<float>(config_.huber_threshold);
    // 対応ゲートはレベルのボクセルサイズ比で広げる (粗レベルは mean が遠いため)。
    const double level_scale = table.voxel_size_m / base_voxel_size;
    lin_config.max_correspondence_distance_m = static_cast<float>(
      config_.max_correspondence_distance_m * std::max(1.0, level_scale));
    lin_config.search_radius_voxels = config_.search_radius_voxels;

    bool level_converged = false;
    for (int iteration = 0; iteration < config_.max_iterations; ++iteration) {
      const gpu::VgicpLinearization lin = use_gpu
        ? engine_->linearize(current, lin_config)
        : gpu::linearizeVgicpCpu(
            source_points, source_covariances, table, current, lin_config);

      if (use_gpu && !lin.gpu_used) {
        result.converged = false;
        result.transform_world_body = current;
        return result;
      }
      if (lin.valid_correspondences < 6) {
        break;
      }

      const Eigen::Matrix<double, 6, 1> delta = lin.hessian.ldlt().solve(lin.gradient);
      const Eigen::Vector3d delta_rotation = delta.head<3>();
      const Eigen::Vector3d delta_translation = delta.tail<3>();

      Eigen::Isometry3d delta_transform = Eigen::Isometry3d::Identity();
      delta_transform.linear() = lie::expSO3(delta_rotation);
      delta_transform.translation() = delta_translation;
      current = delta_transform * current;  // 左摂動
      current.linear() = lie::normalizeRotation(current.linear());

      result.iterations += 1;
      result.final_cost = lin.cost;
      result.num_correspondences = lin.valid_correspondences;

      if (delta_translation.norm() < config_.convergence_translation_m &&
          delta_rotation.norm() < config_.convergence_rotation_rad)
      {
        level_converged = true;
        break;
      }
    }
    // 最終 (最細) レベルの収束を全体の収束とする。
    if (level + 1 == level_tables.size()) {
      result.converged = level_converged;
    }
  }

  result.transform_world_body = current;
  return result;
}

std::string MetalVgicpRegistration::describe() const
{
  std::ostringstream oss;
  oss << "metal_vgicp:max_iter=" << config_.max_iterations
      << ",max_corr=" << config_.max_correspondence_distance_m
      << ",search_radius=" << config_.search_radius_voxels;
  return oss.str();
}

}  // namespace pylot_lio
