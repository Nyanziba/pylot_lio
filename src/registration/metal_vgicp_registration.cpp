// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#include "pylot_lio/registration/metal_vgicp_registration.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <optional>
#include <sstream>
#include <tuple>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

#include "pylot_lio/gpu/metal_vgicp_linearizer.hpp"
#include "pylot_lio/lie_algebra.hpp"
#include "pylot_lio/map/gaussian_voxel.hpp"
#include "pylot_lio/map/normal_map.hpp"
#include "pylot_lio/map/voxel_map.hpp"
#include "pylot_lio/registration/ground_constraint.hpp"
#include "pylot_lio/registration/source_covariance.hpp"

namespace pylot_lio
{

namespace
{

// 指定解像度 level_voxel_size でガウスボクセル群を再ビン (平行軸定理でマージ) し、
// VgicpVoxelTable を作る。 各ガウスの 2 次モーメント (C_i + μ_i μ_iᵀ) を count 重みで
// 合算し、 粗ボクセルの分布 (μ, C) を得る。 level_voxel_size = base のときは各ガウスが
// 自分のボクセルに一意に落ちるので実質そのまま (= 最細レベル)。
// base_voxels はマップ非依存の中間表現 (voxel_map/normal_map の保存済みガウス、 もしくは
// point ベースマップを voxelizePointsToGaussians した結果)。
gpu::VgicpVoxelTable buildLevelTable(
  const std::vector<GaussianVoxel> & base_voxels,
  double level_voxel_size,
  double eigen_floor)
{
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

  for (const GaussianVoxel & voxel : base_voxels) {
    const int cx = static_cast<int>(std::floor(voxel.mean.x() * inv_level));
    const int cy = static_cast<int>(std::floor(voxel.mean.y() * inv_level));
    const int cz = static_cast<int>(std::floor(voxel.mean.z() * inv_level));
    Accumulator & acc = bins[std::make_tuple(cx, cy, cz)];
    acc.coord = Eigen::Vector3i(cx, cy, cz);
    const double n = voxel.count > 0.0 ? voxel.count : 1.0;
    acc.total_count += n;
    acc.sum_weighted_mean += n * voxel.mean;
    acc.sum_weighted_2nd_moment +=
      n * (voxel.covariance + voxel.mean * voxel.mean.transpose());
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
  const std::vector<GaussianVoxel> & base_voxels,
  double base_voxel_size,
  int levels,
  double scaling_factor,
  double eigen_floor)
{
  const int clamped_levels = std::max(1, levels);
  std::vector<gpu::VgicpVoxelTable> tables;
  tables.reserve(clamped_levels);
  for (int level = clamped_levels - 1; level >= 0; --level) {
    const double level_voxel_size =
      base_voxel_size * std::pow(scaling_factor, static_cast<double>(level));
    tables.push_back(buildLevelTable(base_voxels, level_voxel_size, eigen_floor));
  }
  return tables;
}

// マップから VGICP の target ガウス分布をマップ非依存の中間表現で取り出す。
//   - voxel_map / normal_map : 保存済みガウス分布を直接使用 (toPointCloud() は
//     1 ボクセル 1 点で共分散が復元できないため)。
//   - それ以外 (voxel_random_map / voxel_keyframe_submap / kd_tree_map 等) :
//     toPointCloud() の生点群 (ボクセルあたり複数点) をボクセル化してガウスを復元。
// base_voxel_size と eigen_floor も合わせて返す (多重解像度の基準と退化防止に使う)。
struct BaseGaussians
{
  std::vector<GaussianVoxel> voxels;
  double base_voxel_size = 0.5;
  double eigen_floor = 1e-3;
};

BaseGaussians extractBaseGaussians(
  const IPointCloudMap & map_world,
  double fallback_voxel_size,
  double fallback_eigen_floor,
  int fallback_min_points)
{
  BaseGaussians result;
  if (const VoxelMap * voxel_map = dynamic_cast<const VoxelMap *>(&map_world)) {
    result.voxels = voxel_map->toGaussianVoxels();
    result.base_voxel_size = voxel_map->config().voxel_size_m;
    result.eigen_floor = voxel_map->config().covariance_eigen_floor;
    return result;
  }
  if (const NormalMap * normal_map = dynamic_cast<const NormalMap *>(&map_world)) {
    result.voxels = normal_map->toGaussianVoxels();
    result.base_voxel_size = fallback_voxel_size;
    result.eigen_floor = fallback_eigen_floor;
    return result;
  }
  // point ベースマップ: toPointCloud() をボクセル化してガウスを作る。
  const PointCloudPtr cloud = map_world.toPointCloud();
  if (cloud) {
    result.voxels = voxelizePointsToGaussians(
      *cloud, fallback_voxel_size, fallback_eigen_floor, fallback_min_points);
  }
  result.base_voxel_size = fallback_voxel_size;
  result.eigen_floor = fallback_eigen_floor;
  return result;
}

}  // namespace

MetalVgicpRegistration::MetalVgicpRegistration(const Config & config)
: config_(config),
  engine_(std::make_shared<gpu::MetalVgicpEngine>()),
  covariance_engine_(
    config.use_gpu_source_covariance
      ? std::make_shared<gpu::MetalCovarianceEngine>()
      : nullptr)
{
  // engine_ / covariance_engine_ の ctor で device/PSO を 1 回構築する
  // (Metal 無効ビルドでは isValid()=false)。
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

  // source 点を Vector3d 配列に変換 (NaN 点も含めて並びを保つ。 linearizer 側で除外)。
  std::vector<Eigen::Vector3d> source_points;
  source_points.reserve(source_cloud_body.points.size());
  for (const Point & p : source_cloud_body.points) {
    source_points.emplace_back(p.x, p.y, p.z);
  }
  // source 共分散 (前処理)。 use_gpu_source_covariance なら GPU (Metal) でグリッド
  // kNN + 平面正則化を行う。 GPU 無効ビルド / デバイス無では engine 側が CPU 参照
  // (グリッド kNN) に自動フォールバックする。 false なら従来の PCL KdTree を使う。
  std::vector<Eigen::Matrix3d> source_covariances;
  if (config_.use_gpu_source_covariance && covariance_engine_) {
    gpu::CovarianceEstimateConfig cov_config;
    cov_config.num_neighbors = config_.source_covariance_num_neighbors;
    cov_config.plane_epsilon =
      static_cast<float>(config_.source_covariance_plane_epsilon);
    cov_config.cell_size_m =
      static_cast<float>(config_.source_covariance_cell_size_m);
    cov_config.search_radius_cells = 1;
    source_covariances = covariance_engine_->estimate(source_points, cov_config).covariances;
  } else {
    source_covariances = computeSourceCovariances(
      source_cloud_body, config_.source_covariance_num_neighbors,
      config_.source_covariance_plane_epsilon);
  }

  // engine 無効 (Metal 無しビルド / デバイス無) なら未収束で返す。 これは factory が
  // plain_gicp へフォールバックする領域 (metal_vgicp はそもそも使われない想定)。
  if (!engine_ || !engine_->isValid()) {
    result.converged = false;
    return result;
  }

  // target ガウス分布をマップ非依存に取り出す (voxel_map/normal_map は保存済みガウス、
  // それ以外は toPointCloud() をボクセル化)。 ガウスが空なら未収束で返す。
  const BaseGaussians base = extractBaseGaussians(
    map_world, config_.target_voxel_size_m, config_.covariance_eigen_floor,
    config_.min_points_per_voxel);
  if (base.voxels.empty()) {
    result.converged = false;
    return result;
  }

  // 多重解像度テーブルを「粗→細」で構築 (levels=1 なら単一解像度)。
  const std::vector<gpu::VgicpVoxelTable> level_tables =
    buildMultiResolutionTables(
      base.voxels, base.base_voxel_size, config_.voxelmap_levels,
      config_.voxelmap_scaling_factor, base.eigen_floor);
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

  const double base_voxel_size = base.base_voxel_size;

  // 地面 leveling 拘束用の地面法線 (body) を初期姿勢で 1 回だけ推定する。
  // 平面性・傾きゲートを満たさなければ nullopt (= この scan は拘束を使わない)。
  // 振動ゲート: 前フレームの法線 (body) から大きく変化したら拘束をスキップ。
  std::optional<Eigen::Vector3d> ground_normal_body;
  if (config_.enable_ground_constraint) {
    GroundConstraintConfig ground_config;
    ground_config.band_m = config_.ground_band_m;
    ground_config.max_tilt_deg = config_.ground_max_tilt_deg;
    const std::optional<Eigen::Vector3d> fresh_normal = estimateGroundNormalBody(
      source_points, initial_transform_world_body, ground_config);
    ground_normal_body = fresh_normal;

    if (ground_normal_body && previous_ground_normal_body_ &&
        config_.ground_vibration_threshold_deg > 0.0)
    {
      const double cos_angle = std::max(-1.0, std::min(1.0,
        ground_normal_body->normalized().dot(
          previous_ground_normal_body_->normalized())));
      const double change_deg = std::acos(cos_angle) * 180.0 / M_PI;
      if (change_deg > config_.ground_vibration_threshold_deg) {
        ground_normal_body.reset();  // 振動中 → 今回は leveling 拘束を使わない
      }
    }

    if (fresh_normal) {
      previous_ground_normal_body_ = fresh_normal;
    }
  }

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

      // 地面 leveling 拘束を VGICP の正規方程式に加算してから solve する。
      // 重みは VGICP の「回転剛性」 (回転ヘッシアン対角の平均) を基準にスケールする。
      // VGICP の情報行列 (~1/eigen_floor) × 点位置² により回転ヘッシアンは桁違いに
      // 大きくなるため、 点数で割った固定重みでは拘束が埋もれて効かない。 回転剛性比に
      // することで weight=1.0 が「幾何項の回転剛性と同程度」 となり、 シーン規模に依らず
      // 一定の効き方になる (weight を上げるほど地面水平が優先される)。
      Eigen::Matrix<double, 6, 6> hessian = lin.hessian;
      Eigen::Matrix<double, 6, 1> gradient = lin.gradient;
      if (ground_normal_body) {
        const double rotation_stiffness =
          (hessian(0, 0) + hessian(1, 1) + hessian(2, 2)) / 3.0;
        const double ground_weight =
          config_.ground_constraint_weight * std::max(rotation_stiffness, 1.0);
        const double max_correction_rad =
          config_.ground_max_correction_per_frame_deg * M_PI / 180.0;
        addGroundLevelingConstraint(
          hessian, gradient, current, *ground_normal_body, ground_weight,
          max_correction_rad);
      }
      const Eigen::Matrix<double, 6, 1> delta = hessian.ldlt().solve(gradient);
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

  // ジャンプ診断: 予測姿勢 (constant-velocity の initial guess) からの 1 フレーム補正量が
  // 異常に大きいフレームを stderr に出す。 通常は予測との差は小さいので、 大きな補正 =
  // ジャンプ。 回転補正が大きければ地面拘束の急スナップ、 並進補正が大きければ VGICP の
  // 発散・誤対応が疑われる。 地面拘束が効いていたか・地面の傾きも併記する。
  {
    const Eigen::Matrix3d rotation_correction =
      current.linear() * initial_transform_world_body.linear().transpose();
    const double rotation_correction_rad =
      std::abs(Eigen::AngleAxisd(rotation_correction).angle());
    const double translation_correction_m =
      (current.translation() - initial_transform_world_body.translation()).norm();
    // しきい値: 旋回中の通常補正 (数度〜十数度) ではなく、 本当の異常 (発散・誤対応・
    // テレポート) だけを拾う。 並進が主シグナル。 デスキュー導入後はこれを超える発散は
    // ほぼ出ないはず。 診断ノイズを抑えるため回転は緩めに。
    // 大きな (= 本当に異常) Jump だけ拾う閾値。 旋回中・加減速の通常補正 (~1-1.5m / ~10-20°) は
    // 正常動作なので拾わない。 ここを超える補正は誤マッチ/発散の可能性が高い「大ジャンプ」。
    constexpr double kJumpTranslationThresholdM = 0.5;
    constexpr double kJumpRotationThresholdRad = 0.7;  // ~40 deg
    if (translation_correction_m > kJumpTranslationThresholdM ||
        rotation_correction_rad > kJumpRotationThresholdRad)
    {
      double ground_tilt_deg = -1.0;
      if (ground_normal_body) {
        const Eigen::Vector3d v =
          initial_transform_world_body.linear() * *ground_normal_body;
        ground_tilt_deg =
          std::acos(std::max(-1.0, std::min(1.0, v.normalized().z()))) * 180.0 / M_PI;
      }
      std::fprintf(
        stderr,
        "[metal_vgicp][JUMP] trans_corr=%.2fm rot_corr=%.1fdeg ground=%s "
        "ground_tilt=%.1fdeg converged=%d corr=%d cost=%.3f\n",
        translation_correction_m, rotation_correction_rad * 180.0 / M_PI,
        ground_normal_body ? "yes" : "no", ground_tilt_deg,
        result.converged ? 1 : 0, result.num_correspondences, result.final_cost);
    }

    // CV 予測診断: 通常時の baseline を見るため、 JUMP 閾値未満でも 10 align ごとに
    // 同じメトリクスを出す。 これにより JUMP のみだと欠落する「正常時の trans_corr/rot_corr」
    // 分布が観測でき、 CV (等速度) 予測がどれくらい真の運動から外れているかが分かる。
    constexpr std::uint64_t kCvLogIntervalAligns = 10;
    if ((align_call_count_ % kCvLogIntervalAligns) == 0) {
      std::fprintf(
        stderr,
        "[metal_vgicp][CV] trans_corr=%.3fm rot_corr=%.2fdeg converged=%d "
        "iter=%d corr=%d cost=%.3f\n",
        translation_correction_m, rotation_correction_rad * 180.0 / M_PI,
        result.converged ? 1 : 0, result.iterations,
        result.num_correspondences, result.final_cost);
    }
    ++align_call_count_;
  }

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
