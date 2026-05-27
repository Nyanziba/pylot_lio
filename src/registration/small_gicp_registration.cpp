// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#ifdef PYLOT_LIO_HAS_SMALL_GICP

#include "pylot_lio/registration/small_gicp_registration.hpp"

#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <small_gicp/ann/gaussian_voxelmap.hpp>
#include <small_gicp/ann/kdtree_omp.hpp>
#include <small_gicp/factors/gicp_factor.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/registration/reduction_omp.hpp>
#include <small_gicp/registration/registration.hpp>
#include <small_gicp/util/normal_estimation_omp.hpp>

// TBB バックエンドはビルドに TBB が含まれるときだけコンパイルする。
// (reduction_tbb.hpp 等はヘッダオンリーだが <tbb/tbb.h> を要求するため)
#ifdef PYLOT_LIO_HAS_TBB
#include <small_gicp/ann/kdtree_tbb.hpp>
#include <small_gicp/registration/reduction_tbb.hpp>
#include <small_gicp/util/normal_estimation_tbb.hpp>
#endif

namespace pylot_lio
{

// hpp が前方宣言する pimpl。 低レベル API 化で永続エンジンが不要になったため空。
// (unique_ptr<Impl> のデストラクタ実体化のためだけに残す)
struct SmallGicpRegistration::Impl
{
};

namespace
{

using SmallPointCloud = small_gicp::PointCloud;
using SmallKdTree = small_gicp::KdTree<small_gicp::PointCloud>;

// ----------------------------------------------------------------------------
// backend ごとに「KdTree ビルダ」「共分散推定関数」「reduction のスレッド設定」を
// 吸収する traits。 OpenMP と TBB で型名・関数名・スレッド指定方法が異なるため、
// テンプレートの本体 (alignWithBackend) を 1 つに保つための薄い変換層。
// ----------------------------------------------------------------------------
template <typename Reduction>
struct BackendTraits;

template <>
struct BackendTraits<small_gicp::ParallelReductionOMP>
{
  static SmallKdTree::Ptr buildTree(const SmallPointCloud::Ptr & cloud, int num_threads)
  {
    return std::make_shared<SmallKdTree>(cloud, small_gicp::KdTreeBuilderOMP(num_threads));
  }

  static void estimateCovariances(
    SmallPointCloud & cloud, SmallKdTree & tree, int num_neighbors, int num_threads)
  {
    small_gicp::estimate_covariances_omp(cloud, tree, num_neighbors, num_threads);
  }

  // OpenMP は reduction が自前の num_threads メンバを持つので指定する。
  static void setReductionThreads(small_gicp::ParallelReductionOMP & reduction, int num_threads)
  {
    reduction.num_threads = num_threads;
  }
};

#ifdef PYLOT_LIO_HAS_TBB
template <>
struct BackendTraits<small_gicp::ParallelReductionTBB>
{
  static SmallKdTree::Ptr buildTree(const SmallPointCloud::Ptr & cloud, int /*num_threads*/)
  {
    // TBB ビルダはグローバルなタスクスケジューラを使うのでスレッド数を取らない。
    return std::make_shared<SmallKdTree>(cloud, small_gicp::KdTreeBuilderTBB());
  }

  static void estimateCovariances(
    SmallPointCloud & cloud, SmallKdTree & tree, int num_neighbors, int /*num_threads*/)
  {
    small_gicp::estimate_covariances_tbb(cloud, tree, num_neighbors);
  }

  // TBB の reduction は num_threads メンバを持たない (グローバルスケジューラ任せ)。
  static void setReductionThreads(small_gicp::ParallelReductionTBB &, int) {}
};
#endif

// pylot_lio の PointCloud を small_gicp ネイティブ PointCloud に変換する。
// 非有限点 (NaN / inf) は除外する。 共分散は後段の estimate_covariances が埋める。
SmallPointCloud::Ptr toSmallCloud(const PointCloud & cloud_in)
{
  std::vector<Eigen::Vector3d> finite_points;
  finite_points.reserve(cloud_in.points.size());
  for (const Point & input_point : cloud_in.points) {
    if (!std::isfinite(input_point.x) ||
        !std::isfinite(input_point.y) ||
        !std::isfinite(input_point.z))
    {
      continue;
    }
    finite_points.emplace_back(input_point.x, input_point.y, input_point.z);
  }
  return std::make_shared<SmallPointCloud>(finite_points);
}

// 最小点数: small_gicp の align は |cloud| <= 10 で警告を出すため、 それ未満は
// 位置合わせを試みず初期推定を返す。
constexpr std::size_t kMinPointsForAlignment = 10;

// 指定の reduction (OMP/TBB) で 1 回の位置合わせを実行する本体。
// GICP: target を KdTree、 VGICP: target を GaussianVoxelMap として align する。
template <typename Reduction>
IRegistration::AlignResult alignWithBackend(
  const PointCloud & source_cloud_body,
  const IPointCloudMap & map_world,
  const Eigen::Isometry3d & initial_transform_world_body,
  const SmallGicpRegistration::Config & config)
{
  using Traits = BackendTraits<Reduction>;

  IRegistration::AlignResult result;
  result.transform_world_body = initial_transform_world_body;

  const SmallPointCloud::Ptr source = toSmallCloud(source_cloud_body);
  const SmallPointCloud::Ptr target = toSmallCloud(*map_world.toPointCloud());
  if (source->size() < kMinPointsForAlignment ||
      target->size() < kMinPointsForAlignment)
  {
    return result;
  }

  // source は GICP/VGICP の両方で共分散が必要。
  const SmallKdTree::Ptr source_tree = Traits::buildTree(source, config.num_threads);
  Traits::estimateCovariances(
    *source, *source_tree, config.covariance_num_neighbors, config.num_threads);

  // target も共分散を推定する (GICP は KdTree マッチング、 VGICP は voxelmap への
  // ガウス集約に使う)。
  const SmallKdTree::Ptr target_tree = Traits::buildTree(target, config.num_threads);
  Traits::estimateCovariances(
    *target, *target_tree, config.covariance_num_neighbors, config.num_threads);

  small_gicp::Registration<small_gicp::GICPFactor, Reduction> registration;
  Traits::setReductionThreads(registration.reduction, config.num_threads);
  registration.rejector.max_dist_sq =
    config.max_correspondence_distance_m * config.max_correspondence_distance_m;
  registration.optimizer.max_iterations = config.max_iterations;

  small_gicp::RegistrationResult registration_result;
  if (config.variant == SmallGicpRegistration::Variant::VGICP) {
    auto target_voxelmap =
      std::make_shared<small_gicp::GaussianVoxelMap>(config.map_voxel_resolution_m);
    target_voxelmap->insert(*target);
    registration_result = registration.align(
      *target_voxelmap, *source, *target_voxelmap, initial_transform_world_body);
  } else {
    registration_result =
      registration.align(*target, *source, *target_tree, initial_transform_world_body);
  }

  result.transform_world_body = registration_result.T_target_source;
  result.converged = registration_result.converged;
  result.iterations = static_cast<int>(registration_result.iterations);
  result.num_correspondences = static_cast<int>(registration_result.num_inliers);
  result.final_cost = registration_result.error;
  return result;
}

// config.parallel_backend を解決する。 "tbb" 指定でも TBB 非対応ビルドなら
// OpenMP にフォールバックし、 初回のみ警告する。
bool resolveUseTbb(const std::string & requested_backend)
{
  const bool wants_tbb = (requested_backend == "tbb");
#ifdef PYLOT_LIO_HAS_TBB
  return wants_tbb;
#else
  if (wants_tbb) {
    static bool warned_once = false;
    if (!warned_once) {
      std::cerr << "[SmallGicpRegistration] parallel_backend='tbb' was requested but this "
                   "build has no TBB support; falling back to OpenMP." << std::endl;
      warned_once = true;
    }
  }
  return false;
#endif
}

}  // namespace

SmallGicpRegistration::SmallGicpRegistration(const Config & config)
: impl_(std::make_unique<Impl>()),
  config_(config)
{
}

SmallGicpRegistration::~SmallGicpRegistration() = default;

IRegistration::AlignResult SmallGicpRegistration::align(
  const PointCloud & source_cloud_body,
  const IPointCloudMap & map_world,
  const Eigen::Isometry3d & initial_transform_world_body)
{
#ifdef PYLOT_LIO_HAS_TBB
  if (resolveUseTbb(config_.parallel_backend)) {
    return alignWithBackend<small_gicp::ParallelReductionTBB>(
      source_cloud_body, map_world, initial_transform_world_body, config_);
  }
#else
  (void)resolveUseTbb(config_.parallel_backend);  // 警告を出すためだけに呼ぶ
#endif
  return alignWithBackend<small_gicp::ParallelReductionOMP>(
    source_cloud_body, map_world, initial_transform_world_body, config_);
}

std::string SmallGicpRegistration::describe() const
{
  const bool use_tbb = resolveUseTbb(config_.parallel_backend);
  std::ostringstream oss;
  oss << (config_.variant == Variant::VGICP ? "small_gicp_vgicp" : "small_gicp_gicp")
      << ":voxel=" << config_.map_voxel_resolution_m
      << ",threads=" << config_.num_threads
      << ",backend=" << (use_tbb ? "tbb" : "omp");
  return oss.str();
}

}  // namespace pylot_lio

#endif  // PYLOT_LIO_HAS_SMALL_GICP
