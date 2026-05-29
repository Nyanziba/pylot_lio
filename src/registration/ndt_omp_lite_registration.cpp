// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#include "pylot_lio/registration/ndt_omp_lite_registration.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <vector>

#include <Eigen/Eigenvalues>

#include <pcl/filters/voxel_grid_covariance.h>

#include "pylot_lio/lie_algebra.hpp"
#include "pylot_lio/map/i_point_cloud_map.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace pylot_lio
{

namespace
{

// NDT の目的関数: loss = -exp(-1/2 * dᵀ Σ⁻¹ d) を最小化 (d = p_world - mu)。
// 厳密な Magnusson のスケーリング定数 (d1, d2; outlier_ratio から導出) は採用しない。
// 理由: d1, d2 は score の絶対値スケールを変えるだけで Newton 方向に影響しないため、
// 単純な「ガウシアン尤度の負数」 を最小化する形 (d1=d2=1) で同じ収束挙動になる。
// PCL の pcl::NDT は厳密な Magnusson 式を使うが、 こちらは比較目的の簡潔実装。

// 共分散 Σ の最小固有値を floor で持ち上げ、 逆行列を計算する。
// (Magnusson 5.3.6 と同じ退化対策)
Eigen::Matrix3d invertCovarianceWithFloor(
  const Eigen::Matrix3d & covariance,
  double eigenvalue_floor)
{
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
  if (solver.info() != Eigen::Success) {
    // 失敗時は scaled identity の逆行列で代替 (極力影響を抑える)。
    return Eigen::Matrix3d::Identity() / std::max(eigenvalue_floor, 1e-6);
  }
  Eigen::Vector3d eigenvalues = solver.eigenvalues();
  const double max_eigenvalue = eigenvalues(2);
  const double floor_value = std::max(eigenvalue_floor, max_eigenvalue * 1e-3);
  for (int eigen_index = 0; eigen_index < 3; ++eigen_index) {
    if (eigenvalues(eigen_index) < floor_value) {
      eigenvalues(eigen_index) = floor_value;
    }
  }
  const Eigen::Matrix3d eigenvectors = solver.eigenvectors();
  return eigenvectors *
    eigenvalues.cwiseInverse().asDiagonal() *
    eigenvectors.transpose();
}

// 反対称行列。 SO(3) の generator として使う。
inline Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d & vector)
{
  Eigen::Matrix3d skew;
  skew << 0.0, -vector.z(), vector.y(),
    vector.z(), 0.0, -vector.x(),
    -vector.y(), vector.x(), 0.0;
  return skew;
}

// 反復ごとの累積値。 OpenMP の per-thread accumulator として使う。
struct NewtonAccumulator
{
  Eigen::Matrix<double, 6, 6> hessian = Eigen::Matrix<double, 6, 6>::Zero();
  Eigen::Matrix<double, 6, 1> gradient = Eigen::Matrix<double, 6, 1>::Zero();
  double score = 0.0;
  int num_valid_points = 0;

  void addOther(const NewtonAccumulator & other)
  {
    hessian.noalias() += other.hessian;
    gradient.noalias() += other.gradient;
    score += other.score;
    num_valid_points += other.num_valid_points;
  }
};

// PCL 点群を非有限点を除いた std::vector<Eigen::Vector3d> に変換する。
std::vector<Eigen::Vector3d> toFinitePointVector(const PointCloud & cloud_in)
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
  return finite_points;
}

PointCloudPtr toPclCloudWithoutNonFinite(const PointCloud & cloud_in)
{
  PointCloudPtr finite_cloud(new PointCloud());
  finite_cloud->points.reserve(cloud_in.points.size());
  for (const Point & input_point : cloud_in.points) {
    if (!std::isfinite(input_point.x) ||
        !std::isfinite(input_point.y) ||
        !std::isfinite(input_point.z))
    {
      continue;
    }
    finite_cloud->points.push_back(input_point);
  }
  finite_cloud->width = static_cast<std::uint32_t>(finite_cloud->points.size());
  finite_cloud->height = 1;
  finite_cloud->is_dense = true;
  return finite_cloud;
}

// 1 点ぶんの NDT score / gradient / Hessian 寄与を accumulator に足す。
// p_world  : 変換後 source 点
// d        : p_world - mu (target ガウシアン中心からの残差)
// omega    : Σ⁻¹ (target ボクセル共分散の逆行列)
// jacobian : ∂p_world / ∂ξ  (3x6, ξ = (omega, t))
void accumulateNdtContribution(
  NewtonAccumulator & accumulator,
  const Eigen::Vector3d & d,
  const Eigen::Matrix3d & omega,
  const Eigen::Matrix<double, 3, 6> & jacobian)
{
  const Eigen::Vector3d omega_times_d = omega * d;
  const double mahalanobis = d.dot(omega_times_d);
  if (mahalanobis < 0.0 || !std::isfinite(mahalanobis)) {
    return;
  }
  // loss = -exp(-1/2 maha) を最小化。 d → 0 で loss → -1 (最小)、 d → ∞ で loss → 0。
  // ∂loss/∂maha = 1/2 exp(-1/2 maha)
  // ∂maha/∂ξ   = 2 (Σ⁻¹ d)ᵀ J
  // ∂loss/∂ξ   = exp(-1/2 maha) * (Σ⁻¹ d)ᵀ J  → 6x1 transpose: exp * Jᵀ Σ⁻¹ d
  const double exponential = std::exp(-0.5 * mahalanobis);
  if (exponential < 1e-300) {
    return;
  }

  // gradient += exp * Jᵀ (Σ⁻¹ d)
  accumulator.gradient.noalias() += exponential * jacobian.transpose() * omega_times_d;

  // Hessian の Gauss-Newton 近似: H ≈ exp * Jᵀ Σ⁻¹ J (positive-definite)。
  // 厳密 Hessian は (Σ⁻¹ d)(Σ⁻¹ d)ᵀ 項を含むが Newton の収束方向は同じで、
  // Gauss-Newton 近似の方が PSD で LDLT 安定。
  accumulator.hessian.noalias() +=
    exponential * jacobian.transpose() * omega * jacobian;

  accumulator.score += -exponential;
  accumulator.num_valid_points += 1;
}

}  // namespace

NdtOmpLiteRegistration::NdtOmpLiteRegistration(const Config & config)
: config_(config)
{
}

IRegistration::AlignResult NdtOmpLiteRegistration::align(
  const PointCloud & source_cloud_body,
  const IPointCloudMap & map_world,
  const Eigen::Isometry3d & initial_transform_world_body)
{
  AlignResult result;
  result.transform_world_body = initial_transform_world_body;

  const std::vector<Eigen::Vector3d> source_points_body =
    toFinitePointVector(source_cloud_body);
  const PointCloudPtr target_cloud = toPclCloudWithoutNonFinite(*map_world.toPointCloud());

  const std::size_t min_required =
    static_cast<std::size_t>(std::max(1, config_.min_points_for_alignment));
  if (source_points_body.size() < min_required || target_cloud->size() < min_required) {
    return result;
  }

  // target を VoxelGridCovariance でガウシアン化する。 1 回の align ごとに作り直す
  // (map が毎回変わるため target キャッシュは行わない。 公平比較が目的のため)。
  pcl::VoxelGridCovariance<Point> voxel_covariance;
  const float resolution_f = static_cast<float>(config_.resolution_m);
  voxel_covariance.setLeafSize(resolution_f, resolution_f, resolution_f);
  voxel_covariance.setMinPointPerVoxel(config_.min_points_per_voxel);
  voxel_covariance.setInputCloud(target_cloud);
  voxel_covariance.filter(true);  // true = ガウシアン (mean, cov, inverse_cov) も計算する

  if (voxel_covariance.getCentroids()->empty()) {
    // searchable な voxel が無い: NDT 不能。 初期推定を返す。
    return result;
  }

  Eigen::Isometry3d current_transform = initial_transform_world_body;
  const int total_iterations_budget = std::max(1, config_.max_iterations);

  // OpenMP のスレッド数を決める。 num_threads <= 0 なら OMP に任せる。
  const int requested_threads = config_.num_threads;
#ifdef _OPENMP
  const int active_threads = (requested_threads > 0)
    ? requested_threads
    : omp_get_max_threads();
#else
  const int active_threads = 1;
  (void)requested_threads;
#endif

  double previous_score = std::numeric_limits<double>::max();
  int iteration_count = 0;
  bool converged_flag = false;

  for (iteration_count = 0; iteration_count < total_iterations_budget; ++iteration_count) {
    const Eigen::Matrix3d rotation_world_body = current_transform.rotation();
    const Eigen::Vector3d translation_world_body = current_transform.translation();

    // per-thread accumulator: false sharing を避けるため別個に確保し、 最後に合算。
    std::vector<NewtonAccumulator> thread_accumulators(
      static_cast<std::size_t>(std::max(1, active_threads)));

    const int num_source_points = static_cast<int>(source_points_body.size());

#ifdef _OPENMP
    #pragma omp parallel num_threads(active_threads)
    {
      const int thread_index = omp_get_thread_num();
      NewtonAccumulator & local_accumulator = thread_accumulators[thread_index];

      #pragma omp for schedule(static)
      for (int point_index = 0; point_index < num_source_points; ++point_index) {
        const Eigen::Vector3d point_body = source_points_body[point_index];
        const Eigen::Vector3d point_world =
          rotation_world_body * point_body + translation_world_body;

        // PCL VoxelGridCovariance は「点に最も近い voxel ガウシアン」 を返す。
        // k=1 近傍 (Magnusson と同じ流儀)。 thread-safe (内部状態を持たない)。
        std::vector<pcl::VoxelGridCovariance<Point>::LeafConstPtr> matched_leaves;
        Point query_point;
        query_point.x = static_cast<float>(point_world.x());
        query_point.y = static_cast<float>(point_world.y());
        query_point.z = static_cast<float>(point_world.z());
        voxel_covariance.getNeighborhoodAtPoint(query_point, matched_leaves);
        if (matched_leaves.empty()) {
          continue;
        }

        for (const auto & leaf : matched_leaves) {
          if (leaf == nullptr || leaf->nr_points < config_.min_points_per_voxel) {
            continue;
          }
          const Eigen::Vector3d voxel_mean = leaf->getMean();
          const Eigen::Matrix3d voxel_covariance_matrix = leaf->getCov();
          const Eigen::Matrix3d voxel_inverse_covariance =
            invertCovarianceWithFloor(
              voxel_covariance_matrix, config_.covariance_eigenvalue_floor);

          const Eigen::Vector3d residual_d = point_world - voxel_mean;

          // J = [ -R skew(p_body) | I ] 3x6  (ξ = (omega, t), p_world = R p_body + t)
          // ∂(R p_body)/∂omega = -R skew(p_body) (right-perturbation 慣例)。
          Eigen::Matrix<double, 3, 6> jacobian;
          jacobian.block<3, 3>(0, 0) =
            -rotation_world_body * skewSymmetric(point_body);
          jacobian.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();

          accumulateNdtContribution(
            local_accumulator,
            residual_d, voxel_inverse_covariance, jacobian);
        }
      }
    }
#else
    // OpenMP 無効ビルド: 単スレッド逐次。
    NewtonAccumulator & local_accumulator = thread_accumulators[0];
    for (int point_index = 0; point_index < num_source_points; ++point_index) {
      const Eigen::Vector3d point_body = source_points_body[point_index];
      const Eigen::Vector3d point_world =
        rotation_world_body * point_body + translation_world_body;
      std::vector<pcl::VoxelGridCovariance<Point>::LeafConstPtr> matched_leaves;
      Point query_point;
      query_point.x = static_cast<float>(point_world.x());
      query_point.y = static_cast<float>(point_world.y());
      query_point.z = static_cast<float>(point_world.z());
      voxel_covariance.getNeighborhoodAtPoint(query_point, matched_leaves);
      if (matched_leaves.empty()) {
        continue;
      }
      for (const auto & leaf : matched_leaves) {
        if (leaf == nullptr || leaf->nr_points < config_.min_points_per_voxel) {
          continue;
        }
        const Eigen::Vector3d voxel_mean = leaf->getMean();
        const Eigen::Matrix3d voxel_inverse_covariance =
          invertCovarianceWithFloor(leaf->getCov(), config_.covariance_eigenvalue_floor);
        const Eigen::Vector3d residual_d = point_world - voxel_mean;
        Eigen::Matrix<double, 3, 6> jacobian;
        jacobian.block<3, 3>(0, 0) =
          -rotation_world_body * skewSymmetric(point_body);
        jacobian.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
        accumulateNdtContribution(
          local_accumulator,
          residual_d, voxel_inverse_covariance, jacobian);
      }
    }
#endif

    NewtonAccumulator total_accumulator;
    for (const auto & per_thread : thread_accumulators) {
      total_accumulator.addOther(per_thread);
    }

    if (total_accumulator.num_valid_points < config_.min_points_for_alignment) {
      // 有効寄与点が足りない: 続行しても改善できない。
      break;
    }

    // Newton step: H * delta = -g  (positive-definite を期待した solve)。
    // LDLT は対称半正定値で安定。 失敗時は LU フォールバック。
    Eigen::Matrix<double, 6, 1> delta_xi;
    Eigen::LDLT<Eigen::Matrix<double, 6, 6>> ldlt_solver(total_accumulator.hessian);
    if (ldlt_solver.info() == Eigen::Success) {
      delta_xi = ldlt_solver.solve(-total_accumulator.gradient);
    } else {
      delta_xi =
        total_accumulator.hessian.fullPivLu().solve(-total_accumulator.gradient);
    }
    if (!delta_xi.allFinite()) {
      break;
    }

    // step damping: 初期 step_size でスケール。
    Eigen::Matrix<double, 6, 1> applied_delta = config_.step_size * delta_xi;
    Eigen::Isometry3d updated_transform = current_transform;
    // ξ = (ω, t) の Jacobian は右摂動仕様 (∂(R p)/∂ω = -R skew(p)) なので、
    // 更新も R_new = R_old * exp(ω) の右側乗算で対応させる。 並進は world frame の
    // 単純加算で OK (p_world = R p_body + t)。
    updated_transform.translation() += applied_delta.tail<3>();
    updated_transform.linear() =
      current_transform.linear() * lie::expSO3(applied_delta.head<3>());

    // 収束判定 (適用後)。
    const double translation_step_norm = applied_delta.tail<3>().norm();
    const double rotation_step_norm = applied_delta.head<3>().norm();
    current_transform = updated_transform;
    previous_score = total_accumulator.score;

    if (translation_step_norm < config_.transformation_epsilon_m &&
        rotation_step_norm < config_.rotation_epsilon_rad)
    {
      converged_flag = true;
      ++iteration_count;
      break;
    }
  }

  result.transform_world_body = current_transform;
  result.iterations = iteration_count;
  result.converged = converged_flag;
  result.final_cost = previous_score;
  // NDT は対応点ベース手法ではないが、 「ガウシアンに尤度寄与した source 点数」を
  // 対応点数の代理として記録する。 最後のイテレーションの値ではなく、 ここでは
  // accumulator が break 後に消えているため source 点数を上限値として記録する。
  result.num_correspondences = static_cast<int>(source_points_body.size());
  return result;
}

std::string NdtOmpLiteRegistration::describe() const
{
  std::ostringstream summary_stream;
  summary_stream << "ndt_omp_lite"
                 << ":resolution=" << config_.resolution_m
                 << ",max_iterations=" << config_.max_iterations
                 << ",num_threads=" << config_.num_threads;
  return summary_stream.str();
}

}  // namespace pylot_lio
