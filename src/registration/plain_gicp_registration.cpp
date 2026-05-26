// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#include "pylot_lio/registration/plain_gicp_registration.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "pylot_lio/lie_algebra.hpp"

namespace pylot_lio
{

PlainGicpRegistration::PlainGicpRegistration(const Config & config)
: config_(config)
{
}

PlainGicpRegistration::AlignResult PlainGicpRegistration::align(
  const PointCloud & source_cloud_body,
  const IPointCloudMap & map_world,
  const Eigen::Isometry3d & initial_transform_world_body)
{
  AlignResult result;
  result.transform_world_body = initial_transform_world_body;

  Eigen::Isometry3d current_transform = initial_transform_world_body;
  const double max_correspondence_distance_squared =
    config_.max_correspondence_distance_m * config_.max_correspondence_distance_m;

  // OpenMP 並列度の決定。 0 以下なら omp_get_max_threads() に委ねる。
  // OpenMP が無効ビルドや num_threads=1 のときは並列領域を生成せず逐次経路を取る。
  int actual_num_threads = 1;
#ifdef _OPENMP
  if (config_.num_threads <= 0) {
    actual_num_threads = std::max(1, omp_get_max_threads());
  } else {
    actual_num_threads = config_.num_threads;
  }
#else
  (void)config_.num_threads;
#endif

  for (int iteration_index = 0; iteration_index < config_.max_iterations; ++iteration_index) {
    // 法線方程式 H * delta = b を組み立てる。delta は SE(3) 接ベクトル [omega; rho]。
    // 並列領域内で thread-id ごとにローカルに accumulate し、 後で合算する。
    // Matrix6d (288B) ・ Vector6d (48B) は単独でキャッシュライン (64B) を跨ぐサイズなので
    // false sharing は事実上発生しない。 std::vector が要素間に padding を入れないこと
    // は気にしなくて良い。
    std::vector<Eigen::Matrix<double, 6, 6>> per_thread_hessian(
      actual_num_threads, Eigen::Matrix<double, 6, 6>::Zero());
    std::vector<Eigen::Matrix<double, 6, 1>> per_thread_gradient(
      actual_num_threads, Eigen::Matrix<double, 6, 1>::Zero());
    std::vector<double> per_thread_cost(actual_num_threads, 0.0);
    std::vector<int> per_thread_count(actual_num_threads, 0);

    const int num_source_points = static_cast<int>(source_cloud_body.points.size());

    // actual_num_threads == 1 のときは並列領域に入らず逐次経路を取る (デバッグ容易性 +
    // OpenMP runtime overhead 削減)。 _OPENMP 未定義時もこの分岐で逐次パス。
#ifdef _OPENMP
    #pragma omp parallel num_threads(actual_num_threads) if(actual_num_threads > 1)
#endif
    {
      int thread_id = 0;
#ifdef _OPENMP
      thread_id = omp_get_thread_num();
#endif
      Eigen::Matrix<double, 6, 6> & local_hessian = per_thread_hessian[thread_id];
      Eigen::Matrix<double, 6, 1> & local_gradient = per_thread_gradient[thread_id];
      double & local_cost = per_thread_cost[thread_id];
      int & local_count = per_thread_count[thread_id];

#ifdef _OPENMP
      #pragma omp for schedule(static) nowait
#endif
      for (int point_index = 0; point_index < num_source_points; ++point_index) {
        const Point & source_point_body = source_cloud_body.points[point_index];
        if (!std::isfinite(source_point_body.x) ||
            !std::isfinite(source_point_body.y) ||
            !std::isfinite(source_point_body.z))
        {
          continue;
        }
        const Eigen::Vector3d source_in_body(
          source_point_body.x, source_point_body.y, source_point_body.z);
        const Eigen::Vector3d source_in_world = current_transform * source_in_body;

        const PointCorrespondence correspondence =
          map_world.findNearestNeighbor(source_in_world);
        if (!correspondence.valid ||
            correspondence.squared_distance > max_correspondence_distance_squared)
        {
          continue;
        }

        // 残差: マップ平均 - 変換後 source。
        const Eigen::Vector3d residual_world =
          correspondence.target_point_world - source_in_world;

        // Mahalanobis 重み: target_covariance の逆。
        Eigen::Matrix3d information_matrix;
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(correspondence.target_covariance);
        Eigen::Vector3d eigenvalues = solver.eigenvalues();
        for (int axis = 0; axis < 3; ++axis) {
          eigenvalues(axis) = 1.0 / std::max(eigenvalues(axis), 1e-9);
        }
        information_matrix = solver.eigenvectors() * eigenvalues.asDiagonal()
          * solver.eigenvectors().transpose();

        // Huber 重み (ロバスト化)。
        const double mahalanobis_squared =
          residual_world.transpose() * information_matrix * residual_world;
        double robust_weight = 1.0;
        if (mahalanobis_squared > config_.huber_threshold * config_.huber_threshold) {
          robust_weight = config_.huber_threshold / std::sqrt(mahalanobis_squared);
        }

        // SE(3) 左摂動のヤコビアン (シリアル版と同じ式)。
        Eigen::Matrix<double, 3, 6> jacobian_matrix;
        jacobian_matrix.block<3, 3>(0, 0) = lie::skew(source_in_world);
        jacobian_matrix.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();

        local_hessian.noalias() +=
          robust_weight * jacobian_matrix.transpose() * information_matrix * jacobian_matrix;
        local_gradient.noalias() -=
          robust_weight * jacobian_matrix.transpose() * information_matrix * residual_world;
        local_cost += robust_weight * mahalanobis_squared;
        local_count += 1;
      }
    }

    // スレッドローカル累算を合算 (logical reduction)。
    Eigen::Matrix<double, 6, 6> hessian_matrix = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> gradient_vector = Eigen::Matrix<double, 6, 1>::Zero();
    double accumulated_cost = 0.0;
    int valid_correspondences = 0;
    for (int thread_index = 0; thread_index < actual_num_threads; ++thread_index) {
      hessian_matrix += per_thread_hessian[thread_index];
      gradient_vector += per_thread_gradient[thread_index];
      accumulated_cost += per_thread_cost[thread_index];
      valid_correspondences += per_thread_count[thread_index];
    }

    if (valid_correspondences < 6) {
      // 解が一意に決まらない。終了。
      break;
    }

    // 縮退正則化 (Tuna 2024, X-ICP). sycl_points::DegenerateRegularization と同等の処方を、
    // 本実装の「回転は左乗算 / 並進は加算」decoupled 規約に合わせて自前で書き起こした。
    // Apache-2.0, https://arxiv.org/abs/2408.11809
    if (config_.enable_degenerate_regularization) {
      // delta_twist: 「現在姿勢が initial_guess からどれだけ離れたか」
      const Eigen::Matrix3d delta_rotation_matrix =
        current_transform.linear() * initial_transform_world_body.linear().transpose();
      Eigen::Matrix<double, 6, 1> delta_twist;
      delta_twist.head<3>() = lie::logSO3(delta_rotation_matrix);
      delta_twist.tail<3>() =
        current_transform.translation() - initial_transform_world_body.translation();

      const double inlier_count = static_cast<double>(valid_correspondences);
      const double lambda = config_.regularization_base_factor * inlier_count;

      Eigen::Matrix<double, 6, 6> hessian_penalty =
        Eigen::Matrix<double, 6, 6>::Zero();

      // 回転 3x3 ブロックの固有値分解。固有値が小さい方向 = 「点群幾何だけでは
      // 拘束できていない回転自由度」なので、その方向だけ initial_guess に引き戻す。
      if (config_.rotation_eigenvalue_threshold > 0.0) {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver_rotation(
          hessian_matrix.block<3, 3>(0, 0));
        if (solver_rotation.info() == Eigen::Success) {
          for (int axis_index = 0; axis_index < 3; ++axis_index) {
            const double normalized_eigenvalue =
              solver_rotation.eigenvalues()(axis_index) / inlier_count;
            if (normalized_eigenvalue < config_.rotation_eigenvalue_threshold) {
              Eigen::Matrix<double, 6, 1> degenerate_direction =
                Eigen::Matrix<double, 6, 1>::Zero();
              degenerate_direction.head<3>() =
                solver_rotation.eigenvectors().col(axis_index);
              hessian_penalty +=
                lambda * degenerate_direction * degenerate_direction.transpose();
            }
          }
        }
      }
      // 並進 3x3 ブロックも同様 (廊下方向など並進が拘束されない場面で効く)。
      if (config_.translation_eigenvalue_threshold > 0.0) {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver_translation(
          hessian_matrix.block<3, 3>(3, 3));
        if (solver_translation.info() == Eigen::Success) {
          for (int axis_index = 0; axis_index < 3; ++axis_index) {
            const double normalized_eigenvalue =
              solver_translation.eigenvalues()(axis_index) / inlier_count;
            if (normalized_eigenvalue < config_.translation_eigenvalue_threshold) {
              Eigen::Matrix<double, 6, 1> degenerate_direction =
                Eigen::Matrix<double, 6, 1>::Zero();
              degenerate_direction.tail<3>() =
                solver_translation.eigenvectors().col(axis_index);
              hessian_penalty +=
                lambda * degenerate_direction * degenerate_direction.transpose();
            }
          }
        }
      }
      hessian_matrix += hessian_penalty;
      // 本実装の gradient_vector は -J^T W r (= 負勾配) 規約のため、
      // sycl_points の `b += H_pen * delta_twist` (solve(H, -b) 規約) と等価な形は
      // gradient_vector -= H_pen * delta_twist となる。
      gradient_vector -= hessian_penalty * delta_twist;
    }

    // 解く: hessian * delta = gradient ではなく、
    // 上の組み立てで b = -J^T W r, H = J^T W J なので delta = H^{-1} b ではなく、
    // r ~ -J * delta から得る最小化方向 delta = -H^{-1} (J^T W r)
    // → ここでは delta = -H^{-1} gradient_vector ではなく
    //   ↑ gradient_vector を符号反転して詰めているので delta = H^{-1} * gradient_vector
    // (符号は上の noalias 部分で揃えてある)
    const Eigen::Matrix<double, 6, 1> delta_vector =
      hessian_matrix.ldlt().solve(gradient_vector);

    const Eigen::Vector3d delta_rotation = delta_vector.head<3>();
    const Eigen::Vector3d delta_translation = delta_vector.tail<3>();

    Eigen::Isometry3d delta_transform = Eigen::Isometry3d::Identity();
    delta_transform.linear() = lie::expSO3(delta_rotation);
    delta_transform.translation() = delta_translation;
    current_transform = delta_transform * current_transform;
    current_transform.linear() = lie::normalizeRotation(current_transform.linear());

    result.iterations = iteration_index + 1;
    result.final_cost = accumulated_cost;
    result.num_correspondences = valid_correspondences;

    if (delta_translation.norm() < config_.convergence_translation_m &&
        delta_rotation.norm() < config_.convergence_rotation_rad)
    {
      result.converged = true;
      break;
    }
  }

  result.transform_world_body = current_transform;
  return result;
}

std::string PlainGicpRegistration::describe() const
{
  std::ostringstream oss;
  oss << "plain_gicp:max_iter=" << config_.max_iterations
      << ",max_corr=" << config_.max_correspondence_distance_m;
  return oss.str();
}

}  // namespace pylot_lio
