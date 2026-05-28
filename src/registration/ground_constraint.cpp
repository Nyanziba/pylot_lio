// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#include "pylot_lio/registration/ground_constraint.hpp"

#include <algorithm>
#include <cmath>

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <Eigen/LU>

namespace pylot_lio
{

namespace
{

// percentile-based 地面帯抽出: world z の下位 percentile を地面とみなし、 そこから
// band_m まで上の点を地面候補に含める。 単純な最小 z だと外れ値 (地下に飛んだ点) に
// 引っ張られるので 5% percentile を使う。
constexpr double kFloorPercentile = 0.05;
// 平面性チェック: 共分散の最小固有値が中固有値より十分小さくないと「平面」 とみなさない。
// floor の小さい固有値 / 中固有値の比が threshold 未満なら採用。
constexpr double kPlanarityRatioThreshold = 0.1;
// 必要最低限の地面候補点数。 これを下回ったら推定不能。
constexpr std::size_t kMinGroundPoints = 30;

}  // namespace

std::optional<Eigen::Vector3d> estimateGroundNormalBody(
  const std::vector<Eigen::Vector3d> & source_points_body,
  const Eigen::Isometry3d & pose_world_body,
  const GroundConstraintConfig & config)
{
  if (source_points_body.size() < kMinGroundPoints) {
    return std::nullopt;
  }

  // 全 source 点を world z に投影して下位 percentile を求める。
  std::vector<double> world_z_values;
  world_z_values.reserve(source_points_body.size());
  for (const auto & point_body : source_points_body) {
    const Eigen::Vector3d point_world = pose_world_body * point_body;
    world_z_values.push_back(point_world.z());
  }
  std::sort(world_z_values.begin(), world_z_values.end());
  const std::size_t percentile_index = static_cast<std::size_t>(
    kFloorPercentile * static_cast<double>(world_z_values.size()));
  const double floor_z = world_z_values[percentile_index];
  const double ceiling_z = floor_z + config.band_m;

  // 地面候補点 (world z が [floor_z, ceiling_z] 内) を body フレームで集めて共分散を取る。
  std::vector<Eigen::Vector3d> ground_points_body;
  ground_points_body.reserve(source_points_body.size() / 10);
  for (const auto & point_body : source_points_body) {
    const Eigen::Vector3d point_world = pose_world_body * point_body;
    if (point_world.z() >= floor_z && point_world.z() <= ceiling_z) {
      ground_points_body.push_back(point_body);
    }
  }
  if (ground_points_body.size() < kMinGroundPoints) {
    return std::nullopt;
  }

  // 共分散の最小固有ベクトル ≈ 法線 (body)。
  Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  for (const auto & point_body : ground_points_body) {
    mean += point_body;
  }
  mean /= static_cast<double>(ground_points_body.size());

  Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
  for (const auto & point_body : ground_points_body) {
    const Eigen::Vector3d centered = point_body - mean;
    covariance += centered * centered.transpose();
  }
  covariance /= static_cast<double>(ground_points_body.size());

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
  if (solver.info() != Eigen::Success) {
    return std::nullopt;
  }
  const Eigen::Vector3d eigenvalues = solver.eigenvalues();
  // 固有値は昇順。 最小 / 中間 比で平面性を判定。
  if (eigenvalues(1) <= 0.0) {
    return std::nullopt;
  }
  const double planarity_ratio = eigenvalues(0) / eigenvalues(1);
  if (planarity_ratio > kPlanarityRatioThreshold) {
    return std::nullopt;  // 平面性が弱い
  }

  Eigen::Vector3d normal_body = solver.eigenvectors().col(0).normalized();

  // 法線の向きを「上向き」 に揃える。 body 上で計算した法線を world に変換して
  // z 成分が負なら反転する (法線方向の符号曖昧性を解消)。
  const Eigen::Vector3d normal_world = pose_world_body.linear() * normal_body;
  if (normal_world.z() < 0.0) {
    normal_body = -normal_body;
  }

  // 傾きゲート: world に持って行った法線が (0,0,1) からどれだけ傾くか。
  const Eigen::Vector3d normal_world_oriented = pose_world_body.linear() * normal_body;
  const double cos_tilt = std::max(-1.0, std::min(1.0, normal_world_oriented.normalized().z()));
  const double tilt_deg = std::acos(cos_tilt) * 180.0 / M_PI;
  if (tilt_deg > config.max_tilt_deg) {
    return std::nullopt;
  }

  return normal_body;
}

void addGroundLevelingConstraint(
  Eigen::Matrix<double, 6, 6> & hessian,
  Eigen::Matrix<double, 6, 1> & gradient,
  const Eigen::Isometry3d & current_pose_world_body,
  const Eigen::Vector3d & ground_normal_body,
  double weight,
  double max_correction_rad)
{
  // ---- 数学導出 (left-world perturbation) ----
  // metal_vgicp の摂動は `current = exp(δ_world) * current` (world frame 左摂動)。
  // cost: E = 0.5 * weight * ||r||^2, r(δ) = up - exp(δ) * n_world
  //                                       ≈ (up - n_world) - δ × n_world
  // ヤコビアン: J = [n_world]_× (3x3 skew of n_world)
  // J^T J = ||n||^2 I - n n^T → unit n では (I - n n^T) ★ yaw 軸 (= n_world ≈ up_world) は自動的に null
  // J^T r_0 = [n]_×^T (up - n) = -n × (up - n) = -n × up + 0 = up × n_world
  // 通常の GN: H δ = -g where g = J^T r_0
  // 本 code は `delta = H.ldlt().solve(gradient)` で `gradient = -g` の convention なので、
  // hessian += weight * J^T J、 gradient += -weight * (up × n_world) を加算する。

  const Eigen::Vector3d n_world =
    (current_pose_world_body.linear() * ground_normal_body).normalized();
  const Eigen::Vector3d up_world(0.0, 0.0, 1.0);

  // 残差 axis (sin(tilt) * 回転軸)。 すでに水平なら何もしない。
  const Eigen::Vector3d residual_axis = up_world.cross(n_world);
  const double residual_magnitude = residual_axis.norm();
  if (residual_magnitude < 1e-9) {
    return;
  }

  // damping: 1 align あたりの leveling 補正上限 [rad] でスケール。 |residual| が上限を超えたら
  // 同方向に上限値までクランプ (ノイズ・誤検出での急スナップ防止)。
  // residual は |sin(tilt)| ≈ |tilt| なので max_correction_rad に直接比較できる。
  Eigen::Vector3d gradient_contribution = -weight * residual_axis;  // -g (符号反転)
  if (max_correction_rad > 0.0 && residual_magnitude > max_correction_rad) {
    gradient_contribution *= (max_correction_rad / residual_magnitude);
  }

  // hessian の rotation block (上 3x3、 world frame 左摂動の回転 δ_world) に J^T J を加算。
  // [delta_rotation; delta_translation] order (metal_vgicp_linearizer に合わせる)。
  const Eigen::Matrix3d hessian_contribution =
    weight * (Eigen::Matrix3d::Identity() - n_world * n_world.transpose());

  hessian.block<3, 3>(0, 0) += hessian_contribution;
  gradient.head<3>() += gradient_contribution;
}

}  // namespace pylot_lio
