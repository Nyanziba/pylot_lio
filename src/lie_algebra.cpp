// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#include "pylot_lio/lie_algebra.hpp"

#include <cmath>

namespace pylot_lio::lie
{

namespace
{
// 小角度判定の閾値。これ以下では Taylor 展開を使い、sin/cos の分母ゼロを避ける。
constexpr double kSmallAngleThreshold = 1e-8;
}  // namespace

Eigen::Matrix3d skew(const Eigen::Vector3d & vector)
{
  Eigen::Matrix3d skew_matrix;
  skew_matrix <<
        0.0,         -vector.z(),  vector.y(),
        vector.z(),   0.0,        -vector.x(),
       -vector.y(),   vector.x(),  0.0;
  return skew_matrix;
}

Eigen::Matrix3d expSO3(const Eigen::Vector3d & rotation_vector)
{
  const double angle = rotation_vector.norm();
  if (angle < kSmallAngleThreshold) {
    // Taylor 展開: exp(phi^) ~ I + phi^ + 0.5 * phi^^2
    const Eigen::Matrix3d skew_matrix = skew(rotation_vector);
    return Eigen::Matrix3d::Identity() + skew_matrix
      + 0.5 * skew_matrix * skew_matrix;
  }
  const Eigen::Vector3d unit_axis = rotation_vector / angle;
  const Eigen::Matrix3d skew_axis = skew(unit_axis);
  // Rodrigues 公式: R = I + sin(theta) * K + (1 - cos(theta)) * K^2
  return Eigen::Matrix3d::Identity()
    + std::sin(angle) * skew_axis
    + (1.0 - std::cos(angle)) * skew_axis * skew_axis;
}

Eigen::Vector3d logSO3(const Eigen::Matrix3d & rotation_matrix)
{
  const double cos_angle = std::min(
    1.0, std::max(-1.0, 0.5 * (rotation_matrix.trace() - 1.0)));
  const double angle = std::acos(cos_angle);
  if (angle < kSmallAngleThreshold) {
    // 小角度: R - R^T ~ 2 * skew(phi)
    return 0.5 * Eigen::Vector3d(
      rotation_matrix(2, 1) - rotation_matrix(1, 2),
      rotation_matrix(0, 2) - rotation_matrix(2, 0),
      rotation_matrix(1, 0) - rotation_matrix(0, 1));
  }
  const double scale = angle / (2.0 * std::sin(angle));
  return scale * Eigen::Vector3d(
    rotation_matrix(2, 1) - rotation_matrix(1, 2),
    rotation_matrix(0, 2) - rotation_matrix(2, 0),
    rotation_matrix(1, 0) - rotation_matrix(0, 1));
}

Eigen::Matrix3d rightJacobianSO3(const Eigen::Vector3d & rotation_vector)
{
  const double angle = rotation_vector.norm();
  const Eigen::Matrix3d skew_matrix = skew(rotation_vector);
  if (angle < kSmallAngleThreshold) {
    return Eigen::Matrix3d::Identity() - 0.5 * skew_matrix;
  }
  const double angle_squared = angle * angle;
  const double angle_cubed = angle_squared * angle;
  return Eigen::Matrix3d::Identity()
    - ((1.0 - std::cos(angle)) / angle_squared) * skew_matrix
    + ((angle - std::sin(angle)) / angle_cubed) * skew_matrix * skew_matrix;
}

Eigen::Matrix3d rightJacobianInverseSO3(const Eigen::Vector3d & rotation_vector)
{
  const double angle = rotation_vector.norm();
  const Eigen::Matrix3d skew_matrix = skew(rotation_vector);
  if (angle < kSmallAngleThreshold) {
    return Eigen::Matrix3d::Identity() + 0.5 * skew_matrix;
  }
  const double angle_squared = angle * angle;
  const double cot_half = 1.0 / std::tan(0.5 * angle);
  return Eigen::Matrix3d::Identity()
    + 0.5 * skew_matrix
    + (1.0 / angle_squared - 0.5 * cot_half / angle) * skew_matrix * skew_matrix;
}

Eigen::Matrix3d normalizeRotation(const Eigen::Matrix3d & nearly_rotation)
{
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(
    nearly_rotation, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d rotation = svd.matrixU() * svd.matrixV().transpose();
  if (rotation.determinant() < 0.0) {
    Eigen::Matrix3d sign_correction = Eigen::Matrix3d::Identity();
    sign_correction(2, 2) = -1.0;
    rotation = svd.matrixU() * sign_correction * svd.matrixV().transpose();
  }
  return rotation;
}

Eigen::Matrix<double, 6, 1> logSE3(const Eigen::Isometry3d & transform)
{
  const Eigen::Vector3d omega = logSO3(transform.linear());
  const Eigen::Vector3d translation = transform.translation();
  const double theta = omega.norm();

  Eigen::Matrix3d v_inverse;
  if (theta < kSmallAngleThreshold) {
    v_inverse = Eigen::Matrix3d::Identity() - 0.5 * skew(omega);
  } else {
    const Eigen::Matrix3d skew_omega = skew(omega);
    const double half_theta = 0.5 * theta;
    // coeff = (1 - theta * cot(half_theta) / 2) / theta^2
    const double coeff =
      (1.0 - theta * std::cos(half_theta) / (2.0 * std::sin(half_theta)))
      / (theta * theta);
    v_inverse = Eigen::Matrix3d::Identity()
      - 0.5 * skew_omega
      + coeff * skew_omega * skew_omega;
  }

  Eigen::Matrix<double, 6, 1> twist;
  twist.head<3>() = omega;
  twist.tail<3>() = v_inverse * translation;
  return twist;
}

}  // namespace pylot_lio::lie
