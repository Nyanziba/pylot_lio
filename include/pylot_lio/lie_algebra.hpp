// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
//
// SO(3) / SE(3) リー代数の小道具。標準的な定式 (Sola 2017 "Quaternion kinematics
// for the error-state KF", Barfoot "State Estimation for Robotics") に基づく自前実装。
// 外部実装の翻訳ではなく、教科書の数式を直接コード化したもの。

#ifndef PYLOT_LIO__LIE_ALGEBRA_HPP_
#define PYLOT_LIO__LIE_ALGEBRA_HPP_

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace pylot_lio::lie
{

// 3次元ベクトルから 3x3 歪対称行列 (skew-symmetric matrix) を作る。
// クロス積 a x b = skew(a) * b として使われる基本道具。
Eigen::Matrix3d skew(const Eigen::Vector3d & vector);

// SO(3) の指数写像: 回転ベクトル (axis * angle) を回転行列に変換。
// |phi| が極めて小さい場合は Taylor 展開でゼロ除算を回避する。
Eigen::Matrix3d expSO3(const Eigen::Vector3d & rotation_vector);

// SO(3) の対数写像: 回転行列を回転ベクトルに変換。
Eigen::Vector3d logSO3(const Eigen::Matrix3d & rotation_matrix);

// SO(3) の右ヤコビアン J_r(phi)。ESKF の誤差状態更新で使う。
// 数値安定性のため小角度時の Taylor 展開を含む。
Eigen::Matrix3d rightJacobianSO3(const Eigen::Vector3d & rotation_vector);

// 右ヤコビアンの逆行列。状態を誤差で更新するときの対称形に使われる。
Eigen::Matrix3d rightJacobianInverseSO3(const Eigen::Vector3d & rotation_vector);

// 回転行列が SO(3) を満たすように直交化する (数値ドリフト対策)。
Eigen::Matrix3d normalizeRotation(const Eigen::Matrix3d & nearly_rotation);

// SE(3) の対数写像。T = exp([omega^; rho]) を逆変換して [omega; rho] を返す。
// 標準的な閉形式 (Barfoot "State Estimation for Robotics", Eq. 7.96):
//   omega = SO3_log(R)
//   V_inv(omega) = I - 0.5 * skew(omega) + coeff * skew(omega)^2
//   rho = V_inv * t
// 縮退正則化で「初期推定からどれだけ離れたか」を SE(3) 接ベクトルで測るのに使う。
Eigen::Matrix<double, 6, 1> logSE3(const Eigen::Isometry3d & transform);

}  // namespace pylot_lio::lie

#endif  // PYLOT_LIO__LIE_ALGEBRA_HPP_
