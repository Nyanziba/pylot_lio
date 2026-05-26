// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#include <gtest/gtest.h>

#include <cmath>

#include "pylot_lio/lie_algebra.hpp"

namespace pylot_lio::lie
{

constexpr double kTolerance = 1e-9;

TEST(LieAlgebra, SkewCrossProductIdentity)
{
  // skew(a) * b は a x b と等しくなければならない。
  const Eigen::Vector3d a(1.0, 2.0, 3.0);
  const Eigen::Vector3d b(4.0, 5.0, 6.0);
  const Eigen::Vector3d cross_via_skew = skew(a) * b;
  const Eigen::Vector3d cross_via_eigen = a.cross(b);
  EXPECT_NEAR((cross_via_skew - cross_via_eigen).norm(), 0.0, kTolerance);
}

TEST(LieAlgebra, ExpZeroIsIdentity)
{
  EXPECT_NEAR(
    (expSO3(Eigen::Vector3d::Zero()) - Eigen::Matrix3d::Identity()).norm(),
    0.0, kTolerance);
}

TEST(LieAlgebra, ExpAngleAroundZAxis)
{
  const double angle = M_PI / 4.0;
  const Eigen::Matrix3d rotation = expSO3(Eigen::Vector3d(0.0, 0.0, angle));
  Eigen::Matrix3d expected;
  expected <<
    std::cos(angle), -std::sin(angle), 0.0,
    std::sin(angle),  std::cos(angle), 0.0,
    0.0,              0.0,             1.0;
  EXPECT_NEAR((rotation - expected).norm(), 0.0, kTolerance);
}

TEST(LieAlgebra, LogExpRoundTrip)
{
  // log(exp(phi)) = phi が成り立つこと (|phi| < pi の範囲)。
  for (const Eigen::Vector3d & phi : {
      Eigen::Vector3d(0.1, 0.2, 0.3),
      Eigen::Vector3d(-0.5, 1.0, 0.2),
      Eigen::Vector3d(1e-10, 0.0, 0.0),  // 小角度の Taylor 経路
    })
  {
    const Eigen::Vector3d recovered = logSO3(expSO3(phi));
    EXPECT_NEAR((recovered - phi).norm(), 0.0, 1e-7)
      << "phi = " << phi.transpose();
  }
}

TEST(LieAlgebra, ExpProducesRotationMatrix)
{
  // 出力は正規直交かつ行列式 +1 を満たすこと。
  const Eigen::Matrix3d rotation = expSO3(Eigen::Vector3d(0.7, -0.4, 0.9));
  const Eigen::Matrix3d should_be_identity = rotation.transpose() * rotation;
  EXPECT_NEAR((should_be_identity - Eigen::Matrix3d::Identity()).norm(), 0.0, 1e-9);
  EXPECT_NEAR(rotation.determinant(), 1.0, 1e-9);
}

TEST(LieAlgebra, RightJacobianAtZeroIsIdentity)
{
  EXPECT_NEAR(
    (rightJacobianSO3(Eigen::Vector3d::Zero()) - Eigen::Matrix3d::Identity()).norm(),
    0.0, kTolerance);
}

TEST(LieAlgebra, RightJacobianAndInverseAreInverses)
{
  const Eigen::Vector3d phi(0.3, -0.2, 0.5);
  const Eigen::Matrix3d product =
    rightJacobianSO3(phi) * rightJacobianInverseSO3(phi);
  EXPECT_NEAR((product - Eigen::Matrix3d::Identity()).norm(), 0.0, 1e-9);
}

TEST(LieAlgebra, NormalizeProjectsOntoSO3)
{
  Eigen::Matrix3d not_quite_rotation = expSO3(Eigen::Vector3d(0.1, 0.2, 0.3));
  not_quite_rotation(0, 0) += 1e-3;  // 故意に直交性を崩す
  const Eigen::Matrix3d projected = normalizeRotation(not_quite_rotation);
  const Eigen::Matrix3d should_be_identity = projected.transpose() * projected;
  EXPECT_NEAR((should_be_identity - Eigen::Matrix3d::Identity()).norm(), 0.0, 1e-9);
  EXPECT_NEAR(projected.determinant(), 1.0, 1e-9);
}

TEST(LieAlgebra, LogSE3OnIdentityIsZero)
{
  const auto twist = logSE3(Eigen::Isometry3d::Identity());
  EXPECT_NEAR(twist.norm(), 0.0, 1e-12);
}

TEST(LieAlgebra, LogSE3PureTranslationGivesTranslationOnly)
{
  Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
  transform.translation() = Eigen::Vector3d(1.5, -2.0, 3.0);
  const auto twist = logSE3(transform);
  EXPECT_NEAR(twist.head<3>().norm(), 0.0, 1e-9);
  EXPECT_NEAR((twist.tail<3>() - transform.translation()).norm(), 0.0, 1e-9);
}

TEST(LieAlgebra, LogSE3RecoversRotationVector)
{
  Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
  transform.linear() = expSO3(Eigen::Vector3d(0.2, -0.1, 0.3));
  const auto twist = logSE3(transform);
  EXPECT_NEAR((twist.head<3>() - Eigen::Vector3d(0.2, -0.1, 0.3)).norm(), 0.0, 1e-7);
}

}  // namespace pylot_lio::lie

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
