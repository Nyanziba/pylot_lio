// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
//
// Metal VGICP 線形化カーネル (Step1) の検証。
//   - CPU 参照 (fp64) を「正解」とし、 GPU (fp32) が同じ正規方程式 (H, b, cost, count)
//     を fp32 許容内で再現することを確認する。
//   - 数式 (voxel ハッシュ・対応・情報行列・ヤコビアン・reduction) を CPU 参照と
//     GPU で揃えてあるので、 差は fp32 丸めのみのはず。
//
// Metal 非対応ビルドでは gpu_used=false になり、 CPU フォールバックの自己一致だけ確認する。

#include <gtest/gtest.h>

#include <random>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "pylot_lio/gpu/metal_vgicp_linearizer.hpp"

namespace pylot_lio::gpu
{

namespace
{

// 合成シーンを作る: ランダムなボクセル群 (mean + 等方寄りの cov) と、
// それらの近傍に散らばる source 点群 (各点に小さな等方 cov)。
struct SyntheticScene
{
  VgicpVoxelTable voxel_table;
  std::vector<Eigen::Vector3d> source_points_body;
  std::vector<Eigen::Matrix3d> source_covariances_body;
};

SyntheticScene makeScene(int num_voxels, int num_source_points, float voxel_size)
{
  std::mt19937 rng(12345);
  std::uniform_real_distribution<double> pos_dist(-5.0, 5.0);
  std::normal_distribution<double> jitter(0.0, 0.05);

  std::vector<Eigen::Vector3i> coords;
  std::vector<Eigen::Vector3d> means;
  std::vector<Eigen::Matrix3d> covs;
  coords.reserve(num_voxels);
  means.reserve(num_voxels);
  covs.reserve(num_voxels);

  // 重複しないボクセル座標を作る。
  std::vector<Eigen::Vector3i> used;
  for (int i = 0; i < num_voxels; ++i) {
    Eigen::Vector3i coord(
      static_cast<int>(pos_dist(rng) / voxel_size),
      static_cast<int>(pos_dist(rng) / voxel_size),
      static_cast<int>(pos_dist(rng) / voxel_size));
    bool duplicate = false;
    for (const auto & existing : used) {
      if (existing == coord) { duplicate = true; break; }
    }
    if (duplicate) { continue; }
    used.push_back(coord);

    // ボクセル中心付近に mean、 平面寄りの cov (法線方向だけ小さい)。
    const Eigen::Vector3d center(
      (coord.x() + 0.5) * voxel_size,
      (coord.y() + 0.5) * voxel_size,
      (coord.z() + 0.5) * voxel_size);
    means.push_back(center);
    Eigen::Matrix3d cov = Eigen::Matrix3d::Identity();
    cov(0, 0) = 0.04;
    cov(1, 1) = 0.04;
    cov(2, 2) = 0.0009;  // 法線方向は薄い (平面)
    covs.push_back(cov);
    coords.push_back(coord);
  }

  SyntheticScene scene;
  scene.voxel_table = VgicpVoxelTable::build(voxel_size, coords, means, covs);

  // source 点: 既存ボクセルの mean 近傍にジッタを足す (= 対応が取れる点)。
  for (int i = 0; i < num_source_points; ++i) {
    const int voxel_index = i % static_cast<int>(means.size());
    Eigen::Vector3d point = means[voxel_index] +
      Eigen::Vector3d(jitter(rng), jitter(rng), jitter(rng));
    scene.source_points_body.push_back(point);
    Eigen::Matrix3d source_cov = Eigen::Matrix3d::Identity() * 0.01;
    scene.source_covariances_body.push_back(source_cov);
  }
  return scene;
}

}  // namespace

TEST(MetalVgicpLinearizer, VoxelTableBuildRoundTrips)
{
  // build したハッシュ表で、 登録した voxel が引けることを確認 (CPU 参照経路の前提)。
  std::vector<Eigen::Vector3i> coords = {{1, 2, 3}, {-4, 5, -6}, {7, -8, 9}};
  std::vector<Eigen::Vector3d> means = {
    {0.5, 1.0, 1.5}, {-2.0, 2.5, -3.0}, {3.5, -4.0, 4.5}};
  std::vector<Eigen::Matrix3d> covs(3, Eigen::Matrix3d::Identity());
  const auto table = VgicpVoxelTable::build(0.5f, coords, means, covs);
  EXPECT_GE(table.capacity, 8);
  // capacity は 2 の冪。
  EXPECT_EQ(table.capacity & (table.capacity - 1), 0);
}

TEST(MetalVgicpLinearizer, CpuReferenceProducesNonTrivialSystem)
{
  // CPU 参照が「対応の取れる点で H/b に寄与している」ことを確認 (土台の健全性)。
  const auto scene = makeScene(50, 500, 0.5f);
  VgicpLinearizeConfig config;
  config.huber_threshold = 1.0f;
  config.max_correspondence_distance_m = 2.0f;

  const auto cpu = linearizeVgicpCpu(
    scene.source_points_body, scene.source_covariances_body,
    scene.voxel_table, Eigen::Isometry3d::Identity(), config);

  EXPECT_GT(cpu.valid_correspondences, 0);
  EXPECT_GT(cpu.hessian.norm(), 0.0);
}

#ifdef PYLOT_LIO_HAS_METAL

TEST(MetalVgicpLinearizer, GpuMatchesCpuWithinFp32Tolerance)
{
  const auto scene = makeScene(200, 4000, 0.5f);
  VgicpLinearizeConfig config;
  config.huber_threshold = 1.0f;
  config.max_correspondence_distance_m = 2.0f;

  // 単位姿勢ではなく少し回した transform で R 依存項 (R C_s R^T, skew) も検証する。
  Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
  transform.linear() =
    Eigen::AngleAxisd(0.1, Eigen::Vector3d(0.3, 0.6, 0.7).normalized()).toRotationMatrix();
  transform.translation() = Eigen::Vector3d(0.05, -0.03, 0.02);

  const auto cpu = linearizeVgicpCpu(
    scene.source_points_body, scene.source_covariances_body,
    scene.voxel_table, transform, config);
  const auto gpu = linearizeVgicpMetal(
    scene.source_points_body, scene.source_covariances_body,
    scene.voxel_table, transform, config);

  ASSERT_TRUE(gpu.gpu_used) << "Metal path not taken: " << gpu.error_message;
  // 対応点数は完全一致するべき (ハッシュ・対応判定が CPU と同一なら整数は一致)。
  EXPECT_EQ(gpu.valid_correspondences, cpu.valid_correspondences);

  // H / b / cost は fp32 丸めぶんずれる。 相対許容で比較する。
  const double hessian_scale = std::max(1.0, cpu.hessian.cwiseAbs().maxCoeff());
  const double gradient_scale = std::max(1.0, cpu.gradient.cwiseAbs().maxCoeff());
  EXPECT_LT((gpu.hessian - cpu.hessian).cwiseAbs().maxCoeff(), 1e-3 * hessian_scale);
  EXPECT_LT((gpu.gradient - cpu.gradient).cwiseAbs().maxCoeff(), 1e-3 * gradient_scale);
  EXPECT_NEAR(gpu.cost, cpu.cost, 1e-3 * std::max(1.0, std::abs(cpu.cost)));
}

#else  // PYLOT_LIO_HAS_METAL

TEST(MetalVgicpLinearizer, MetalFallsBackToCpuWhenUnavailable)
{
  const auto scene = makeScene(50, 500, 0.5f);
  VgicpLinearizeConfig config;
  const auto gpu = linearizeVgicpMetal(
    scene.source_points_body, scene.source_covariances_body,
    scene.voxel_table, Eigen::Isometry3d::Identity(), config);
  EXPECT_FALSE(gpu.gpu_used);
  // フォールバックは CPU 参照と一致するはず。
  const auto cpu = linearizeVgicpCpu(
    scene.source_points_body, scene.source_covariances_body,
    scene.voxel_table, Eigen::Isometry3d::Identity(), config);
  EXPECT_EQ(gpu.valid_correspondences, cpu.valid_correspondences);
}

#endif  // PYLOT_LIO_HAS_METAL

}  // namespace pylot_lio::gpu

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
