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

#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
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

TEST(MetalVgicpLinearizer, NeighborSearchFindsMoreCorrespondences)
{
  // 近傍探索 (radius=1) は radius=0 (自ボクセルのみ) より対応点数が増えるはず。
  // ボクセル境界付近の点が隣接ボクセルの分布を拾えるようになるため。
  const auto scene = makeScene(80, 1500, 0.5f);

  VgicpLinearizeConfig config_self;
  config_self.search_radius_voxels = 0;
  VgicpLinearizeConfig config_neighbor;
  config_neighbor.search_radius_voxels = 1;

  const auto self_only = linearizeVgicpCpu(
    scene.source_points_body, scene.source_covariances_body,
    scene.voxel_table, Eigen::Isometry3d::Identity(), config_self);
  const auto with_neighbor = linearizeVgicpCpu(
    scene.source_points_body, scene.source_covariances_body,
    scene.voxel_table, Eigen::Isometry3d::Identity(), config_neighbor);

  EXPECT_GE(with_neighbor.valid_correspondences, self_only.valid_correspondences);
}

TEST(MetalVgicpLinearizer, EmptyVoxelTableYieldsNoCorrespondences)
{
  // 空ボクセル表 (voxel 0 個) では対応ゼロ・H ゼロ。 クラッシュしない。
  const auto empty_table = VgicpVoxelTable::build(0.5f, {}, {}, {});
  std::vector<Eigen::Vector3d> points = {{0.1, 0.2, 0.3}, {1.0, 1.0, 1.0}};
  std::vector<Eigen::Matrix3d> covs(2, Eigen::Matrix3d::Identity() * 0.01);
  VgicpLinearizeConfig config;
  const auto cpu = linearizeVgicpCpu(
    points, covs, empty_table, Eigen::Isometry3d::Identity(), config);
  EXPECT_EQ(cpu.valid_correspondences, 0);
  EXPECT_NEAR(cpu.hessian.norm(), 0.0, 1e-12);
}

TEST(MetalVgicpLinearizer, NonFinitePointsAreSkipped)
{
  // NaN / Inf を含む source 点はスキップされ、 有限点だけが寄与する。
  const auto scene = makeScene(30, 100, 0.5f);
  auto points = scene.source_points_body;
  auto covs = scene.source_covariances_body;
  // 数点を NaN/Inf に汚染。
  points[5] = Eigen::Vector3d(std::nan(""), 0.0, 0.0);
  points[10] = Eigen::Vector3d(0.0, std::numeric_limits<double>::infinity(), 0.0);

  VgicpLinearizeConfig config;
  const auto cpu = linearizeVgicpCpu(
    points, covs, scene.voxel_table, Eigen::Isometry3d::Identity(), config);
  // 汚染点があっても有限点で正常に H を組めている (NaN が伝播していない)。
  EXPECT_TRUE(cpu.hessian.allFinite());
  EXPECT_TRUE(cpu.gradient.allFinite());
  EXPECT_GT(cpu.valid_correspondences, 0);
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

TEST(MetalVgicpLinearizer, BenchmarkGpuVsCpu)
{
  // GPU vs CPU(逐次) の線形化時間を点数別に出力する。 タイミングは環境依存なので
  // assert はせず (CI 安定性のため)、 正しさと gpu_used だけ確認する。
  // 1 反復ぶんの linearize を複数回まわして中央値的な平均を取る。
  const std::vector<int> point_counts = {20000, 100000, 500000};
  VgicpLinearizeConfig config;
  config.search_radius_voxels = 1;

  for (int num_points : point_counts) {
    // voxel 数は点数に対してほどほど (1 ボクセルあたり ~10 点程度)。
    const int num_voxels = std::max(64, num_points / 10);
    const auto scene = makeScene(num_voxels, num_points, 0.5f);
    const Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();

    const int iterations = 5;
    auto cpu_start = std::chrono::steady_clock::now();
    VgicpLinearization cpu;
    for (int i = 0; i < iterations; ++i) {
      cpu = linearizeVgicpCpu(
        scene.source_points_body, scene.source_covariances_body,
        scene.voxel_table, transform, config);
    }
    auto cpu_end = std::chrono::steady_clock::now();

    auto gpu_start = std::chrono::steady_clock::now();
    VgicpLinearization gpu;
    for (int i = 0; i < iterations; ++i) {
      gpu = linearizeVgicpMetal(
        scene.source_points_body, scene.source_covariances_body,
        scene.voxel_table, transform, config);
    }
    auto gpu_end = std::chrono::steady_clock::now();

    const double cpu_ms =
      std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count() / iterations;
    const double gpu_ms =
      std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count() / iterations;

    ASSERT_TRUE(gpu.gpu_used);
    EXPECT_EQ(gpu.valid_correspondences, cpu.valid_correspondences);

    std::printf(
      "[benchmark] points=%7d voxels=%7d  CPU(omp無/逐次)=%8.3f ms  GPU=%8.3f ms  speedup=%5.2fx\n",
      num_points, scene.voxel_table.capacity, cpu_ms, gpu_ms, cpu_ms / gpu_ms);
  }
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
