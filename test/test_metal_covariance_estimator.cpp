// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
//
// Metal source-covariance estimator (前処理 GPU 化) の検証。
//   - CPU 参照 (グリッド kNN + closed-form 法線 + 平面正則化) を「正解」とし、
//     GPU (fp32) が同じ共分散を fp32 許容内で再現することを確認する。
//   - 平面上の点では正則化共分散が I − (1−ε)·n·nᵀ となり、 最小固有値方向 (法線) が
//     平面法線に一致することを確認する。
//
// Metal 非対応ビルドでは gpu_used=false になり、 CPU フォールバックの自己一致だけ確認する。

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "pylot_lio/gpu/metal_covariance_estimator.hpp"

namespace pylot_lio::gpu
{

namespace
{

// z=0 平面上に格子状の点を撒き、 z 方向に薄いノイズを足す (= 平面構造)。
// spacing 間隔、 半幅 half_extent。 各点は周囲に十分な近傍を持つ。
std::vector<Eigen::Vector3d> makePlaneCloud(
  double spacing, int half_count, double z_noise_sigma)
{
  std::mt19937 rng(2026);
  std::normal_distribution<double> z_noise(0.0, z_noise_sigma);
  std::vector<Eigen::Vector3d> points;
  points.reserve(static_cast<std::size_t>((2 * half_count + 1) * (2 * half_count + 1)));
  for (int ix = -half_count; ix <= half_count; ++ix) {
    for (int iy = -half_count; iy <= half_count; ++iy) {
      points.emplace_back(ix * spacing, iy * spacing, z_noise(rng));
    }
  }
  return points;
}

// 共分散の最小固有値方向 (= 法線) を Eigen で取り出す。
Eigen::Vector3d smallestEigenDirection(const Eigen::Matrix3d & covariance)
{
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
  return solver.eigenvectors().col(0);  // 昇順 → col(0) が最小固有値方向
}

}  // namespace

TEST(MetalCovarianceEstimator, PlaneNormalIsRecovered)
{
  // 平面上の点では正則化共分散の最小固有値方向が平面法線 (z 軸) に一致するはず。
  const auto points = makePlaneCloud(0.1, 12, 0.002);
  CovarianceEstimateConfig config;
  config.num_neighbors = 10;
  config.plane_epsilon = 1e-3f;
  config.cell_size_m = 0.3f;
  config.search_radius_cells = 1;

  const auto covariances = estimateSourceCovariancesGridCpu(points, config);
  ASSERT_EQ(covariances.size(), points.size());

  // 中央付近の点 (近傍が全方位に揃う) で検証する。
  int checked = 0;
  for (std::size_t i = 0; i < points.size(); ++i) {
    if (std::abs(points[i].x()) > 0.5 || std::abs(points[i].y()) > 0.5) {
      continue;
    }
    const Eigen::Vector3d normal = smallestEigenDirection(covariances[i]);
    // 法線は ±z に近い (|n_z| ≈ 1)。
    EXPECT_GT(std::abs(normal.z()), 0.95)
      << "point " << i << " normal=" << normal.transpose();
    // 最小固有値 ≈ ε、 最大 ≈ 1。
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariances[i]);
    EXPECT_NEAR(solver.eigenvalues()(0), config.plane_epsilon, 1e-2);
    EXPECT_NEAR(solver.eigenvalues()(2), 1.0, 1e-2);
    ++checked;
  }
  EXPECT_GT(checked, 0);
}

TEST(MetalCovarianceEstimator, TooFewPointsYieldIsotropicDefault)
{
  // 3 点未満では共分散を組めない → ε·I を返す (クラッシュしない)。
  std::vector<Eigen::Vector3d> points = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
  CovarianceEstimateConfig config;
  config.plane_epsilon = 1e-3f;
  const auto covariances = estimateSourceCovariancesGridCpu(points, config);
  ASSERT_EQ(covariances.size(), 2u);
  for (const auto & covariance : covariances) {
    EXPECT_TRUE(covariance.isApprox(
      Eigen::Matrix3d::Identity() * config.plane_epsilon, 1e-9));
  }
}

TEST(MetalCovarianceEstimator, IsolatedPointFallsBackToIsotropic)
{
  // 近傍が 3 点に満たない孤立点は ε·I のまま (グリッド探索で候補が集まらない)。
  // 密集した塊 + 遠く離れた 1 点。 離れた点は近傍が自分だけ。
  auto points = makePlaneCloud(0.1, 4, 0.001);
  points.emplace_back(1000.0, 1000.0, 1000.0);  // 孤立点
  CovarianceEstimateConfig config;
  config.num_neighbors = 10;
  config.cell_size_m = 0.3f;
  config.plane_epsilon = 1e-3f;

  const auto covariances = estimateSourceCovariancesGridCpu(points, config);
  const auto & isolated = covariances.back();
  EXPECT_TRUE(isolated.isApprox(
    Eigen::Matrix3d::Identity() * config.plane_epsilon, 1e-9))
    << isolated;
}

#ifdef PYLOT_LIO_HAS_METAL

TEST(MetalCovarianceEstimator, GpuMatchesCpuWithinFp32Tolerance)
{
  const auto points = makePlaneCloud(0.1, 30, 0.003);
  CovarianceEstimateConfig config;
  config.num_neighbors = 10;
  config.plane_epsilon = 1e-3f;
  config.cell_size_m = 0.3f;
  config.search_radius_cells = 1;

  const auto cpu = estimateSourceCovariancesGridCpu(points, config);
  const auto gpu = estimateSourceCovariancesMetal(points, config);

  ASSERT_TRUE(gpu.gpu_used) << "Metal path not taken: " << gpu.error_message;
  ASSERT_EQ(gpu.covariances.size(), cpu.size());

  // 近傍選択がタイ (等距離) のとき fp32/fp64 で k 番目が入れ替わり得るので、
  // 全点厳密一致は要求せず、 平均 Frobenius 差が小さいこと + 大半の点が近いことを見る。
  double total_diff = 0.0;
  int large_diff_count = 0;
  for (std::size_t i = 0; i < cpu.size(); ++i) {
    const double diff = (gpu.covariances[i] - cpu[i]).norm();
    total_diff += diff;
    if (diff > 0.05) {
      ++large_diff_count;
    }
  }
  const double mean_diff = total_diff / static_cast<double>(cpu.size());
  EXPECT_LT(mean_diff, 1e-2) << "mean Frobenius diff too large";
  // 近傍タイによる外れは全体の 5% 未満であるべき。
  EXPECT_LT(large_diff_count, static_cast<int>(cpu.size()) / 20);
}

TEST(MetalCovarianceEstimator, GpuRecoversPlaneNormal)
{
  // GPU 経路でも平面法線が復元できることを直接確認する。
  const auto points = makePlaneCloud(0.1, 20, 0.002);
  CovarianceEstimateConfig config;
  config.num_neighbors = 10;
  config.plane_epsilon = 1e-3f;
  config.cell_size_m = 0.3f;

  const auto gpu = estimateSourceCovariancesMetal(points, config);
  ASSERT_TRUE(gpu.gpu_used);

  int checked = 0;
  for (std::size_t i = 0; i < points.size(); ++i) {
    if (std::abs(points[i].x()) > 0.5 || std::abs(points[i].y()) > 0.5) {
      continue;
    }
    const Eigen::Vector3d normal = smallestEigenDirection(gpu.covariances[i]);
    EXPECT_GT(std::abs(normal.z()), 0.9) << "point " << i;
    ++checked;
  }
  EXPECT_GT(checked, 0);
}

TEST(MetalCovarianceEstimator, EngineReuseMatchesSingleShot)
{
  // 永続エンジンの estimate() が単発関数と同じ結果を返す (gpu_used=true)。
  const auto points = makePlaneCloud(0.1, 15, 0.002);
  CovarianceEstimateConfig config;
  config.num_neighbors = 10;
  config.cell_size_m = 0.3f;

  MetalCovarianceEngine engine;
  ASSERT_TRUE(engine.isValid());
  const auto first = engine.estimate(points, config);
  const auto second = engine.estimate(points, config);  // 使い回し
  ASSERT_TRUE(first.gpu_used);
  ASSERT_TRUE(second.gpu_used);
  ASSERT_EQ(first.covariances.size(), second.covariances.size());
  for (std::size_t i = 0; i < first.covariances.size(); ++i) {
    EXPECT_TRUE(first.covariances[i].isApprox(second.covariances[i], 1e-6));
  }
}

TEST(MetalCovarianceEstimator, BenchmarkGpuVsCpu)
{
  // GPU vs CPU(逐次) の共分散推定時間を点数別に出力する。 環境依存なので assert はせず。
  const std::vector<int> half_counts = {70, 150, 320};  // ~2万 / ~9万 / ~41万点
  CovarianceEstimateConfig config;
  config.num_neighbors = 10;
  config.cell_size_m = 0.3f;

  MetalCovarianceEngine engine;
  ASSERT_TRUE(engine.isValid());

  for (int half : half_counts) {
    const auto points = makePlaneCloud(0.1, half, 0.003);
    const int iterations = 3;

    auto cpu_start = std::chrono::steady_clock::now();
    std::vector<Eigen::Matrix3d> cpu;
    for (int i = 0; i < iterations; ++i) {
      cpu = estimateSourceCovariancesGridCpu(points, config);
    }
    auto cpu_end = std::chrono::steady_clock::now();

    auto gpu_start = std::chrono::steady_clock::now();
    CovarianceEstimation gpu;
    for (int i = 0; i < iterations; ++i) {
      gpu = engine.estimate(points, config);
    }
    auto gpu_end = std::chrono::steady_clock::now();

    const double cpu_ms =
      std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count() / iterations;
    const double gpu_ms =
      std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count() / iterations;
    ASSERT_TRUE(gpu.gpu_used);
    std::printf(
      "[benchmark] points=%7zu  CPU(逐次)=%8.3f ms  GPU=%8.3f ms  speedup=%5.2fx\n",
      points.size(), cpu_ms, gpu_ms, cpu_ms / gpu_ms);
  }
}

#else  // PYLOT_LIO_HAS_METAL

TEST(MetalCovarianceEstimator, MetalFallsBackToCpuWhenUnavailable)
{
  const auto points = makePlaneCloud(0.1, 10, 0.002);
  CovarianceEstimateConfig config;
  config.num_neighbors = 10;
  config.cell_size_m = 0.3f;

  const auto gpu = estimateSourceCovariancesMetal(points, config);
  EXPECT_FALSE(gpu.gpu_used);
  const auto cpu = estimateSourceCovariancesGridCpu(points, config);
  ASSERT_EQ(gpu.covariances.size(), cpu.size());
  for (std::size_t i = 0; i < cpu.size(); ++i) {
    EXPECT_TRUE(gpu.covariances[i].isApprox(cpu[i], 1e-9));
  }
}

#endif  // PYLOT_LIO_HAS_METAL

}  // namespace pylot_lio::gpu

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
