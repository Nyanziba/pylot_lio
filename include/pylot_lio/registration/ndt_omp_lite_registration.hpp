// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#ifndef PYLOT_LIO__REGISTRATION__NDT_OMP_LITE_REGISTRATION_HPP_
#define PYLOT_LIO__REGISTRATION__NDT_OMP_LITE_REGISTRATION_HPP_

#include <string>

#include "pylot_lio/registration/i_registration.hpp"

namespace pylot_lio
{

// NDT (Magnusson 2009) のスクラッチ実装。 target を PCL の VoxelGridCovariance で
// ボクセル化してガウシアン化し、 各 source 点について「変換後点が落ちるボクセルの
// 負対数尤度」を SE(3) 上で Gauss-Newton (NDT 標準 = Magnusson の Newton 法) で
// 最小化する。 source 点ごとのスコア / 勾配 / Hessian 累積を OpenMP で並列化する。
//
// pcl::NDT との位置づけ:
//   - 計算結果は概ね一致する (同じ目的関数、 同じ target 表現)。
//   - こちらは Hessian 累積を OpenMP parallel + per-thread accumulator で行うため、
//     大規模 source 点群でスケールしやすい (Mid-360 60k 点で 2〜4 倍速を目指す)。
//   - PCL の More-Thuente line search ではなく、 単純 damping (固定 step_size に
//     スケールしてから不採用なら半減) を使う。 実装の簡潔さ優先。
class NdtOmpLiteRegistration : public IRegistration
{
public:
  struct Config
  {
    // ボクセル一辺 [m]。 pcl_ndt と同じ意味。
    double resolution_m = 1.0;

    // Newton 更新の初期 step スケール (0〜1)。 1.0 で生の Newton step、
    // 0.5 で半分。 大きいと振動、 小さいと収束遅。
    double step_size = 1.0;

    // 並進収束判定 [m]: |delta_t| < eps_t かつ |delta_r| < eps_r で停止。
    double transformation_epsilon_m = 1e-4;
    double rotation_epsilon_rad = 1e-4;

    // 最大反復回数。
    int max_iterations = 30;

    // OpenMP スレッド数。 <= 0 で omp_get_max_threads()、 1 で逐次。
    int num_threads = 0;

    // ボクセルあたりの最小点数。 これ未満のボクセルはガウシアンを作らず破棄する。
    // PCL の VoxelGridCovariance のデフォルトは 6。
    int min_points_per_voxel = 6;

    // 共分散の最小固有値 floor (退化対策)。 ガウシアン共分散の最小固有値を
    // この値以下にクランプして逆行列が爆発するのを防ぐ。
    double covariance_eigenvalue_floor = 1e-3;

    // align 対象の最小点数。
    int min_points_for_alignment = 10;
  };

  explicit NdtOmpLiteRegistration(const Config & config);

  AlignResult align(
    const PointCloud & source_cloud_body,
    const IPointCloudMap & map_world,
    const Eigen::Isometry3d & initial_transform_world_body) override;

  std::string describe() const override;

private:
  Config config_;
};

}  // namespace pylot_lio

#endif  // PYLOT_LIO__REGISTRATION__NDT_OMP_LITE_REGISTRATION_HPP_
