// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#ifndef PYLOT_LIO__REGISTRATION__METAL_NDT_REGISTRATION_HPP_
#define PYLOT_LIO__REGISTRATION__METAL_NDT_REGISTRATION_HPP_

#include <memory>
#include <string>

#include <Eigen/Core>

#include "metal_gpu_kernels/metal_ndt_linearizer.hpp"
#include "pylot_lio/registration/i_registration.hpp"

namespace pylot_lio
{

// Apple Metal GPU 上で NDT を解く registration。
// 各反復の線形化を Metal カーネルにオフロードし、 6x6 の solve と transform 更新・
// 収束判定は CPU で行う Newton (Gauss-Newton 近似)。
//
// target マップの扱い:
//   - 任意の IPointCloudMap から toPointCloud() で取得した点群を、 target_voxel_size_m
//     でボクセル化してガウシアン化する (VGICP と同じ流儀)。
//   - VGICP の VgicpVoxelTable を再利用する (target 表現が同一: voxel hash + Gaussian)。
//
// METAL_GPU_KERNELS_HAS_METAL 無効ビルドでは GPU が無いので CPU 参照 (linearizeNdtCpu)
// に自動フォールバックする。
class MetalNdtRegistration : public IRegistration
{
public:
  struct Config
  {
    // ボクセル一辺 [m]。 pcl_ndt / ndt_omp_lite と同じ意味。
    double resolution_m = 1.0;
    // Newton step スケール。 1.0 で純 Newton、 0.5 で半減 damping。
    double step_size = 1.0;
    // 並進 / 回転収束しきい値。
    double convergence_translation_m = 1e-4;
    double convergence_rotation_rad = 1e-4;
    int max_iterations = 30;
    // 残差距離の最大値 [m] (大外し safety net)。
    double max_correspondence_distance_m = 5.0;
    // ガウシアン構築時のボクセル最小点数。
    int min_points_per_voxel = 6;
    // 退化対策: 共分散逆行列の det floor (Metal shader 側、 fp32 安定性のため)。
    double covariance_det_floor = 1e-9;
    // ガウシアン構築時の最小固有値 floor (NDT の Σ_t の退化対策、 平面性正則化)。
    // 1e-3 程度が標準 (VGICP の voxelmap_builder のデフォルトと同じ)。
    double gaussian_eigenvalue_floor = 1e-3;
    // 近傍探索半径 (0=自ボクセルのみ、 1=27 近傍)。
    int search_radius_voxels = 1;
    // GPU を使う最小 source 点数。 これ未満は CPU 参照に自動切り替え。
    // 0 で常に GPU、 巨大値で常に CPU。
    int gpu_min_points = 50000;
    // align 対象の最小点数。
    int min_points_for_alignment = 10;
  };

  explicit MetalNdtRegistration(const Config & config);

  AlignResult align(
    const PointCloud & source_cloud_body,
    const IPointCloudMap & map_world,
    const Eigen::Isometry3d & initial_transform_world_body) override;

  std::string describe() const override;

  static bool isAvailable();

private:
  Config config_;
  std::shared_ptr<metal_gpu_kernels::MetalNdtEngine> engine_;
};

}  // namespace pylot_lio

#endif  // PYLOT_LIO__REGISTRATION__METAL_NDT_REGISTRATION_HPP_
