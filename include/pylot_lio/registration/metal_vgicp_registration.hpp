// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#ifndef PYLOT_LIO__REGISTRATION__METAL_VGICP_REGISTRATION_HPP_
#define PYLOT_LIO__REGISTRATION__METAL_VGICP_REGISTRATION_HPP_

#include <string>

#include "pylot_lio/registration/i_registration.hpp"

namespace pylot_lio
{

// Apple Metal GPU 上で VGICP を解く registration (Step2)。
// 各反復の線形化 (正規方程式 H, b の構築) を Metal カーネルにオフロードし、
// 6x6 の solve と transform 更新・収束判定は CPU で行う Gauss-Newton。
//
// 制約 (Step2 時点):
//   - target マップは VoxelMap であること (dynamic_cast で取り出す)。 他の
//     IPointCloudMap 実装が渡された場合は未収束 (converged=false) を返す。
//   - source 共分散は内部で PCL KdTree から計算する (plain_gicp と同方式)。
//   - source/voxel バッファは反復ごとに再アップロードする素朴版 (最適化は後段)。
//   - fp32 精度 (Apple GPU は fp64 不可)。 omp/tbb とビット一致はしない。
//
// PYLOT_LIO_HAS_METAL 無効ビルドでは align は常に converged=false を返す
// (factory 側で plain_gicp にフォールバックさせる前提)。
class MetalVgicpRegistration : public IRegistration
{
public:
  struct Config
  {
    int max_iterations = 20;
    double convergence_translation_m = 1e-4;
    double convergence_rotation_rad = 1e-4;
    double max_correspondence_distance_m = 2.0;
    double huber_threshold = 1.0;

    // source 共分散 (GICP の distribution-to-distribution 化)。 plain_gicp と同義。
    int source_covariance_num_neighbors = 10;
    double source_covariance_plane_epsilon = 1e-3;

    // 近傍ボクセル探索半径 (0=自ボクセルのみ, 1=27 近傍)。
    int search_radius_voxels = 1;
  };

  explicit MetalVgicpRegistration(const Config & config);

  AlignResult align(
    const PointCloud & source_cloud_body,
    const IPointCloudMap & map_world,
    const Eigen::Isometry3d & initial_transform_world_body) override;

  std::string describe() const override;

  // このビルドで Metal バックエンドが利用可能か (PYLOT_LIO_HAS_METAL)。
  // factory が "metal_vgicp" 指定時にフォールバック要否を判定するのに使う。
  static bool isAvailable();

private:
  Config config_;
};

}  // namespace pylot_lio

#endif  // PYLOT_LIO__REGISTRATION__METAL_VGICP_REGISTRATION_HPP_
