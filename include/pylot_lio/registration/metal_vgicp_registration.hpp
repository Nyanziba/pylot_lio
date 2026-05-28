// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#ifndef PYLOT_LIO__REGISTRATION__METAL_VGICP_REGISTRATION_HPP_
#define PYLOT_LIO__REGISTRATION__METAL_VGICP_REGISTRATION_HPP_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include <Eigen/Core>

#include "metal_gpu_kernels/metal_covariance_estimator.hpp"
#include "metal_gpu_kernels/metal_vgicp_linearizer.hpp"
#include "pylot_lio/registration/i_registration.hpp"

namespace pylot_lio
{

// Apple Metal GPU 上で VGICP を解く registration (Step2)。
// 各反復の線形化 (正規方程式 H, b の構築) を Metal カーネルにオフロードし、
// 6x6 の solve と transform 更新・収束判定は CPU で行う Gauss-Newton。
//
// target マップの扱い (マップ非依存):
//   - voxel_map / normal_map : 保存済みガウス分布 (toGaussianVoxels) を直接使う。
//   - voxel_random_map / voxel_keyframe_submap / kd_tree_map : toPointCloud() の生点群を
//     target_voxel_size_m でボクセル化してガウス分布を復元する。
//   どのマップでもガウスが空 (点が無い) なら未収束 (converged=false) を返す。
//
// その他:
//   - source 共分散は GPU (Metal) か PCL KdTree で計算 (use_gpu_source_covariance)。
//   - fp32 精度 (Apple GPU は fp64 不可)。 omp/tbb とビット一致はしない。
//
// METAL_GPU_KERNELS_HAS_METAL 無効ビルドでは align は常に converged=false を返す
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

    // source 共分散の推定を GPU (Metal) で行うか。 true なら前処理 (k 近傍探索 +
    // 平面正則化) を GPU にオフロードする (glim 相当の前処理 GPU 化)。 GPU 無効ビルド /
    // デバイス無では自動的に CPU 参照 (グリッド kNN) にフォールバックする。 false なら
    // 従来の PCL KdTree (computeSourceCovariances) を使う。
    bool use_gpu_source_covariance = true;
    // GPU 共分散推定の近傍探索グリッドのセル一辺 [m]。 source 点群密度に対して
    // 「k 近傍が 27 近傍セルに収まる」 程度に取る (粗すぎると候補過多、 細かすぎると不足)。
    double source_covariance_cell_size_m = 0.5;

    // 近傍ボクセル探索半径 (0=自ボクセルのみ, 1=27 近傍)。
    int search_radius_voxels = 1;

    // GPU を使う最小 source 点数。 これ未満は GPU 起動オーバヘッドが支配的になり
    // CPU VGICP の方が速いため、 align 内で自動的に CPU 経路に切り替える。
    // Apple M4 ベンチの交差点 (~5万〜10万点で GPU が逆転) を踏まえた既定値。
    // 0 にすると常に GPU、 非常に大きくすると常に CPU。
    int gpu_min_points = 50000;

    // 多重解像度 (coarse-to-fine) VGICP。 glim の voxelmap_levels 相当。
    //   levels=1: 単一解像度 (map の voxel_size のみ)。
    //   levels>=2: 粗いレベルを scaling_factor 倍ずつ作り、 粗→細で GN を回す。
    //     粗いレベルは収束盆が広く初期推定が悪くても引き込みやすい。 細いレベルで精度。
    // 粗レベルのガウス分布は細レベルのセルを平行軸定理でマージして作る。
    int voxelmap_levels = 2;
    double voxelmap_scaling_factor = 2.0;

    // ---- voxel_map / normal_map 以外のマップ (point ベース) を target にするときの設定 ----
    // voxel_keyframe_submap / voxel_random_map / kd_tree_map は保存済みガウス分布を持たない
    // ので、 toPointCloud() の生点群をこの voxel_size でボクセル化してガウスを復元する。
    // voxel_map / normal_map のときはマップ自身の voxel_size / eigen_floor が使われ、 この
    // 3 つは無視される。
    double target_voxel_size_m = 0.5;
    double covariance_eigen_floor = 1e-3;
    int min_points_per_voxel = 1;

    // ---- 地面平面 leveling 拘束 (IMU 重力が使えないときのピッチドリフト対策) ----
    // 各スキャンで支配的地面の法線を world-up (0,0,1) に合わせる soft 拘束を GN に加える。
    // 平地走行が前提。 詳細は ground_constraint.hpp。
    bool enable_ground_constraint = false;
    // 拘束の強さ。 内部で VGICP の回転剛性 (回転ヘッシアン対角の平均) に対する比として
    // スケールするので、 1.0 で「幾何項の回転剛性と同程度」、 大きいほど地面水平が優先。
    double ground_constraint_weight = 1.0;
    // 最低 world z からこの高さ [m] までを地面候補とする。
    double ground_band_m = 0.5;
    // 推定法線が (0,0,1) からこの角度 [deg] を超えて傾いていたら地面とみなさない。
    double ground_max_tilt_deg = 30.0;
    // damping: 1 align あたりの leveling 補正上限 [deg]。 0 で無制限 (hard 拘束)。
    double ground_max_correction_per_frame_deg = 1.0;
    // 振動ゲート: 前フレームの地面法線 (body) からの角度差がこれ [deg] を超えるフレームは
    // 拘束をスキップ (車体ピッチ振動中とみなす)。 0 で振動ゲート無効。
    double ground_vibration_threshold_deg = 3.0;
  };

  explicit MetalVgicpRegistration(const Config & config);

  AlignResult align(
    const PointCloud & source_cloud_body,
    const IPointCloudMap & map_world,
    const Eigen::Isometry3d & initial_transform_world_body) override;

  std::string describe() const override;

  // このビルドで Metal バックエンドが利用可能か (METAL_GPU_KERNELS_HAS_METAL)。
  // factory が "metal_vgicp" 指定時にフォールバック要否を判定するのに使う。
  static bool isAvailable();

private:
  Config config_;
  // GPU リソース (device/PSO) を align 間で使い回すための永続エンジン。
  // PSO コンパイル (~100ms) を毎 align で繰り返さないよう、 registration インスタンスが
  // 1 つ保持する。 Metal 無効ビルドでは isValid()=false のスタブ。
  std::shared_ptr<metal_gpu_kernels::MetalVgicpEngine> engine_;
  // source 共分散推定 (前処理) を GPU で行う永続エンジン。 linearizer とは別の PSO を
  // 持つため独立に保持する。 use_gpu_source_covariance=false のときは未使用。
  std::shared_ptr<metal_gpu_kernels::MetalCovarianceEngine> covariance_engine_;
  // 振動検出用: 前 align で採用した地面法線 (body)。 次フレームでの急変判定に使う。
  std::optional<Eigen::Vector3d> previous_ground_normal_body_;
  // CV 予測診断用: 全 align の通し番号。 N スキャンごとに [CV] ログを出す間引きに使う。
  std::uint64_t align_call_count_ = 0;
};

}  // namespace pylot_lio

#endif  // PYLOT_LIO__REGISTRATION__METAL_VGICP_REGISTRATION_HPP_
