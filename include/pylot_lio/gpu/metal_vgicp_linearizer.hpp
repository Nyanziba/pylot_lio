// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#ifndef PYLOT_LIO__GPU__METAL_VGICP_LINEARIZER_HPP_
#define PYLOT_LIO__GPU__METAL_VGICP_LINEARIZER_HPP_

#include <cstdint>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace pylot_lio::gpu
{

// ============================================================
// Metal VGICP linearizer (Step1: GPU 線形化カーネル + CPU 検証)
// ============================================================
// glim / gtsam_points の GaussianVoxelMapGPU + VGICP を Metal に移植した最初の部品。
// 「固定 transform で 1 反復ぶんの正規方程式 (H, b) を組む」 線形化だけを担う。
// GN ループ・収束判定・transform 更新は呼び出し側 (将来の MetalVgicpRegistration)。
//
// VGICP 定式化: source 点 p (body) を T で world に移し、 その world 座標が属する
// ボクセルの累積ガウス分布 N(μ_t, C_t) を target とする (KdTree 不要)。
//   残差     d = μ_t - T p
//   重み      Ω = (C_t + R C_s R^T)^{-1}
//   ヤコビアン J = [ skew(T p) | -I ]   (SE(3) 左摂動、 残差 d = μ_t - T(p) の微分)
//   H += w J^T Ω J,  b += -w J^T Ω d,  cost += w dᵀΩd   (w は Huber 重み)

// ボクセルハッシュ表 (CPU 側で構築 → GPU へアップロードする中間表現)。
// オープンアドレッシング: capacity スロット (2 の冪)。 空きスロットは occupied=0。
// keys/means/covariances は capacity 長で、 同じスロット index に対応する。
struct VgicpVoxelTable
{
  int capacity = 0;            // スロット数 (2 の冪、 voxel 数の 2 倍以上を推奨)
  float voxel_size_m = 0.5f;   // ボクセル一辺 [m]

  // 各スロット (capacity 個)。
  std::vector<std::uint8_t> occupied;       // capacity        : 1=有効, 0=空き
  std::vector<std::int32_t> voxel_keys_xyz; // capacity * 3     : ボクセル整数座標
  std::vector<float> means_xyz;             // capacity * 3     : μ_t (world)
  std::vector<float> covariances;           // capacity * 9     : C_t (world, row-major)

  // CPU 側でハッシュ表を組むヘルパ。 voxel_coords / means / covs は同じ長さ・並び順。
  // capacity は num_voxels の 2 倍以上の最小 2 冪に自動設定する。
  static VgicpVoxelTable build(
    float voxel_size_m,
    const std::vector<Eigen::Vector3i> & voxel_coords,
    const std::vector<Eigen::Vector3d> & means,
    const std::vector<Eigen::Matrix3d> & covariances);
};

struct VgicpLinearizeConfig
{
  float huber_threshold = 1.0f;
  float max_correspondence_distance_m = 2.0f;

  // 近傍ボクセル探索半径 (ボクセル単位)。
  //   0: source 点が属する 1 ボクセルだけを引く (最速)
  //   1: 3x3x3 = 27 近傍を走査し、 mean が最も近い有効ボクセルを target にする
  // 境界付近の点は floor で決まる自ボクセルが空 / 隣の分布の方が近いことがあるため、
  // 1 にすると対応の脱落・誤対応が減る (glim の VGICP に近い挙動)。 CPU/GPU 同一。
  int search_radius_voxels = 1;
};

// 1 反復ぶんの線形化結果。 GPU の fp32 部分和を CPU で合算して double で保持する。
struct VgicpLinearization
{
  Eigen::Matrix<double, 6, 6> hessian = Eigen::Matrix<double, 6, 6>::Zero();
  Eigen::Matrix<double, 6, 1> gradient = Eigen::Matrix<double, 6, 1>::Zero();
  double cost = 0.0;
  int valid_correspondences = 0;
  bool gpu_used = false;        // GPU 経路を通ったか (false = Metal 無効ビルド/デバイス無)
  std::string error_message;
};

// CPU 参照実装 (fp64)。 GPU 結果の「正解」として、 また Metal 非対応ビルドの
// フォールバックとして使う。 source は body フレーム、 transform は T_world_body。
VgicpLinearization linearizeVgicpCpu(
  const std::vector<Eigen::Vector3d> & source_points_body,
  const std::vector<Eigen::Matrix3d> & source_covariances_body,
  const VgicpVoxelTable & voxel_table,
  const Eigen::Isometry3d & transform_world_body,
  const VgicpLinearizeConfig & config);

// Metal GPU 実装。 PYLOT_LIO_HAS_METAL が無効なら gpu_used=false で即返す。
// 数式は linearizeVgicpCpu と同一だが内部 fp32 のため結果は厳密一致しない。
VgicpLinearization linearizeVgicpMetal(
  const std::vector<Eigen::Vector3d> & source_points_body,
  const std::vector<Eigen::Matrix3d> & source_covariances_body,
  const VgicpVoxelTable & voxel_table,
  const Eigen::Isometry3d & transform_world_body,
  const VgicpLinearizeConfig & config);

}  // namespace pylot_lio::gpu

#endif  // PYLOT_LIO__GPU__METAL_VGICP_LINEARIZER_HPP_
