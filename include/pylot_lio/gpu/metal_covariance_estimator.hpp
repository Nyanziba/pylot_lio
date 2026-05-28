// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#ifndef PYLOT_LIO__GPU__METAL_COVARIANCE_ESTIMATOR_HPP_
#define PYLOT_LIO__GPU__METAL_COVARIANCE_ESTIMATOR_HPP_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace pylot_lio::gpu
{

// ============================================================
// Metal source-covariance estimator (前処理 GPU 化)
// ============================================================
// GICP の distribution-to-distribution 化に必要な source 点群の局所共分散 C_s を
// GPU で推定する。 これまで CPU の PCL KdTree (computeSourceCovariances) が担っていた
// 「k 近傍探索 + 各点の固有値分解 + 平面正則化」 を Metal にオフロードする。
//
// glim / gtsam_points と同じ妥協として、 近傍探索は KdTree ではなく空間ハッシュ
// グリッドで行う (GPU で KdTree は不得手)。 そのため厳密な k 近傍とは一致しないが、
// 共分散は実質同等。 GPU は fp32。
//
// 平面正則化の単純化 (重要):
//   正則化共分散 = eigenvectors · diag(ε,1,1) · eigenvectorsᵀ は
//   = I − (1−ε)·n·nᵀ   (n = 最小固有値方向の単位法線) と数学的に等価。
//   よって全固有値分解は不要で、 最小固有値に対応する固有ベクトル (法線) だけを
//   対称 3x3 の closed-form で求めればよい。 反復ソルバ無し = GPU スレッドで決定的。

// ------------------------------------------------------------
// PointHashGrid: source 点を一辺 cell_size のセルに割り当てた CSR グリッド。
// 「セル → そのセルに属する点 index の連続範囲」 を保持し、 27 近傍セル探索で
// 候補点を集められるようにする。 CPU/GPU で完全に同一のハッシュ・floor を使う。
// ------------------------------------------------------------
struct PointHashGrid
{
  int capacity = 0;            // ハッシュスロット数 (2 の冪、 セル数の 2 倍以上)
  float cell_size_m = 0.5f;    // セル一辺 [m]
  int num_points = 0;

  // 点本体 (num_points * 3, fp32, body フレーム)。 GPU へそのまま転送する。
  std::vector<float> points_xyz;

  // 各スロット (capacity 個)。 occupied=1 のスロットだけ有効。
  std::vector<std::uint8_t> occupied;            // capacity
  std::vector<std::int32_t> cell_keys_xyz;       // capacity * 3 : 整数セル座標
  std::vector<std::int32_t> cell_point_offset;   // capacity     : point_indices 内開始
  std::vector<std::int32_t> cell_point_count;    // capacity     : そのセルの点数

  // セル順 (occupied スロット走査順) に並べ替えた元 index。 長さ num_points。
  std::vector<std::int32_t> point_indices;

  // CPU 側でグリッドを組むヘルパ。 points は body フレーム。
  static PointHashGrid build(
    float cell_size_m,
    const std::vector<Eigen::Vector3d> & points_body);
};

struct CovarianceEstimateConfig
{
  // 近傍点数 k。 最小 3 (平面 2 方向 + 法線 1 を分離するため)。
  int num_neighbors = 10;
  // 平面正則化の最小固有値 (法線方向の縮退度)。
  float plane_epsilon = 1e-3f;
  // 近傍セル探索半径 (セル単位)。 1 で 3x3x3 = 27 近傍。
  int search_radius_cells = 1;
  // グリッドのセル一辺 [m]。 これより近い近傍は同/隣接セルに入る前提なので、
  // 点群密度に対して「k 点が 27 近傍セルに収まる」 程度に取ること。
  float cell_size_m = 0.5f;
};

// 推定結果。 covariances は入力点と同じ並び・長さ。
struct CovarianceEstimation
{
  std::vector<Eigen::Matrix3d> covariances;
  bool gpu_used = false;          // GPU 経路を通ったか (false = Metal 無効 / デバイス無)
  std::string error_message;
};

// CPU 参照実装 (グリッド kNN + closed-form 法線 + 平面正則化)。 GPU 結果の
// 「正解」 として、 また Metal 非対応ビルドのフォールバックとして使う。
// 既存 computeSourceCovariances (PCL KdTree) とは近傍が違うのでビット一致しない。
std::vector<Eigen::Matrix3d> estimateSourceCovariancesGridCpu(
  const std::vector<Eigen::Vector3d> & points_body,
  const CovarianceEstimateConfig & config);

// Metal GPU 実装 (単発)。 PYLOT_LIO_HAS_METAL 無効 / デバイス無なら CPU 参照に
// フォールバックし gpu_used=false で返す。 内部で一時エンジンを生成する
// (PSO コンパイルを含むので単発用途向け)。
CovarianceEstimation estimateSourceCovariancesMetal(
  const std::vector<Eigen::Vector3d> & points_body,
  const CovarianceEstimateConfig & config);

// ============================================================
// MetalCovarianceEngine: device/PSO を永続化して毎フレームの再コンパイルを避ける。
// metal-cpp 型はヘッダに出さず PImpl で隠蔽する。
// ============================================================
class MetalCovarianceEngine
{
public:
  MetalCovarianceEngine();
  ~MetalCovarianceEngine();

  MetalCovarianceEngine(const MetalCovarianceEngine &) = delete;
  MetalCovarianceEngine & operator=(const MetalCovarianceEngine &) = delete;

  // device/PSO を構築できたか (= Metal 対応ビルド & GPU 有 & カーネルコンパイル成功)。
  bool isValid() const;

  // 1 フレーム 1 回: source 点群の共分散を推定する。 内部でグリッドを構築し
  // GPU にアップロード → カーネル実行 → 結果読み出しを行う。
  // isValid()==false のときは CPU 参照にフォールバックする。
  CovarianceEstimation estimate(
    const std::vector<Eigen::Vector3d> & points_body,
    const CovarianceEstimateConfig & config);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace pylot_lio::gpu

#endif  // PYLOT_LIO__GPU__METAL_COVARIANCE_ESTIMATOR_HPP_
