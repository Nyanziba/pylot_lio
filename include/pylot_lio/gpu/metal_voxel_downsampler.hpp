// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#ifndef PYLOT_LIO__GPU__METAL_VOXEL_DOWNSAMPLER_HPP_
#define PYLOT_LIO__GPU__METAL_VOXEL_DOWNSAMPLER_HPP_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace pylot_lio::gpu
{

// ============================================================
// Metal voxel random downsampler (前処理 GPU 化、 GLIM の randomgrid downsampling 相当)
// ============================================================
// 生点群を一辺 voxel_size のボクセルに振り分け、 各ボクセルから
//   m = max(1, round(ボクセル内点数 × sampling_rate))
// 点をランダムに保持する (= ボクセル内率保持)。 sampling_rate=0 でも各ボクセル最低 1 点
// を残すので、 従来の「各ボクセル 1 点」 と一致する (後方互換)。 1.0 で全点保持。
//
// GPU 化の肝:
//   - 出力点数は事前に分からないが、 各ボクセルの保持数 m はグリッド構築直後に CPU で
//     求まる。 m の exclusive prefix sum で各ボクセルの「出力開始位置」 を確定させ、
//     各 GPU スレッド (= 1 ボクセル) が自分の区間に衝突なく書き込む。
//   - ボクセル内の m 点抽出は reservoir sampling (Algorithm R)。 乱数は seed + スロット
//     から決定的に生成する xorshift32。 整数演算のみなので CPU 参照と GPU は完全一致する
//     (共分散推定の fp32 と違いビット一致を検証できる)。

struct VoxelDownsampleConfig
{
  float voxel_size_m = 0.3f;
  // 0〜1。 各ボクセルから round(点数 × rate) 点 (最低 1) を保持する。
  // 0: 各ボクセル 1 点 (= 従来動作)。 1: 全点保持。
  float sampling_rate = 0.0f;
  // 同じ seed で再現性のある間引きになる。 0 はマシン乱数 (呼び出し側で解決して渡す)。
  std::uint32_t random_seed = 12345u;
};

struct VoxelDownsampleResult
{
  // 入力点群に対する「保持する点の index」。 呼び出し側はこの index で元の点
  // (intensity 等 全フィールド) をコピーできる。 順序は不定 (ボクセル走査順)。
  std::vector<std::int32_t> selected_indices;
  bool gpu_used = false;          // GPU 経路を通ったか (false = Metal 無効 / デバイス無)
  std::string error_message;
};

// CPU 参照実装 (グリッド + reservoir sampling)。 GPU と同一の xorshift32 / Algorithm R を
// 使うので、 同じ入力・seed なら GPU と selected_indices が完全一致する。
// 非有限 (NaN/Inf) 点はどのボクセルにも属さず、 選択されない。
std::vector<std::int32_t> voxelRandomDownsampleGridCpu(
  const std::vector<Eigen::Vector3d> & points,
  const VoxelDownsampleConfig & config);

// Metal GPU 実装 (単発)。 PYLOT_LIO_HAS_METAL 無効 / デバイス無なら CPU 参照に
// フォールバックし gpu_used=false で返す。 内部で一時エンジンを生成する。
VoxelDownsampleResult voxelRandomDownsampleMetal(
  const std::vector<Eigen::Vector3d> & points,
  const VoxelDownsampleConfig & config);

// ============================================================
// MetalVoxelDownsampler: device/PSO を永続化して毎フレームの再コンパイルを避ける。
// metal-cpp 型はヘッダに出さず PImpl で隠蔽する。
// ============================================================
class MetalVoxelDownsampler
{
public:
  MetalVoxelDownsampler();
  ~MetalVoxelDownsampler();

  MetalVoxelDownsampler(const MetalVoxelDownsampler &) = delete;
  MetalVoxelDownsampler & operator=(const MetalVoxelDownsampler &) = delete;

  bool isValid() const;

  // 1 フレーム 1 回: 点群を間引いて「保持する入力 index」を返す。
  // isValid()==false のときは CPU 参照にフォールバックする。
  VoxelDownsampleResult downsample(
    const std::vector<Eigen::Vector3d> & points,
    const VoxelDownsampleConfig & config);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace pylot_lio::gpu

#endif  // PYLOT_LIO__GPU__METAL_VOXEL_DOWNSAMPLER_HPP_
