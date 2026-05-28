// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#ifndef PYLOT_LIO__PREPROCESS__VOXEL_RANDOM_SAMPLING_PREPROCESSOR_HPP_
#define PYLOT_LIO__PREPROCESS__VOXEL_RANDOM_SAMPLING_PREPROCESSOR_HPP_

#include <cstdint>
#include <memory>
#include <random>
#include <string>

#include "metal_gpu_kernels/metal_voxel_downsampler.hpp"
#include "pylot_lio/preprocess/i_preprocessor.hpp"

namespace pylot_lio
{

// 点を voxel グリッドに振り分け、 各 voxel から「実点」を率に応じてランダムに選ぶ前処理。
//
// voxel_grid (重心ベース) との違い:
//   - 重心は voxel 内の点を平均化するため、 ノイズ点と実点が混ざる voxel で
//     「実際には点が存在しない位置」に代表点を作ってしまう。
//   - voxel_random_sampling は「元々あった点」を保つので、 ノイズ点に引きずられた
//     平均値を作らず ICP/法線推定が安定しやすい。
//
// サンプリング率 (GLIM の randomgrid downsampling 相当):
//   各 voxel から m = max(1, round(voxel 内点数 × sampling_rate)) 点を保持する。
//   sampling_rate=0 でも各 voxel 最低 1 点 (= 従来の「voxel 1 点」 と一致、 後方互換)。
//   1.0 で全点保持。 空間的に均一に間引きつつ、 密集 voxel ほど多く残せる。
//
// GPU 高速化:
//   use_gpu=true なら間引きを Apple Metal (MetalVoxelDownsampler) で行う。 整数演算
//   (reservoir sampling + xorshift32) なので CPU 参照と完全一致する。 GPU 無効ビルド /
//   デバイス無では同じグリッド方式を CPU で実行する。
class VoxelRandomSamplingPreprocessor : public IPreprocessor
{
public:
  struct Config
  {
    double voxel_size_m = 0.3;
    // 同じ seed で起動すると再現性のある間引きになる。 0 はマシン乱数で初期化。
    uint32_t random_seed = 12345u;
    // 0〜1。 各 voxel から round(点数 × rate) 点 (最低 1) を保持。 0 で従来の voxel 1 点。
    double sampling_rate = 0.0;
    // 間引きを GPU (Metal) で行うか。 GPU 無効ビルド / デバイス無では CPU に自動フォールバック。
    bool use_gpu = true;
  };

  explicit VoxelRandomSamplingPreprocessor(const Config & config);

  PointCloudPtr process(const PointCloudConstPtr & input_cloud) override;

  std::string describe() const override;

private:
  Config config_;
  std::mt19937 random_engine_;
  // GPU リソース (device/PSO) を process 間で使い回す永続エンジン。 use_gpu=false や
  // Metal 無効ビルドでは nullptr / isValid()=false。
  std::shared_ptr<metal_gpu_kernels::MetalVoxelDownsampler> downsampler_;
};

}  // namespace pylot_lio

#endif  // PYLOT_LIO__PREPROCESS__VOXEL_RANDOM_SAMPLING_PREPROCESSOR_HPP_
