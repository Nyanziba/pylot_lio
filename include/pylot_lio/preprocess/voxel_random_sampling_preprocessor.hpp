// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#ifndef PYLOT_LIO__PREPROCESS__VOXEL_RANDOM_SAMPLING_PREPROCESSOR_HPP_
#define PYLOT_LIO__PREPROCESS__VOXEL_RANDOM_SAMPLING_PREPROCESSOR_HPP_

#include <cstdint>
#include <random>
#include <string>

#include "pylot_lio/preprocess/i_preprocessor.hpp"

namespace pylot_lio
{

// 点を voxel グリッドに振り分け、 各 voxel から「実点 1 点」をランダムに選ぶ前処理。
//
// voxel_grid (重心ベース) との違い:
//   - 重心は voxel 内の点を平均化するため、 ノイズ点と実点が混ざる voxel で
//     「実際には点が存在しない位置」に代表点を作ってしまう。
//   - voxel_random_sampling は「元々あった点」のうち 1 つを保つので、
//     ノイズ点に引きずられた平均値を作らず ICP/法線推定が安定しやすい。
//
// 出力点数 ≈ シーン内の voxel 数 (= シーンの占有体積 / voxel_size^3 のうち点が落ちた数)。
// random_sampling (目標点数を一様サンプル) との違いは「空間が均一に間引かれる」点。
class VoxelRandomSamplingPreprocessor : public IPreprocessor
{
public:
  struct Config
  {
    double voxel_size_m = 0.3;
    // 同じ seed で起動すると再現性のある間引きになる。 0 はマシン乱数で初期化。
    uint32_t random_seed = 12345u;
  };

  explicit VoxelRandomSamplingPreprocessor(const Config & config);

  PointCloudPtr process(const PointCloudConstPtr & input_cloud) override;

  std::string describe() const override;

private:
  Config config_;
  std::mt19937 random_engine_;
};

}  // namespace pylot_lio

#endif  // PYLOT_LIO__PREPROCESS__VOXEL_RANDOM_SAMPLING_PREPROCESSOR_HPP_
