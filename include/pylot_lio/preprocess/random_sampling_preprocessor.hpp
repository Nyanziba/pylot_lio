// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#ifndef PYLOT_LIO__PREPROCESS__RANDOM_SAMPLING_PREPROCESSOR_HPP_
#define PYLOT_LIO__PREPROCESS__RANDOM_SAMPLING_PREPROCESSOR_HPP_

#include <cstddef>
#include <cstdint>
#include <random>
#include <string>

#include "pylot_lio/preprocess/i_preprocessor.hpp"

namespace pylot_lio
{

// 入力点群から目標点数 N をランダムに抽出するダウンサンプラ。
// 同じシードで再現性があり、空間バイアスが Voxel グリッドと違って入りにくい。
// ただしマップ縁辺の希薄な領域では Voxel に比べて点が足りなくなりがち。
class RandomSamplingPreprocessor : public IPreprocessor
{
public:
  RandomSamplingPreprocessor(std::size_t target_point_count, uint32_t random_seed);

  PointCloudPtr process(const PointCloudConstPtr & input_cloud) override;
  std::string describe() const override;

private:
  std::size_t target_point_count_;
  std::mt19937 random_engine_;
};

}  // namespace pylot_lio

#endif  // PYLOT_LIO__PREPROCESS__RANDOM_SAMPLING_PREPROCESSOR_HPP_
