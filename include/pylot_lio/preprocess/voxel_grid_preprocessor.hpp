// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#ifndef PYLOT_LIO__PREPROCESS__VOXEL_GRID_PREPROCESSOR_HPP_
#define PYLOT_LIO__PREPROCESS__VOXEL_GRID_PREPROCESSOR_HPP_

#include <string>

#include "pylot_lio/preprocess/i_preprocessor.hpp"

namespace pylot_lio
{

// 一辺 voxel_size_m の立方体ボクセルに空間を区切り、各ボクセル内の点群を
// その重心 1 点で代表させる古典的なダウンサンプリング。
// 出力の点密度がほぼ均一になるためマッチングが安定するが、密集部の細部は失われる。
class VoxelGridPreprocessor : public IPreprocessor
{
public:
  explicit VoxelGridPreprocessor(double voxel_size_m);

  PointCloudPtr process(const PointCloudConstPtr & input_cloud) override;
  std::string describe() const override;

private:
  double voxel_size_m_;
};

}  // namespace pylot_lio

#endif  // PYLOT_LIO__PREPROCESS__VOXEL_GRID_PREPROCESSOR_HPP_
