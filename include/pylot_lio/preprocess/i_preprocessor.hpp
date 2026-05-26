// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#ifndef PYLOT_LIO__PREPROCESS__I_PREPROCESSOR_HPP_
#define PYLOT_LIO__PREPROCESS__I_PREPROCESSOR_HPP_

#include <memory>
#include <string>

#include "pylot_lio/types.hpp"

namespace pylot_lio
{

// 1スキャン分の点群を「マッチングに使う代表点だけ」に間引く層の抽象。
// 同じ入力 ROS トピックに対し、複数の間引き手法 (Voxel グリッド / ランダム
// サンプリング) を差し替えて比較できるようにするためのインタフェース。
class IPreprocessor
{
public:
  virtual ~IPreprocessor() = default;

  // 入力点群を間引いて返す。返り値は入力とは別の PointCloud を保持する。
  // 入力の点数が極端に少ない場合は何もせず元の点群相当を返す。
  virtual PointCloudPtr process(const PointCloudConstPtr & input_cloud) = 0;

  // 設定値をログに出すための識別子 (例: "voxel_grid:0.30").
  virtual std::string describe() const = 0;
};

using IPreprocessorPtr = std::unique_ptr<IPreprocessor>;

}  // namespace pylot_lio

#endif  // PYLOT_LIO__PREPROCESS__I_PREPROCESSOR_HPP_
