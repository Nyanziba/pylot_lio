// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#ifndef PYLOT_LIO__GPU__METAL_COMPUTE_PROBE_HPP_
#define PYLOT_LIO__GPU__METAL_COMPUTE_PROBE_HPP_

#include <string>
#include <vector>

namespace pylot_lio::gpu
{

// Metal GPU バックエンドの「土台」検証用プローブの結果。
// 本格的な GICP の GPU 化に進む前に、 metal-cpp の vendoring・framework リンク・
// カーネルの実行時コンパイル・unified memory バッファの往復が colcon ビルドで
// 通ることを切り分けて確認するために使う。
struct MetalProbeResult
{
  bool metal_available = false;   // MTL::Device を取得できたか (= Metal 対応ビルド & GPU 有)
  bool kernel_compiled = false;   // saxpy カーネルを実行時コンパイルできたか
  bool result_correct = false;    // GPU 計算結果が CPU 計算と一致したか
  std::string device_name;        // GPU デバイス名 (診断用)
  std::string error_message;      // 失敗時の理由 (成功時は空)
};

// out[i] = a * x[i] + y[i] を Metal GPU 上で計算し、 CPU 計算と照合する。
// x と y は同じ長さでなければならない (異なる場合は error_message を立てて返す)。
//
// PYLOT_LIO_HAS_METAL が未定義のビルド (非 Apple など) では、 metal_available=false・
// error_message="Metal support not compiled in" を即座に返す (例外は投げない)。
MetalProbeResult runMetalSaxpyProbe(
  float a,
  const std::vector<float> & x,
  const std::vector<float> & y);

}  // namespace pylot_lio::gpu

#endif  // PYLOT_LIO__GPU__METAL_COMPUTE_PROBE_HPP_
