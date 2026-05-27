// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
//
// Metal GPU バックエンドの「土台」検証。 GICP ロジックとは切り離して、
//   - metal-cpp の vendoring とリンク
//   - カーネルの実行時コンパイル
//   - unified memory バッファの往復
// が成立することだけを saxpy で確認する。
//
// PYLOT_LIO_HAS_METAL が定義されたビルド (macOS + Metal) では GPU 実行と
// 数値一致を要求する。 非対応ビルドでは「スタブが安全に false を返す」ことだけ確認する。

#include <gtest/gtest.h>

#include <vector>

#include "pylot_lio/gpu/metal_compute_probe.hpp"

namespace pylot_lio::gpu
{

TEST(MetalComputeProbe, MismatchedInputLengthsReturnError)
{
  // 長さ不一致は Metal の有無に関わらずエラー扱い (例外は投げない)。
  const auto result = runMetalSaxpyProbe(2.0f, {1.0f, 2.0f}, {1.0f});
  EXPECT_FALSE(result.result_correct);
  EXPECT_FALSE(result.error_message.empty());
}

#ifdef PYLOT_LIO_HAS_METAL

TEST(MetalComputeProbe, SaxpyMatchesCpuReference)
{
  // out[i] = a * x[i] + y[i] を GPU で計算し、 CPU 基準と一致することを確認。
  const float a = 3.5f;
  std::vector<float> x;
  std::vector<float> y;
  const int element_count = 4096;
  x.reserve(element_count);
  y.reserve(element_count);
  for (int i = 0; i < element_count; ++i) {
    x.push_back(static_cast<float>(i) * 0.5f);
    y.push_back(static_cast<float>(i) * -0.25f + 1.0f);
  }

  const auto result = runMetalSaxpyProbe(a, x, y);

  ASSERT_TRUE(result.metal_available)
    << "Metal device unavailable on a PYLOT_LIO_HAS_METAL build: " << result.error_message;
  EXPECT_TRUE(result.kernel_compiled) << result.error_message;
  EXPECT_TRUE(result.result_correct) << result.error_message;
  EXPECT_FALSE(result.device_name.empty());
}

#else  // PYLOT_LIO_HAS_METAL

TEST(MetalComputeProbe, StubReportsUnavailableWithoutCrashing)
{
  // 非 Metal ビルドでは metal_available=false で理由を返すだけ (クラッシュしない)。
  const auto result = runMetalSaxpyProbe(2.0f, {1.0f, 2.0f, 3.0f}, {0.0f, 0.0f, 0.0f});
  EXPECT_FALSE(result.metal_available);
  EXPECT_FALSE(result.error_message.empty());
}

#endif  // PYLOT_LIO_HAS_METAL

}  // namespace pylot_lio::gpu

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
