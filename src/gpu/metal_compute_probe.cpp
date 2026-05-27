// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#include "pylot_lio/gpu/metal_compute_probe.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

// metal-cpp は宣言のみのヘッダオンリー配布で、 各ラッパー関数の「実体」は
// この 3 マクロを立てた *唯一の* 翻訳単位に生成される。 複数 TU で立てると
// 同じシンボルが重複定義してリンクエラーになるため、 ここ 1 箇所だけで定義する。
// (compute 専用なので QuartzCore/CA は不要。 Foundation + Metal のみ。)
//
// 重要: これらの #include は必ず *ファイルスコープ* (どの namespace の外) で行うこと。
// namespace pylot_lio::gpu の内側で include すると、 metal-cpp が引き込む <functional>
// などの標準ヘッダが pylot_lio::gpu::std として解釈され、 libc++ 内部が壊れる。
#ifdef PYLOT_LIO_HAS_METAL
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#endif

namespace pylot_lio::gpu
{

#ifndef PYLOT_LIO_HAS_METAL

// Metal 非対応ビルド (非 Apple、 または Metal を無効化したビルド) 向けスタブ。
// 例外は投げず、 metal_available=false で理由だけ返す。
MetalProbeResult runMetalSaxpyProbe(
  float /*a*/,
  const std::vector<float> & /*x*/,
  const std::vector<float> & /*y*/)
{
  MetalProbeResult result;
  result.metal_available = false;
  result.error_message = "Metal support not compiled in (PYLOT_LIO_HAS_METAL undefined)";
  return result;
}

#else  // PYLOT_LIO_HAS_METAL

namespace
{

// saxpy: out[i] = a * x[i] + y[i] を計算する最小 compute カーネル。
// GICP ロジックとは無関係で、 「Metal ツールチェーンが通るか」だけを確かめる土台。
const char * kSaxpyKernelSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

kernel void saxpy(
    device const float * x   [[buffer(0)]],
    device const float * y   [[buffer(1)]],
    device float       * out [[buffer(2)]],
    constant float     & a   [[buffer(3)]],
    uint                 index [[thread_position_in_grid]])
{
  out[index] = a * x[index] + y[index];
}
)METAL";

}  // namespace

MetalProbeResult runMetalSaxpyProbe(
  float a,
  const std::vector<float> & x,
  const std::vector<float> & y)
{
  MetalProbeResult result;

  if (x.size() != y.size()) {
    result.error_message = "x and y must have the same length";
    return result;
  }
  const std::size_t element_count = x.size();
  if (element_count == 0) {
    result.error_message = "input vectors are empty";
    return result;
  }

  // metal-cpp の autoreleased オブジェクト (newFunction 以外の string 等) を
  // スコープ末で確実に解放するためのプール。 明示 new した PSO/buffer 等は手動 release。
  NS::AutoreleasePool * autorelease_pool = NS::AutoreleasePool::alloc()->init();

  MTL::Device * device = MTL::CreateSystemDefaultDevice();
  if (device == nullptr) {
    result.error_message = "MTL::CreateSystemDefaultDevice returned null (no Metal GPU)";
    autorelease_pool->release();
    return result;
  }
  result.metal_available = true;
  result.device_name = device->name()->utf8String();

  // ---- カーネルの実行時コンパイル ----
  NS::Error * compile_error = nullptr;
  NS::String * kernel_source =
    NS::String::string(kSaxpyKernelSource, NS::UTF8StringEncoding);
  MTL::Library * library = device->newLibrary(kernel_source, nullptr, &compile_error);
  if (library == nullptr) {
    result.error_message = "newLibrary failed: ";
    if (compile_error != nullptr) {
      result.error_message += compile_error->localizedDescription()->utf8String();
    }
    device->release();
    autorelease_pool->release();
    return result;
  }

  NS::String * function_name = NS::String::string("saxpy", NS::UTF8StringEncoding);
  MTL::Function * saxpy_function = library->newFunction(function_name);
  if (saxpy_function == nullptr) {
    result.error_message = "newFunction('saxpy') returned null";
    library->release();
    device->release();
    autorelease_pool->release();
    return result;
  }

  NS::Error * pipeline_error = nullptr;
  MTL::ComputePipelineState * pipeline_state =
    device->newComputePipelineState(saxpy_function, &pipeline_error);
  if (pipeline_state == nullptr) {
    result.error_message = "newComputePipelineState failed: ";
    if (pipeline_error != nullptr) {
      result.error_message += pipeline_error->localizedDescription()->utf8String();
    }
    saxpy_function->release();
    library->release();
    device->release();
    autorelease_pool->release();
    return result;
  }
  result.kernel_compiled = true;

  MTL::CommandQueue * command_queue = device->newCommandQueue();

  // ---- unified memory バッファ (Apple Silicon では GPU/CPU でゼロコピー共有) ----
  const std::size_t bytes = element_count * sizeof(float);
  MTL::Buffer * buffer_x =
    device->newBuffer(x.data(), bytes, MTL::ResourceStorageModeShared);
  MTL::Buffer * buffer_y =
    device->newBuffer(y.data(), bytes, MTL::ResourceStorageModeShared);
  MTL::Buffer * buffer_out =
    device->newBuffer(bytes, MTL::ResourceStorageModeShared);
  MTL::Buffer * buffer_a =
    device->newBuffer(&a, sizeof(float), MTL::ResourceStorageModeShared);

  // ---- ディスパッチ ----
  MTL::CommandBuffer * command_buffer = command_queue->commandBuffer();
  MTL::ComputeCommandEncoder * encoder = command_buffer->computeCommandEncoder();
  encoder->setComputePipelineState(pipeline_state);
  encoder->setBuffer(buffer_x, 0, 0);
  encoder->setBuffer(buffer_y, 0, 1);
  encoder->setBuffer(buffer_out, 0, 2);
  encoder->setBuffer(buffer_a, 0, 3);

  // 1 次元グリッド: element_count スレッド。 threadgroup サイズは PSO の
  // 推奨上限を使い、 端数はグリッド側で吸収させる (dispatchThreads は非整数倍を許容)。
  const NS::UInteger threads_per_threadgroup =
    std::min<NS::UInteger>(
      pipeline_state->maxTotalThreadsPerThreadgroup(),
      static_cast<NS::UInteger>(element_count));
  const MTL::Size grid_size = MTL::Size::Make(element_count, 1, 1);
  const MTL::Size threadgroup_size = MTL::Size::Make(threads_per_threadgroup, 1, 1);
  encoder->dispatchThreads(grid_size, threadgroup_size);
  encoder->endEncoding();
  command_buffer->commit();
  command_buffer->waitUntilCompleted();

  // ---- 結果照合 (shared バッファなので read back はポインタ参照のみ) ----
  const float * gpu_output = static_cast<const float *>(buffer_out->contents());
  bool all_match = true;
  for (std::size_t i = 0; i < element_count; ++i) {
    const float expected = a * x[i] + y[i];
    // fp32 同士の比較。 相対 + 絶対の緩めの許容で「同じ計算をした」ことを確認する。
    const float diff = std::fabs(gpu_output[i] - expected);
    const float tolerance = 1e-4f * (1.0f + std::fabs(expected));
    if (diff > tolerance) {
      all_match = false;
      break;
    }
  }
  result.result_correct = all_match;
  if (!all_match) {
    result.error_message = "GPU saxpy result did not match CPU reference";
  }

  // ---- 明示 new したオブジェクトの解放 (生成と逆順) ----
  buffer_a->release();
  buffer_out->release();
  buffer_y->release();
  buffer_x->release();
  command_queue->release();
  pipeline_state->release();
  saxpy_function->release();
  library->release();
  device->release();
  autorelease_pool->release();

  return result;
}

#endif  // PYLOT_LIO_HAS_METAL

}  // namespace pylot_lio::gpu
