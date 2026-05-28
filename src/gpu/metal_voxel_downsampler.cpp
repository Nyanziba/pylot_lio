// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#include "pylot_lio/gpu/metal_voxel_downsampler.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

#include "pylot_lio/gpu/metal_covariance_estimator.hpp"  // PointHashGrid を再利用

// metal-cpp の実装マクロは metal_compute_probe.cpp の唯一の TU で定義済み。 ここでは
// 宣言だけ取り込む (必ずファイルスコープで include)。
#ifdef PYLOT_LIO_HAS_METAL
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#endif

namespace pylot_lio::gpu
{

namespace
{

// 1 ボクセルから保持する点の上限。 GPU カーネルの thread-local 配列サイズと一致させる。
// これを超える m はクランプする (CPU/GPU 同一に)。
constexpr int kMaxKeepPerVoxel = 256;

// CPU/GPU で完全に同一でなければならない xorshift32 乱数。
inline std::uint32_t xorshift32(std::uint32_t & state)
{
  std::uint32_t x = state;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  state = x;
  return x;
}

// スロットごとの乱数初期 state。 seed と slot から決定的に作る (0 を避ける)。
inline std::uint32_t slotSeed(std::uint32_t seed, std::uint32_t slot)
{
  const std::uint32_t mixed = seed * 747796405u + (slot + 1u) * 2891336453u;
  return mixed == 0u ? 1u : mixed;
}

// ボクセル内点数 count と rate から保持数 m を決める (CPU/GPU 同一)。
inline int keepCount(int count, float sampling_rate)
{
  const float rate = std::min(1.0f, std::max(0.0f, sampling_rate));
  int m = static_cast<int>(std::floor(static_cast<float>(count) * rate + 0.5f));
  if (m < 1) {
    m = 1;  // 各ボクセル最低 1 点。
  }
  if (m > count) {
    m = count;
  }
  if (m > kMaxKeepPerVoxel) {
    m = kMaxKeepPerVoxel;
  }
  return m;
}

}  // namespace

std::vector<std::int32_t> voxelRandomDownsampleGridCpu(
  const std::vector<Eigen::Vector3d> & points,
  const VoxelDownsampleConfig & config)
{
  std::vector<std::int32_t> selected;
  if (points.empty()) {
    return selected;
  }
  const PointHashGrid grid = PointHashGrid::build(config.voxel_size_m, points);

  // 各スロットの保持数 m と、 exclusive prefix sum による出力オフセットを先に確定する。
  std::vector<int> keep_per_slot(grid.capacity, 0);
  std::vector<int> out_offset(grid.capacity, 0);
  int total = 0;
  for (int slot = 0; slot < grid.capacity; ++slot) {
    if (grid.occupied[slot] == 0) {
      continue;
    }
    const int m = keepCount(grid.cell_point_count[slot], config.sampling_rate);
    keep_per_slot[slot] = m;
    out_offset[slot] = total;
    total += m;
  }
  selected.resize(total);

  // 各スロットで reservoir sampling (Algorithm R) し、 自分の区間へ書き込む。
  for (int slot = 0; slot < grid.capacity; ++slot) {
    if (grid.occupied[slot] == 0) {
      continue;
    }
    const int count = grid.cell_point_count[slot];
    const int in_offset = grid.cell_point_offset[slot];
    const int m = keep_per_slot[slot];
    const int out = out_offset[slot];

    std::array<int, kMaxKeepPerVoxel> keep;  // local index 0..count-1 のうち保持する m 個
    std::uint32_t state = slotSeed(config.random_seed, static_cast<std::uint32_t>(slot));
    for (int i = 0; i < count; ++i) {
      if (i < m) {
        keep[i] = i;
      } else {
        const int j = static_cast<int>(xorshift32(state) % static_cast<std::uint32_t>(i + 1));
        if (j < m) {
          keep[j] = i;
        }
      }
    }
    for (int t = 0; t < m; ++t) {
      selected[out + t] = grid.point_indices[in_offset + keep[t]];
    }
  }
  return selected;
}

#ifndef PYLOT_LIO_HAS_METAL

VoxelDownsampleResult voxelRandomDownsampleMetal(
  const std::vector<Eigen::Vector3d> & points,
  const VoxelDownsampleConfig & config)
{
  VoxelDownsampleResult result;
  result.selected_indices = voxelRandomDownsampleGridCpu(points, config);
  result.gpu_used = false;
  result.error_message = "Metal support not compiled in (PYLOT_LIO_HAS_METAL undefined)";
  return result;
}

struct MetalVoxelDownsampler::Impl {};
MetalVoxelDownsampler::MetalVoxelDownsampler() : impl_(nullptr) {}
MetalVoxelDownsampler::~MetalVoxelDownsampler() = default;
bool MetalVoxelDownsampler::isValid() const { return false; }
VoxelDownsampleResult MetalVoxelDownsampler::downsample(
  const std::vector<Eigen::Vector3d> & points,
  const VoxelDownsampleConfig & config)
{
  VoxelDownsampleResult result;
  result.selected_indices = voxelRandomDownsampleGridCpu(points, config);
  result.gpu_used = false;
  result.error_message = "Metal support not compiled in (PYLOT_LIO_HAS_METAL undefined)";
  return result;
}

#else  // PYLOT_LIO_HAS_METAL

namespace
{

struct DownsampleKernelParams
{
  std::uint32_t random_seed;
  std::uint32_t capacity;
  std::int32_t padding0;
  std::int32_t padding1;
};

constexpr int kThreadgroupSize = 64;

const char * kDownsampleKernelSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

constant int kMaxKeep = 256;

struct DownsampleKernelParams {
  uint random_seed;
  uint capacity;
  int  padding0;
  int  padding1;
};

inline uint xorshift32(thread uint & state) {
  uint x = state;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  state = x;
  return x;
}

inline uint slot_seed(uint seed, uint slot) {
  uint mixed = seed * 747796405u + (slot + 1u) * 2891336453u;
  return mixed == 0u ? 1u : mixed;
}

// 1 thread = 1 ボクセルスロット。 reservoir sampling で m 点を選び自分の区間へ書く。
kernel void voxel_downsample(
    device const uchar * occupied      [[buffer(0)]],  // capacity
    device const int   * cell_offset   [[buffer(1)]],  // capacity (in_offset)
    device const int   * cell_count    [[buffer(2)]],  // capacity
    device const int   * point_indices [[buffer(3)]],  // num_points
    device const int   * keep_per_slot [[buffer(4)]],  // capacity (m、 CPU で確定済み)
    device const int   * out_offset    [[buffer(5)]],  // capacity (prefix sum)
    constant DownsampleKernelParams & params [[buffer(6)]],
    device int         * selected      [[buffer(7)]],  // total
    uint gid [[thread_position_in_grid]])
{
  if (gid >= params.capacity) { return; }
  if (occupied[gid] == 0) { return; }

  int count = cell_count[gid];
  int in_offset = cell_offset[gid];
  int m = keep_per_slot[gid];
  int out = out_offset[gid];
  if (m > kMaxKeep) { m = kMaxKeep; }

  int keep[kMaxKeep];
  uint state = slot_seed(params.random_seed, gid);
  for (int i = 0; i < count; ++i) {
    if (i < m) {
      keep[i] = i;
    } else {
      int j = int(xorshift32(state) % uint(i + 1));
      if (j < m) { keep[j] = i; }
    }
  }
  for (int t = 0; t < m; ++t) {
    selected[out + t] = point_indices[in_offset + keep[t]];
  }
}
)METAL";

}  // namespace

struct MetalVoxelDownsampler::Impl
{
  MTL::Device * device = nullptr;
  MTL::CommandQueue * queue = nullptr;
  MTL::Library * library = nullptr;
  MTL::Function * function = nullptr;
  MTL::ComputePipelineState * pso = nullptr;
  MTL::Buffer * buf_params = nullptr;

  static void releaseBuffer(MTL::Buffer *& buffer)
  {
    if (buffer != nullptr) {
      buffer->release();
      buffer = nullptr;
    }
  }

  ~Impl()
  {
    releaseBuffer(buf_params);
    if (pso != nullptr) { pso->release(); }
    if (function != nullptr) { function->release(); }
    if (library != nullptr) { library->release(); }
    if (queue != nullptr) { queue->release(); }
    if (device != nullptr) { device->release(); }
  }
};

MetalVoxelDownsampler::MetalVoxelDownsampler() : impl_(std::make_unique<Impl>())
{
  NS::AutoreleasePool * pool = NS::AutoreleasePool::alloc()->init();
  impl_->device = MTL::CreateSystemDefaultDevice();
  if (impl_->device != nullptr) {
    NS::Error * compile_error = nullptr;
    NS::String * source = NS::String::string(kDownsampleKernelSource, NS::UTF8StringEncoding);
    impl_->library = impl_->device->newLibrary(source, nullptr, &compile_error);
    if (impl_->library != nullptr) {
      impl_->function = impl_->library->newFunction(
        NS::String::string("voxel_downsample", NS::UTF8StringEncoding));
      NS::Error * pso_error = nullptr;
      impl_->pso = impl_->device->newComputePipelineState(impl_->function, &pso_error);
      impl_->queue = impl_->device->newCommandQueue();
      impl_->buf_params = impl_->device->newBuffer(
        sizeof(DownsampleKernelParams), MTL::ResourceStorageModeShared);
    }
  }
  pool->release();
}

MetalVoxelDownsampler::~MetalVoxelDownsampler() = default;

bool MetalVoxelDownsampler::isValid() const
{
  return impl_ && impl_->device != nullptr && impl_->pso != nullptr &&
    impl_->queue != nullptr && impl_->buf_params != nullptr;
}

VoxelDownsampleResult MetalVoxelDownsampler::downsample(
  const std::vector<Eigen::Vector3d> & points,
  const VoxelDownsampleConfig & config)
{
  VoxelDownsampleResult result;
  if (!isValid() || points.empty()) {
    result.selected_indices = voxelRandomDownsampleGridCpu(points, config);
    result.gpu_used = false;
    if (!isValid()) {
      result.error_message = "engine not valid (no Metal device / PSO); CPU fallback";
    }
    return result;
  }

  NS::AutoreleasePool * pool = NS::AutoreleasePool::alloc()->init();

  const PointHashGrid grid = PointHashGrid::build(config.voxel_size_m, points);

  // 保持数 m と出力オフセット (exclusive prefix sum) を CPU で確定する。
  std::vector<std::int32_t> keep_per_slot(grid.capacity, 0);
  std::vector<std::int32_t> out_offset(grid.capacity, 0);
  int total = 0;
  for (int slot = 0; slot < grid.capacity; ++slot) {
    if (grid.occupied[slot] == 0) {
      continue;
    }
    const int m = keepCount(grid.cell_point_count[slot], config.sampling_rate);
    keep_per_slot[slot] = m;
    out_offset[slot] = total;
    total += m;
  }
  result.selected_indices.assign(total, 0);
  if (total == 0) {
    result.gpu_used = true;
    pool->release();
    return result;
  }

  auto make_buffer = [&](const void * data, std::size_t bytes) {
    return impl_->device->newBuffer(data, bytes, MTL::ResourceStorageModeShared);
  };
  MTL::Buffer * buf_occupied = make_buffer(grid.occupied.data(), grid.occupied.size());
  MTL::Buffer * buf_offset =
    make_buffer(grid.cell_point_offset.data(), grid.cell_point_offset.size() * sizeof(std::int32_t));
  MTL::Buffer * buf_count =
    make_buffer(grid.cell_point_count.data(), grid.cell_point_count.size() * sizeof(std::int32_t));
  MTL::Buffer * buf_pidx =
    make_buffer(grid.point_indices.data(), grid.point_indices.size() * sizeof(std::int32_t));
  MTL::Buffer * buf_keep =
    make_buffer(keep_per_slot.data(), keep_per_slot.size() * sizeof(std::int32_t));
  MTL::Buffer * buf_outoff =
    make_buffer(out_offset.data(), out_offset.size() * sizeof(std::int32_t));
  MTL::Buffer * buf_selected = impl_->device->newBuffer(
    static_cast<std::size_t>(total) * sizeof(std::int32_t), MTL::ResourceStorageModeShared);

  DownsampleKernelParams * params =
    static_cast<DownsampleKernelParams *>(impl_->buf_params->contents());
  params->random_seed = config.random_seed;
  params->capacity = static_cast<std::uint32_t>(grid.capacity);
  params->padding0 = 0;
  params->padding1 = 0;

  MTL::CommandBuffer * command_buffer = impl_->queue->commandBuffer();
  MTL::ComputeCommandEncoder * encoder = command_buffer->computeCommandEncoder();
  encoder->setComputePipelineState(impl_->pso);
  encoder->setBuffer(buf_occupied, 0, 0);
  encoder->setBuffer(buf_offset, 0, 1);
  encoder->setBuffer(buf_count, 0, 2);
  encoder->setBuffer(buf_pidx, 0, 3);
  encoder->setBuffer(buf_keep, 0, 4);
  encoder->setBuffer(buf_outoff, 0, 5);
  encoder->setBuffer(impl_->buf_params, 0, 6);
  encoder->setBuffer(buf_selected, 0, 7);
  const int num_groups = (grid.capacity + kThreadgroupSize - 1) / kThreadgroupSize;
  const MTL::Size grid_size = MTL::Size::Make(static_cast<NS::UInteger>(num_groups), 1, 1);
  const MTL::Size tg_size = MTL::Size::Make(kThreadgroupSize, 1, 1);
  encoder->dispatchThreadgroups(grid_size, tg_size);
  encoder->endEncoding();
  command_buffer->commit();
  command_buffer->waitUntilCompleted();

  const std::int32_t * out = static_cast<const std::int32_t *>(buf_selected->contents());
  std::copy(out, out + total, result.selected_indices.begin());
  result.gpu_used = true;

  Impl::releaseBuffer(buf_occupied);
  Impl::releaseBuffer(buf_offset);
  Impl::releaseBuffer(buf_count);
  Impl::releaseBuffer(buf_pidx);
  Impl::releaseBuffer(buf_keep);
  Impl::releaseBuffer(buf_outoff);
  Impl::releaseBuffer(buf_selected);
  pool->release();
  return result;
}

VoxelDownsampleResult voxelRandomDownsampleMetal(
  const std::vector<Eigen::Vector3d> & points,
  const VoxelDownsampleConfig & config)
{
  // 単発用途: 一時エンジンを作って 1 回だけ走らせる (PSO コンパイルを毎回含む)。
  // 毎フレーム使うなら MetalVoxelDownsampler を保持して使い回すこと。
  MetalVoxelDownsampler downsampler;
  return downsampler.downsample(points, config);
}

#endif  // PYLOT_LIO_HAS_METAL

}  // namespace pylot_lio::gpu
