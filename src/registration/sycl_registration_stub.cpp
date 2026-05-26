// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
//
// SYCL コンパイラ (Intel oneAPI DPC++ / IntelLLVM) が見つからない環境で
// SyclRegistration のシンボルを満たすためのスタブ。コンストラクト時に throw する。
#include "pylot_lio/registration/sycl_registration.hpp"

#include <stdexcept>

namespace pylot_lio
{

SyclRegistration::SyclRegistration(const Config & config)
: config_(config)
{
  throw std::runtime_error(
    "SyclRegistration: this build was produced without a SYCL-capable compiler. "
    "Use plain_gicp / small_gicp_vgicp instead, or rebuild with Intel oneAPI (icpx).");
}

SyclRegistration::~SyclRegistration() = default;

IRegistration::AlignResult SyclRegistration::align(
  const PointCloud & /*source_cloud_body*/,
  const IPointCloudMap & /*map_world*/,
  const Eigen::Isometry3d & initial_transform_world_body)
{
  AlignResult result;
  result.transform_world_body = initial_transform_world_body;
  return result;
}

std::string SyclRegistration::describe() const
{
  return "sycl_stub(unavailable)";
}

}  // namespace pylot_lio
