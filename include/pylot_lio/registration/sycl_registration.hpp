// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#ifndef PYLOT_LIO__REGISTRATION__SYCL_REGISTRATION_HPP_
#define PYLOT_LIO__REGISTRATION__SYCL_REGISTRATION_HPP_

#include <string>

#include "pylot_lio/registration/i_registration.hpp"

namespace pylot_lio
{

// SYCL (Intel oneAPI DPC++ / spir64) で GICP を実装するバックエンド。
// 本実装は Linux + IntelLLVM コンパイラのときだけ有効化される。
// macOS / stock clang ではビルド時に sycl_registration_stub.cpp が代わりにリンクされ、
// このクラスのインスタンス化は throw する。S4 のターゲット。
class SyclRegistration : public IRegistration
{
public:
  struct Config
  {
    int max_iterations = 30;
    double max_correspondence_distance_m = 2.0;
  };

  explicit SyclRegistration(const Config & config);
  ~SyclRegistration() override;

  AlignResult align(
    const PointCloud & source_cloud_body,
    const IPointCloudMap & map_world,
    const Eigen::Isometry3d & initial_transform_world_body) override;

  std::string describe() const override;

private:
  Config config_;
};

}  // namespace pylot_lio

#endif  // PYLOT_LIO__REGISTRATION__SYCL_REGISTRATION_HPP_
