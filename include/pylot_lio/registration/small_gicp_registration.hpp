// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#ifndef PYLOT_LIO__REGISTRATION__SMALL_GICP_REGISTRATION_HPP_
#define PYLOT_LIO__REGISTRATION__SMALL_GICP_REGISTRATION_HPP_

#ifdef PYLOT_LIO_HAS_SMALL_GICP

#include <memory>
#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "pylot_lio/registration/i_registration.hpp"

namespace pylot_lio
{

// small_gicp の VGICP/GICP を IRegistration インタフェースに被せた薄いラッパ。
// マップ実装には依存せず、毎回 IPointCloudMap::toPointCloud() から target を作る。
// (target が毎フレーム変わるため target キャッシュは行わない; 純粋比較のための単純実装。)
class SmallGicpRegistration : public IRegistration
{
public:
  enum class Variant
  {
    GICP,
    VGICP,
  };

  struct Config
  {
    Variant variant = Variant::VGICP;
    double map_voxel_resolution_m = 0.5;
    double max_correspondence_distance_m = 2.0;
    int num_threads = 4;
    int max_iterations = 30;
  };

  explicit SmallGicpRegistration(const Config & config);
  ~SmallGicpRegistration() override;

  AlignResult align(
    const PointCloud & source_cloud_body,
    const IPointCloudMap & map_world,
    const Eigen::Isometry3d & initial_transform_world_body) override;

  std::string describe() const override;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  Config config_;
};

}  // namespace pylot_lio

#endif  // PYLOT_LIO_HAS_SMALL_GICP
#endif  // PYLOT_LIO__REGISTRATION__SMALL_GICP_REGISTRATION_HPP_
