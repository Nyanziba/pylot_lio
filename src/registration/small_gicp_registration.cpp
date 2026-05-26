// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#ifdef PYLOT_LIO_HAS_SMALL_GICP

#include "pylot_lio/registration/small_gicp_registration.hpp"

#include <memory>
#include <sstream>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <small_gicp/pcl/pcl_registration.hpp>

namespace pylot_lio
{

struct SmallGicpRegistration::Impl
{
  std::unique_ptr<small_gicp::RegistrationPCL<pcl::PointXYZ, pcl::PointXYZ>> engine;
};

SmallGicpRegistration::SmallGicpRegistration(const Config & config)
: impl_(std::make_unique<Impl>()),
  config_(config)
{
  impl_->engine = std::make_unique<
    small_gicp::RegistrationPCL<pcl::PointXYZ, pcl::PointXYZ>>();
  if (config_.variant == Variant::VGICP) {
    impl_->engine->setRegistrationType("VGICP");
  } else {
    impl_->engine->setRegistrationType("GICP");
  }
  impl_->engine->setNumThreads(config_.num_threads);
  impl_->engine->setVoxelResolution(config_.map_voxel_resolution_m);
  impl_->engine->setMaxCorrespondenceDistance(config_.max_correspondence_distance_m);
  impl_->engine->setMaximumIterations(config_.max_iterations);
}

SmallGicpRegistration::~SmallGicpRegistration() = default;

IRegistration::AlignResult SmallGicpRegistration::align(
  const PointCloud & source_cloud_body,
  const IPointCloudMap & map_world,
  const Eigen::Isometry3d & initial_transform_world_body)
{
  AlignResult result;
  result.transform_world_body = initial_transform_world_body;

  auto source_pcl = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  source_pcl->points.reserve(source_cloud_body.size());
  for (const Point & input_point : source_cloud_body.points) {
    pcl::PointXYZ converted_point;
    converted_point.x = input_point.x;
    converted_point.y = input_point.y;
    converted_point.z = input_point.z;
    source_pcl->points.push_back(converted_point);
  }
  source_pcl->width = static_cast<uint32_t>(source_pcl->points.size());
  source_pcl->height = 1;

  auto target_intensity_cloud = map_world.toPointCloud();
  auto target_pcl = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  target_pcl->points.reserve(target_intensity_cloud->size());
  for (const Point & map_point : target_intensity_cloud->points) {
    pcl::PointXYZ converted_point;
    converted_point.x = map_point.x;
    converted_point.y = map_point.y;
    converted_point.z = map_point.z;
    target_pcl->points.push_back(converted_point);
  }
  target_pcl->width = static_cast<uint32_t>(target_pcl->points.size());
  target_pcl->height = 1;

  if (target_pcl->empty() || source_pcl->empty()) {
    return result;
  }

  impl_->engine->setInputTarget(target_pcl);
  impl_->engine->setInputSource(source_pcl);

  pcl::PointCloud<pcl::PointXYZ> aligned;
  Eigen::Matrix4f initial_matrix = initial_transform_world_body.matrix().cast<float>();
  impl_->engine->align(aligned, initial_matrix);

  const Eigen::Matrix4d final_matrix =
    impl_->engine->getFinalTransformation().cast<double>();
  result.transform_world_body.matrix() = final_matrix;
  // small_gicp::RegistrationPCL は nr_iterations_ が protected のため、診断値は推定値で埋める。
  result.iterations = impl_->engine->hasConverged() ? -1 : config_.max_iterations;
  result.converged = impl_->engine->hasConverged();
  result.num_correspondences = static_cast<int>(source_pcl->size());
  return result;
}

std::string SmallGicpRegistration::describe() const
{
  std::ostringstream oss;
  oss << (config_.variant == Variant::VGICP ? "small_gicp_vgicp" : "small_gicp_gicp")
      << ":voxel=" << config_.map_voxel_resolution_m
      << ",threads=" << config_.num_threads;
  return oss.str();
}

}  // namespace pylot_lio

#endif  // PYLOT_LIO_HAS_SMALL_GICP
