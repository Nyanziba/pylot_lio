// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#include "pylot_lio/loop/loop_closure_icp.hpp"

#include <memory>

#ifdef PYLOT_LIO_HAS_SMALL_GICP
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <small_gicp/pcl/pcl_registration.hpp>
#endif

namespace pylot_lio
{

namespace
{

#ifdef PYLOT_LIO_HAS_SMALL_GICP
pcl::PointCloud<pcl::PointXYZ>::Ptr toPlainXyz(const PointCloud & input)
{
  auto output = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  output->points.reserve(input.points.size());
  for (const Point & sample : input.points) {
    pcl::PointXYZ converted_point;
    converted_point.x = sample.x;
    converted_point.y = sample.y;
    converted_point.z = sample.z;
    output->points.push_back(converted_point);
  }
  output->width = static_cast<uint32_t>(output->points.size());
  output->height = 1;
  output->is_dense = true;
  return output;
}
#endif

}  // namespace

LoopClosureIcpResult refineLoopTransform(
  const PointCloud & query_cloud_body,
  const PointCloud & match_cloud_body,
  const Eigen::Isometry3d & initial_transform_match_from_query,
  const LoopClosureIcpConfig & config)
{
  LoopClosureIcpResult result;
  result.transform_match_from_query = initial_transform_match_from_query;

#ifdef PYLOT_LIO_HAS_SMALL_GICP
  result.backend_name = "small_gicp_vgicp";
  // source = query, target = match の方が「query 系点群を match 系に持っていく」
  // 変換が直接得られて自然。
  auto source = toPlainXyz(query_cloud_body);
  auto target = toPlainXyz(match_cloud_body);
  if (source->empty() || target->empty()) {
    return result;
  }

  small_gicp::RegistrationPCL<pcl::PointXYZ, pcl::PointXYZ> engine;
  engine.setRegistrationType("VGICP");
  engine.setVoxelResolution(config.voxel_resolution_m);
  engine.setMaxCorrespondenceDistance(config.max_correspondence_distance_m);
  engine.setMaximumIterations(config.max_iterations);
  engine.setNumThreads(config.num_threads);
  engine.setInputTarget(target);
  engine.setInputSource(source);

  pcl::PointCloud<pcl::PointXYZ> aligned;
  Eigen::Matrix4f initial_guess_4x4 =
    initial_transform_match_from_query.matrix().cast<float>();
  engine.align(aligned, initial_guess_4x4);

  result.converged = engine.hasConverged();
  result.fitness_score = engine.getFitnessScore();
  const Eigen::Matrix4f final_transform = engine.getFinalTransformation();
  result.transform_match_from_query.matrix() = final_transform.cast<double>();
#else
  result.backend_name = "fallback_identity";
  (void)query_cloud_body;
  (void)match_cloud_body;
  (void)config;
#endif

  return result;
}

}  // namespace pylot_lio
