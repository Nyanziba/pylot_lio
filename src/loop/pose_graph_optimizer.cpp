// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#include "pylot_lio/loop/pose_graph_optimizer.hpp"

#include <sstream>
#include <stdexcept>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

namespace pylot_lio
{

namespace
{

// keyframe_id → GTSAM Symbol ('x', kf_id) のシンプル変換。
gtsam::Symbol toSymbol(uint32_t keyframe_id)
{
  return gtsam::Symbol('x', keyframe_id);
}

gtsam::Pose3 toGtsamPose(const Eigen::Isometry3d & pose)
{
  return gtsam::Pose3(gtsam::Rot3(pose.linear()), gtsam::Point3(pose.translation()));
}

Eigen::Isometry3d fromGtsamPose(const gtsam::Pose3 & pose)
{
  Eigen::Isometry3d result = Eigen::Isometry3d::Identity();
  result.linear() = pose.rotation().matrix();
  result.translation() = pose.translation();
  return result;
}

gtsam::SharedNoiseModel toNoiseModel(const PoseGraphOptimizer::NoiseSigmas & sigmas)
{
  // GTSAM Pose3 の tangent space は [rx, ry, rz, tx, ty, tz] (rotation first)。
  gtsam::Vector6 sigma_vector;
  sigma_vector << sigmas.rot_x_rad, sigmas.rot_y_rad, sigmas.rot_z_rad,
    sigmas.trans_x_m, sigmas.trans_y_m, sigmas.trans_z_m;
  return gtsam::noiseModel::Diagonal::Sigmas(sigma_vector);
}

}  // namespace

struct PoseGraphOptimizer::Impl
{
  gtsam::ISAM2 isam2;
  gtsam::NonlinearFactorGraph pending_factors;
  gtsam::Values pending_values;
  gtsam::Values current_estimate;
  std::size_t total_factors_added = 0;
  std::size_t total_keyframes_added = 0;

  Impl()
  {
    gtsam::ISAM2Params params;
    params.relinearizeThreshold = 0.01;
    params.relinearizeSkip = 1;
    isam2 = gtsam::ISAM2(params);
  }
};

PoseGraphOptimizer::PoseGraphOptimizer()
: impl_(std::make_unique<Impl>())
{
}

PoseGraphOptimizer::~PoseGraphOptimizer() = default;

void PoseGraphOptimizer::addKeyframePrior(
  uint32_t keyframe_id, const Eigen::Isometry3d & pose_world_keyframe)
{
  // 最初の keyframe を world に強く固定する prior。 sigma は 1e-6 オーダー。
  gtsam::Vector6 strong_sigmas;
  strong_sigmas << 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6;
  auto noise = gtsam::noiseModel::Diagonal::Sigmas(strong_sigmas);
  impl_->pending_factors.add(
    gtsam::PriorFactor<gtsam::Pose3>(
      toSymbol(keyframe_id), toGtsamPose(pose_world_keyframe), noise));
  if (!impl_->pending_values.exists(toSymbol(keyframe_id)) &&
      !impl_->current_estimate.exists(toSymbol(keyframe_id)))
  {
    impl_->pending_values.insert(toSymbol(keyframe_id), toGtsamPose(pose_world_keyframe));
    ++impl_->total_keyframes_added;
  }
  ++impl_->total_factors_added;
}

void PoseGraphOptimizer::addOdometryFactor(
  uint32_t prev_keyframe_id, uint32_t curr_keyframe_id,
  const Eigen::Isometry3d & relative_transform_prev_to_curr,
  const NoiseSigmas & sigmas)
{
  impl_->pending_factors.add(
    gtsam::BetweenFactor<gtsam::Pose3>(
      toSymbol(prev_keyframe_id), toSymbol(curr_keyframe_id),
      toGtsamPose(relative_transform_prev_to_curr), toNoiseModel(sigmas)));
  ++impl_->total_factors_added;
}

void PoseGraphOptimizer::addLoopFactor(
  uint32_t query_keyframe_id, uint32_t match_keyframe_id,
  const Eigen::Isometry3d & relative_transform_query_to_match,
  const NoiseSigmas & sigmas)
{
  // BetweenFactor<Pose3>(A, B, T_a_b) は T_world_a · T_a_b = T_world_b を意味する。
  // 「query 系の点を match 系に持っていく変換」を「query → match の相対変換」と
  // 解釈してそのまま渡す。
  impl_->pending_factors.add(
    gtsam::BetweenFactor<gtsam::Pose3>(
      toSymbol(query_keyframe_id), toSymbol(match_keyframe_id),
      toGtsamPose(relative_transform_query_to_match), toNoiseModel(sigmas)));
  ++impl_->total_factors_added;
}

void PoseGraphOptimizer::addKeyframeInitialEstimate(
  uint32_t keyframe_id, const Eigen::Isometry3d & pose_world_keyframe)
{
  if (impl_->pending_values.exists(toSymbol(keyframe_id)) ||
      impl_->current_estimate.exists(toSymbol(keyframe_id)))
  {
    return;
  }
  impl_->pending_values.insert(toSymbol(keyframe_id), toGtsamPose(pose_world_keyframe));
  ++impl_->total_keyframes_added;
}

void PoseGraphOptimizer::optimize()
{
  if (impl_->pending_factors.empty() && impl_->pending_values.empty()) {
    return;
  }
  impl_->isam2.update(impl_->pending_factors, impl_->pending_values);
  impl_->pending_factors.resize(0);
  impl_->pending_values.clear();
  impl_->current_estimate = impl_->isam2.calculateEstimate();
}

Eigen::Isometry3d PoseGraphOptimizer::optimizedPose(uint32_t keyframe_id) const
{
  const gtsam::Symbol key = toSymbol(keyframe_id);
  if (impl_->current_estimate.exists(key)) {
    return fromGtsamPose(impl_->current_estimate.at<gtsam::Pose3>(key));
  }
  if (impl_->pending_values.exists(key)) {
    return fromGtsamPose(impl_->pending_values.at<gtsam::Pose3>(key));
  }
  return Eigen::Isometry3d::Identity();
}

std::vector<std::pair<uint32_t, Eigen::Isometry3d>>
PoseGraphOptimizer::allOptimizedPoses() const
{
  std::vector<std::pair<uint32_t, Eigen::Isometry3d>> output;
  for (const auto & key_value_pair : impl_->current_estimate) {
    const gtsam::Symbol symbol(key_value_pair.key);
    if (symbol.chr() != 'x') {
      continue;
    }
    output.emplace_back(
      static_cast<uint32_t>(symbol.index()),
      fromGtsamPose(key_value_pair.value.cast<gtsam::Pose3>()));
  }
  return output;
}

std::size_t PoseGraphOptimizer::numFactors() const
{
  return impl_->total_factors_added;
}

std::size_t PoseGraphOptimizer::numKeyframes() const
{
  return impl_->total_keyframes_added;
}

std::string PoseGraphOptimizer::describe() const
{
  std::ostringstream oss;
  oss << "pose_graph_optimizer(gtsam_isam2):"
      << "factors=" << impl_->total_factors_added
      << ",keyframes=" << impl_->total_keyframes_added;
  return oss.str();
}

}  // namespace pylot_lio
