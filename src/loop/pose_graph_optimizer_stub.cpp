// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
// GTSAM が利用できない環境向けの stub。 全 API は no-op で、 optimizedPose は identity。
// lio_node が無条件に PoseGraphOptimizer を持てるようにする (機能は無効化される)。
#include "pylot_lio/loop/pose_graph_optimizer.hpp"

#include <sstream>

namespace pylot_lio
{

struct PoseGraphOptimizer::Impl
{
};

PoseGraphOptimizer::PoseGraphOptimizer()
: impl_(std::make_unique<Impl>())
{
}

PoseGraphOptimizer::~PoseGraphOptimizer() = default;

void PoseGraphOptimizer::addKeyframePrior(
  uint32_t /*keyframe_id*/, const Eigen::Isometry3d & /*pose*/)
{
}

void PoseGraphOptimizer::addOdometryFactor(
  uint32_t /*prev*/, uint32_t /*curr*/, const Eigen::Isometry3d & /*relative*/,
  const NoiseSigmas & /*sigmas*/)
{
}

void PoseGraphOptimizer::addLoopFactor(
  uint32_t /*query*/, uint32_t /*match*/, const Eigen::Isometry3d & /*relative*/,
  const NoiseSigmas & /*sigmas*/)
{
}

void PoseGraphOptimizer::addKeyframeInitialEstimate(
  uint32_t /*keyframe_id*/, const Eigen::Isometry3d & /*pose*/)
{
}

void PoseGraphOptimizer::optimize()
{
}

Eigen::Isometry3d PoseGraphOptimizer::optimizedPose(uint32_t /*keyframe_id*/) const
{
  return Eigen::Isometry3d::Identity();
}

std::vector<std::pair<uint32_t, Eigen::Isometry3d>>
PoseGraphOptimizer::allOptimizedPoses() const
{
  return {};
}

std::size_t PoseGraphOptimizer::numFactors() const
{
  return 0;
}

std::size_t PoseGraphOptimizer::numKeyframes() const
{
  return 0;
}

std::string PoseGraphOptimizer::describe() const
{
  return "pose_graph_optimizer(stub):gtsam_unavailable";
}

}  // namespace pylot_lio
