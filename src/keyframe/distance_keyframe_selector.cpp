// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#include "pylot_lio/keyframe/distance_keyframe_selector.hpp"

#include <sstream>

#include "pylot_lio/lie_algebra.hpp"

namespace pylot_lio
{

DistanceKeyframeSelector::DistanceKeyframeSelector(const Config & config)
: config_(config),
  scans_since_last_keyframe_(0),
  keyframe_count_(0)
{
}

bool DistanceKeyframeSelector::shouldCreateKeyframe(const Eigen::Isometry3d & current_pose)
{
  ++scans_since_last_keyframe_;
  if (!last_keyframe_pose_.has_value()) {
    return true;  // 初回は必ず keyframe
  }
  const Eigen::Isometry3d & last_pose = *last_keyframe_pose_;
  const double translation_diff =
    (current_pose.translation() - last_pose.translation()).norm();
  const Eigen::Matrix3d delta_rotation_matrix =
    current_pose.linear() * last_pose.linear().transpose();
  const double rotation_diff = lie::logSO3(delta_rotation_matrix).norm();

  const bool translation_triggered =
    config_.min_translation_m > 0.0 && translation_diff >= config_.min_translation_m;
  const bool rotation_triggered =
    config_.min_rotation_rad > 0.0 && rotation_diff >= config_.min_rotation_rad;
  const bool max_scans_triggered =
    config_.max_scans_between_keyframes > 0
    && scans_since_last_keyframe_ >= config_.max_scans_between_keyframes;
  return translation_triggered || rotation_triggered || max_scans_triggered;
}

void DistanceKeyframeSelector::commit(const Eigen::Isometry3d & accepted_pose)
{
  last_keyframe_pose_ = accepted_pose;
  scans_since_last_keyframe_ = 0;
  ++keyframe_count_;
}

std::size_t DistanceKeyframeSelector::keyframeCount() const
{
  return keyframe_count_;
}

std::string DistanceKeyframeSelector::describe() const
{
  std::ostringstream oss;
  oss << "distance_keyframe:translation=" << config_.min_translation_m
      << "m,rotation=" << config_.min_rotation_rad
      << "rad,max_scans=" << config_.max_scans_between_keyframes
      << ",count=" << keyframe_count_;
  return oss.str();
}

}  // namespace pylot_lio
