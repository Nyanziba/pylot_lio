// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#include <gtest/gtest.h>

#include "pylot_lio/keyframe/distance_keyframe_selector.hpp"
#include "pylot_lio/lie_algebra.hpp"

namespace pylot_lio
{

namespace
{
Eigen::Isometry3d makePose(double tx, double ty, double tz, double yaw_rad = 0.0)
{
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.linear() = lie::expSO3(Eigen::Vector3d(0.0, 0.0, yaw_rad));
  pose.translation() = Eigen::Vector3d(tx, ty, tz);
  return pose;
}
}  // namespace

TEST(DistanceKeyframeSelector, FirstCallAlwaysCreatesKeyframe)
{
  DistanceKeyframeSelector selector(DistanceKeyframeSelector::Config{});
  EXPECT_TRUE(selector.shouldCreateKeyframe(makePose(0.0, 0.0, 0.0)));
}

TEST(DistanceKeyframeSelector, SmallMotionDoesNotTriggerKeyframe)
{
  DistanceKeyframeSelector::Config config;
  config.min_translation_m = 1.0;
  config.min_rotation_rad = 0.5;
  config.max_scans_between_keyframes = 1000;  // 大きく取って距離/回転のみで判定
  DistanceKeyframeSelector selector(config);

  selector.shouldCreateKeyframe(makePose(0.0, 0.0, 0.0));
  selector.commit(makePose(0.0, 0.0, 0.0));

  EXPECT_FALSE(selector.shouldCreateKeyframe(makePose(0.1, 0.1, 0.0)));
}

TEST(DistanceKeyframeSelector, TranslationThresholdTriggers)
{
  DistanceKeyframeSelector::Config config;
  config.min_translation_m = 1.0;
  config.min_rotation_rad = 1e9;        // 回転は実質無効
  config.max_scans_between_keyframes = 0;
  DistanceKeyframeSelector selector(config);

  selector.shouldCreateKeyframe(makePose(0.0, 0.0, 0.0));
  selector.commit(makePose(0.0, 0.0, 0.0));

  EXPECT_TRUE(selector.shouldCreateKeyframe(makePose(1.5, 0.0, 0.0)));
}

TEST(DistanceKeyframeSelector, RotationThresholdTriggers)
{
  DistanceKeyframeSelector::Config config;
  config.min_translation_m = 1e9;
  config.min_rotation_rad = 0.1;
  config.max_scans_between_keyframes = 0;
  DistanceKeyframeSelector selector(config);

  selector.shouldCreateKeyframe(makePose(0.0, 0.0, 0.0));
  selector.commit(makePose(0.0, 0.0, 0.0));

  EXPECT_TRUE(selector.shouldCreateKeyframe(makePose(0.0, 0.0, 0.0, 0.2)));
}

TEST(DistanceKeyframeSelector, MaxScansTriggers)
{
  DistanceKeyframeSelector::Config config;
  config.min_translation_m = 1e9;
  config.min_rotation_rad = 1e9;
  config.max_scans_between_keyframes = 3;
  DistanceKeyframeSelector selector(config);

  selector.shouldCreateKeyframe(makePose(0.0, 0.0, 0.0));
  selector.commit(makePose(0.0, 0.0, 0.0));
  EXPECT_FALSE(selector.shouldCreateKeyframe(makePose(0.0, 0.0, 0.0)));  // scans=1
  EXPECT_FALSE(selector.shouldCreateKeyframe(makePose(0.0, 0.0, 0.0)));  // scans=2
  EXPECT_TRUE(selector.shouldCreateKeyframe(makePose(0.0, 0.0, 0.0)));   // scans=3
}

TEST(DistanceKeyframeSelector, CommitIncrementsCount)
{
  DistanceKeyframeSelector selector(DistanceKeyframeSelector::Config{});
  EXPECT_EQ(selector.keyframeCount(), 0u);
  selector.shouldCreateKeyframe(makePose(0.0, 0.0, 0.0));
  selector.commit(makePose(0.0, 0.0, 0.0));
  EXPECT_EQ(selector.keyframeCount(), 1u);
}

}  // namespace pylot_lio

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
