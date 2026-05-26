// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#ifndef PYLOT_LIO__KEYFRAME__DISTANCE_KEYFRAME_SELECTOR_HPP_
#define PYLOT_LIO__KEYFRAME__DISTANCE_KEYFRAME_SELECTOR_HPP_

#include <optional>
#include <string>

#include "pylot_lio/keyframe/i_keyframe_selector.hpp"

namespace pylot_lio
{

// 距離 / 回転 / 経過カウントの 3 条件 OR で keyframe を作る素朴な判定器。
// 直前 keyframe との SE(3) 差分を見るだけなのでロジックは ROS 非依存。
class DistanceKeyframeSelector : public IKeyframeSelector
{
public:
  struct Config
  {
    // 直前 keyframe から並進が これ以上 動いたら新 keyframe (0 以下で無効)。
    double min_translation_m = 1.0;
    // 同じく回転 (rad)。0 以下で無効。
    double min_rotation_rad = 0.2;  // ~11.5 deg
    // 上の閾値より小さくても、N スキャンに 1 回は強制的に keyframe を作る (0 以下で無効)。
    // 停止中は IMU しか動かない場面で「最低限の進捗確認」用。
    int max_scans_between_keyframes = 50;
  };

  explicit DistanceKeyframeSelector(const Config & config);

  bool shouldCreateKeyframe(const Eigen::Isometry3d & current_pose) override;
  void commit(const Eigen::Isometry3d & accepted_pose) override;
  std::size_t keyframeCount() const override;
  std::string describe() const override;

private:
  Config config_;
  std::optional<Eigen::Isometry3d> last_keyframe_pose_;
  int scans_since_last_keyframe_;
  std::size_t keyframe_count_;
};

}  // namespace pylot_lio

#endif  // PYLOT_LIO__KEYFRAME__DISTANCE_KEYFRAME_SELECTOR_HPP_
