// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#ifndef PYLOT_LIO__KEYFRAME__I_KEYFRAME_SELECTOR_HPP_
#define PYLOT_LIO__KEYFRAME__I_KEYFRAME_SELECTOR_HPP_

#include <memory>
#include <string>

#include <Eigen/Geometry>

namespace pylot_lio
{

// 「現在姿勢が新しいキーフレームに値するか」を判定する役割。
// LIO ノードは毎スキャンで registration はするが、map.insertScan は keyframe のときだけ行う。
// これにより:
//   - map が重複点で肥大化しない
//   - 停止中 (姿勢変化なし) は新規挿入が止まる
//   - 後段の pose graph 最適化や loop closure に渡しやすい「節点」が定義できる
class IKeyframeSelector
{
public:
  virtual ~IKeyframeSelector() = default;

  // 初回呼び出し時は強制的に true を返す (= 最初のスキャンで初期 keyframe を作る)。
  // それ以降は実装ごとの基準で判定する。
  // shouldCreateKeyframe が true を返したら呼び出し側は keyframe として処理した上で
  // commit(current_pose) を呼んで「直近 keyframe」状態を更新すること。
  virtual bool shouldCreateKeyframe(const Eigen::Isometry3d & current_pose) = 0;

  virtual void commit(const Eigen::Isometry3d & accepted_pose) = 0;

  // これまでに生成された keyframe 数。
  virtual std::size_t keyframeCount() const = 0;

  virtual std::string describe() const = 0;
};

using IKeyframeSelectorPtr = std::unique_ptr<IKeyframeSelector>;

}  // namespace pylot_lio

#endif  // PYLOT_LIO__KEYFRAME__I_KEYFRAME_SELECTOR_HPP_
