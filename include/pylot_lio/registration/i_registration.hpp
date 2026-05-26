// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#ifndef PYLOT_LIO__REGISTRATION__I_REGISTRATION_HPP_
#define PYLOT_LIO__REGISTRATION__I_REGISTRATION_HPP_

#include <memory>
#include <string>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "pylot_lio/map/i_point_cloud_map.hpp"
#include "pylot_lio/types.hpp"

namespace pylot_lio
{

// 1スキャンの点群を、既存のマップに重ね合わせるための「最適化器」の抽象。
// 入力: 未変換の source 点群、マップ、初期 T_world_body。
// 出力: マッチング後の T_world_body と診断情報。
//
// IESKF / HGO は IPointCloudMap::findNearestNeighbor を直接使うため必須ではないが、
// GICP-only 状態推定器はこのインタフェースを 1 回呼ぶだけで姿勢推定を完結させる。
class IRegistration
{
public:
  struct AlignResult
  {
    Eigen::Isometry3d transform_world_body = Eigen::Isometry3d::Identity();
    int iterations = 0;
    double final_cost = 0.0;
    bool converged = false;
    int num_correspondences = 0;
  };

  virtual ~IRegistration() = default;

  // source_cloud は body フレーム (LiDAR フレーム = body と仮定; 別の場合は呼び出し側で変換)。
  // initial_transform_world_body を初期値として最適化を行う。
  virtual AlignResult align(
    const PointCloud & source_cloud_body,
    const IPointCloudMap & map_world,
    const Eigen::Isometry3d & initial_transform_world_body) = 0;

  virtual std::string describe() const = 0;
};

using IRegistrationPtr = std::unique_ptr<IRegistration>;

}  // namespace pylot_lio

#endif  // PYLOT_LIO__REGISTRATION__I_REGISTRATION_HPP_
