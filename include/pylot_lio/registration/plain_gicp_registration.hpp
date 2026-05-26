// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#ifndef PYLOT_LIO__REGISTRATION__PLAIN_GICP_REGISTRATION_HPP_
#define PYLOT_LIO__REGISTRATION__PLAIN_GICP_REGISTRATION_HPP_

#include <string>

#include "pylot_lio/registration/i_registration.hpp"

namespace pylot_lio
{

// 外部ライブラリ非依存の Generalized ICP (point-to-distribution) 実装。
// IPointCloudMap が返す近傍点の局所共分散を Mahalanobis 重みとして使い、
// SE(3) 上の Gauss-Newton 法で T_world_body を最適化する。
// 数学的詳細は docs/ALGORITHMS.md を参照。
class PlainGicpRegistration : public IRegistration
{
public:
  struct Config
  {
    int max_iterations = 20;
    double convergence_translation_m = 1e-4;
    double convergence_rotation_rad = 1e-4;
    double max_correspondence_distance_m = 2.0;
    double huber_threshold = 1.0;

    // ============================================================
    // 縮退正則化 (Tuna 2024, "X-ICP: Informed, Constrained, Aligned" 風)
    // 参照: sycl_points::algorithms::registration::DegenerateRegularization
    //       (Apache-2.0, https://arxiv.org/abs/2408.11809)
    //
    // 6x6 Hessian を [rotation 3x3 | translation 3x3] のブロック対角に分けて
    // それぞれ固有値分解し、固有値 < threshold の方向は「縮退している」とみなして
    // その方向だけ Tikhonov 正則化 (初期推定に引き戻すペナルティ) を加える。
    // GICP-only で回転自由度が点群の幾何から拘束されない (= 廊下のように平行直線が
    // 多く回転がほぼ自由) 場面で姿勢が暴れるのを防ぐ。
    // ============================================================
    bool enable_degenerate_regularization = false;
    // しきい値は「inlier 数で正規化した固有値」と比較する。
    // 小さくすると正則化される方向が増え、初期推定への引き戻しが強くなる。
    double rotation_eigenvalue_threshold = 10.0;
    double translation_eigenvalue_threshold = 1.0;
    // ペナルティ全体の倍率。inlier 数を掛けて使うので 1.0 でも十分強い。
    double regularization_base_factor = 1.0;
  };

  explicit PlainGicpRegistration(const Config & config);

  AlignResult align(
    const PointCloud & source_cloud_body,
    const IPointCloudMap & map_world,
    const Eigen::Isometry3d & initial_transform_world_body) override;

  std::string describe() const override;

private:
  Config config_;
};

}  // namespace pylot_lio

#endif  // PYLOT_LIO__REGISTRATION__PLAIN_GICP_REGISTRATION_HPP_
