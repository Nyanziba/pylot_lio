// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#ifndef PYLOT_LIO__REGISTRATION__PCL_NDT_REGISTRATION_HPP_
#define PYLOT_LIO__REGISTRATION__PCL_NDT_REGISTRATION_HPP_

#include <string>

#include "pylot_lio/registration/i_registration.hpp"

namespace pylot_lio
{

// PCL の Normal Distributions Transform (NDT) を IRegistration インタフェースに
// 被せた薄いラッパ。 NDT は target 空間をボクセルに区切り、 各ボクセル内の点群を
// 1 個の 3D ガウシアンで要約する。 source 点はそのガウシアンに対する負対数尤度を
// 最小化する向きに動かされる。
//
// GICP との違い:
//   - 「source 点と target 点の最近傍ペア」 という対応付けが無い。
//     source 点が落ちたボクセルのガウシアンに対して尤度を計算するだけ。
//   - 対応付けがイテレーション間で離散的に切り替わらないので、 目的関数が滑らかで
//     初期姿勢に頑健になりやすい (代わりに resolution への感度が大きい)。
//
// 本クラスはまず CPU ベースラインとしての pcl::NDT を使う。
// ndt_omp (OpenMP 並列) や Metal NDT は別ラッパとして追加予定。
class PclNdtRegistration : public IRegistration
{
public:
  struct Config
  {
    // ボクセル一辺の長さ [m]。 Livox Mid-360 屋内では 0.5〜2.0 m 程度を rosbag で
    // スイープして決めるのが定石。 大きすぎるとガウシアンが太って精度が出ず、
    // 小さすぎると点不足でボクセルが破棄されすぎる。
    double resolution_m = 1.0;

    // More-Thuente line search の最大ステップ長 [m]。 PCL のデフォルト 0.1。
    double step_size_m = 0.1;

    // 並進収束判定しきい値 [m]。 PCL のデフォルト 0.01。
    double transformation_epsilon_m = 0.01;

    // 最大反復回数。
    int max_iterations = 35;

    // 最小点数。 source または target がこれ未満の場合は align を試みず初期推定を返す。
    int min_points_for_alignment = 10;
  };

  explicit PclNdtRegistration(const Config & config);

  AlignResult align(
    const PointCloud & source_cloud_body,
    const IPointCloudMap & map_world,
    const Eigen::Isometry3d & initial_transform_world_body) override;

  std::string describe() const override;

private:
  Config config_;
};

}  // namespace pylot_lio

#endif  // PYLOT_LIO__REGISTRATION__PCL_NDT_REGISTRATION_HPP_
