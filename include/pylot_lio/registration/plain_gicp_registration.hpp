// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#ifndef PYLOT_LIO__REGISTRATION__PLAIN_GICP_REGISTRATION_HPP_
#define PYLOT_LIO__REGISTRATION__PLAIN_GICP_REGISTRATION_HPP_

#include <string>
#include <vector>

#include <Eigen/Core>

#include "pylot_lio/registration/i_registration.hpp"

namespace pylot_lio
{

// 外部ライブラリ非依存の Generalized ICP (Segal et al. 2009) 実装。
// source 点群と target マップの双方に局所共分散を持たせ、 残差 d_i に対する
// 重み行列を (C_t + R C_s R^T)^{-1} とする「distribution-to-distribution」型。
//   - C_t: IPointCloudMap が返す近傍点の局所共分散 (target, world フレーム)
//   - C_s: source 点群の近傍 k 点から計算する局所共分散 (body フレーム)。
//          source は body フレームで定義されるため、 world フレームの残差 d と
//          整合させるべく R C_s R^T で world に回してから C_t と足す。
// SE(3) 上の Gauss-Newton 法で T_world_body を最適化する。
// 数学的詳細は docs/ALGORITHMS.md / docs/MATH.md を参照。
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
    // source 共分散 C_s の計算 (GICP の distribution-to-distribution 化)
    // ============================================================
    // source 点群の各点について、 近傍 k 点から局所共分散を推定する。
    // k が小さいと共分散が退化しやすく、 大きいと近傍が広がりすぎて
    // 細部の構造をならしてしまう。 10 前後が標準。
    int source_covariance_num_neighbors = 10;
    // 平面性正則化 (Segal 2009, plane-to-plane): 推定した共分散を固有値分解し、
    // 固有値を (epsilon, 1, 1) に置換する (最小固有値方向 = 法線方向 を epsilon、
    // 平面に沿う 2 方向を 1)。 これにより「点は局所平面上に乗っている」という
    // 事前知識を共分散に埋め込み、 平面に沿うズレは罰さず法線方向のズレを罰する。
    double source_covariance_plane_epsilon = 1e-3;

    // OpenMP 並列化用スレッド数。
    //   <= 0: omp_get_max_threads() に任せる (OpenMP が無効の場合は逐次)
    //   1   : 逐次実行 (OpenMP 経路を踏まない、 デバッグ用途)
    //   >=2 : 明示的スレッド数を num_threads(...) clause で指定
    // 小規模点群 (~1000 以下) では並列化オーバヘッドが目立つので 1 〜 2 が無難。
    // BackendConfig::registration_num_threads (small_gicp と共通) から流し込まれる。
    int num_threads = 1;

    // 並列バックエンド: "omp" (OpenMP) | "tbb" (Intel TBB)。
    // 点ごとの Hessian/gradient 累算を、 OpenMP では per-thread 手動 accumulate→合算、
    // TBB では tbb::parallel_reduce で行う。 TBB 非対応ビルド (PYLOT_LIO_HAS_TBB 未定義)
    // では "tbb" 指定でも OpenMP/逐次にフォールバックする。
    std::string parallel_backend = "omp";

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
  // source 点群の各点について、 body フレームでの局所共分散 C_s を計算する。
  // R に依存しないので align ループの外で 1 回だけ呼ぶ。 戻り値は
  // source_cloud_body.points と同じ並び順・同じ長さ。 近傍が 3 点未満で
  // 共分散を組めない点は epsilon * I を入れる (= ほぼ等方、 重みを効かせない)。
  std::vector<Eigen::Matrix3d> computeSourceCovariances(
    const PointCloud & source_cloud_body) const;

  Config config_;
};

}  // namespace pylot_lio

#endif  // PYLOT_LIO__REGISTRATION__PLAIN_GICP_REGISTRATION_HPP_
