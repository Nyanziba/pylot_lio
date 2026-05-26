// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#ifndef PYLOT_LIO__ESTIMATOR__GICP_ONLY_ESTIMATOR_HPP_
#define PYLOT_LIO__ESTIMATOR__GICP_ONLY_ESTIMATOR_HPP_

#include <cstdint>
#include <string>

#include "pylot_lio/estimator/i_state_estimator.hpp"

namespace pylot_lio
{

// IMU を一切使わず、毎スキャンの registration 結果をそのまま姿勢推定値とする
// 最小実装。比較ベースラインとして使う。
class GicpOnlyEstimator : public IStateEstimator
{
public:
  struct Config
  {
    // --- 等速度モデル外挿のクランプ ---
    // 上限を超えた相対変換は初期推定として危険 (前回の誤対応や物理的な急加速で
    // 実態と乖離している可能性が高い) なので、上限にクランプして GICP の収束範囲を
    // 超えた飛びを抑える。 0 以下で無効化。
    double max_extrapolation_translation_m = 1.0;
    double max_extrapolation_rotation_rad = 0.5;  // ~28.6°

    // --- A. 多候補 initial guess ---
    // true の場合、 (等速度外挿) に加えて (現状 pose = 静止仮定) を候補として
    // align を 2 回叩き、 final_cost が小さい方を採用する。
    // 等速度モデルが一過性に外れたケース (急停止、 前回 align の頭脱) でも
    // 静止仮定側で救済できる。 計算量は 2 倍。
    bool enable_static_candidate = false;

    // --- E. 等加速度モデル候補 ---
    // true の場合、 「前々回 delta」 と 「前回 delta」 から jerk 相当の差を計算し、
    // 等速度外挿に加算した姿勢を候補に加える。 つまり「直前 1 ステップで生まれた
    // 加速度をもう 1 ステップ持続する」 と仮定する。 候補数は最大 3 (等速度・静止・等加速度)。
    // 急加減速時に効くが、 ノイズ増幅のリスクがあるので jerk クランプも合わせて使う。
    bool enable_acceleration_candidate = false;

    // --- B. align 結果のサニティチェック (initial_guess 基準) ---
    // align 結果 pose と initial_guess の SE(3) 差が以下を超えた場合は採用せず、
    // initial_guess を据え置く。 0 以下で無効化。
    // 閾値を緩めるほど暴走を見逃し、 厳しくするほど高速回転・PGO 直後の正当な
    // 補正も落とす。 max_extrapolation_* と同じくらいに揃えるのが目安。
    double max_correction_translation_m = 0.0;
    double max_correction_rotation_rad = 0.0;

    // --- B'. 前回 delta との差 (jerk) によるサニティチェック ---
    // 「前回 1 スキャン分の delta」 と 「今回 align で得られた delta」 の差を見る。
    // 高速回転中は initial_guess との差は大きくても OK だが、 直前 delta との差
    // (=jerk 相当) が大きい場合は急加減速 → 誤対応の可能性が高い。
    // B (max_correction_*) と併用することで「正当な大きな補正」と
    // 「一過性の誤対応ジャンプ」を区別できる。 0 以下で無効化。
    double max_jerk_translation_m = 0.0;
    double max_jerk_rotation_rad = 0.0;

    // --- C. 静止検出 (align 抑制) ---
    // 前回 align 結果と現状 pose の SE(3) delta (= 直前 1 スキャン分の運動) が
    // 並進・回転とも以下の閾値以下のスキャンが stationary_streak_required 回
    // 連続したら、 align をスキップして pose 据え置きにする。
    // ドリフトをマップに焼き込む事故を防ぐ。 0 以下で無効化。
    double stationary_translation_threshold_m = 0.0;
    double stationary_rotation_threshold_rad = 0.0;
    int stationary_streak_required = 3;

    // --- D. SE(3) tangent space EMA ---
    // 出力 pose を「前回 filtered pose から align 結果まで α 分だけ近づける」
    // 形で平滑化する。 1.0 で平滑化なし (filter 無効)、 0.5 で半分、 0 で凍結。
    // 並進と回転で別々に設定する。
    double ema_alpha_translation = 1.0;
    double ema_alpha_rotation = 1.0;
  };

  GicpOnlyEstimator() : GicpOnlyEstimator(Config{}) {}
  explicit GicpOnlyEstimator(const Config & config);

  void initialize(const RobotState & initial_state) override;
  bool isInitialized() const override;

  void predictWithImu(const ImuSample & imu_sample) override;
  void updateWithScan(
    const PointCloud & scan_cloud_body,
    IPointCloudMap & map_world,
    IRegistration & registration,
    int64_t scan_timestamp_ns) override;

  RobotState getState() const override;
  EstimatorDiagnostics getDiagnostics() const override;
  std::string describe() const override;
  bool usesImu() const override { return false; }

  void setPose(const Eigen::Isometry3d & pose_world_body) override;

private:
  Config config_;
  bool initialized_;
  RobotState current_state_;
  // 等速度モデル用。
  bool has_previous_pose_;
  Eigen::Isometry3d previous_pose_world_body_;
  int64_t previous_scan_timestamp_ns_;
  double previous_dt_s_;  // 0 で「未確定」。
  // 等加速度モデル用: 前々回スキャンの pose を保持。 has_older_pose_=true なら
  // 「pose N-2 → pose N-1 (older step)」 を計算でき、 これと 「pose N-1 → pose N (current step)」
  // の差を jerk として等加速度外挿に利用できる。
  bool has_older_pose_;
  Eigen::Isometry3d older_pose_world_body_;
  // 静止検出用ストリークカウンタ。
  int stationary_streak_count_;
  EstimatorDiagnostics last_diagnostics_;
};

}  // namespace pylot_lio

#endif  // PYLOT_LIO__ESTIMATOR__GICP_ONLY_ESTIMATOR_HPP_
