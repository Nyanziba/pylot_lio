// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#ifndef PYLOT_LIO__FACTORY_HPP_
#define PYLOT_LIO__FACTORY_HPP_

#include <string>
#include <vector>

#include "pylot_lio/estimator/i_state_estimator.hpp"
#include "pylot_lio/map/i_point_cloud_map.hpp"
#include "pylot_lio/preprocess/i_preprocessor.hpp"
#include "pylot_lio/registration/i_registration.hpp"

namespace pylot_lio
{

// ROS パラメータの組み合わせから 4 つの抽象実装インスタンスを生成する単一エントリポイント。
// lio_node はここから受け取った unique_ptr 4 つだけを所有する。
struct LioBackendConfig
{
  // preprocessor
  std::string preprocessor_name = "voxel_grid";
    // voxel_grid | random_sampling | voxel_random_sampling
  double voxel_grid_size_m = 0.3;
  int random_sampling_target_count = 8192;
  // voxel_random_sampling 用: 乱数 seed (0 でマシン乱数)
  int voxel_random_sampling_seed = 12345;

  // map
  std::string map_name = "voxel_keyframe_submap";
    // voxel_map | normal_map | kd_tree_map | voxel_random_map | voxel_keyframe_submap
  double map_voxel_size_m = 0.5;
  int map_min_points_per_cell = 5;
  std::size_t map_max_total_cells = 200000;
  // voxel_random_map 専用パラメータ
  int voxel_random_map_max_points_per_voxel = 8;
  int voxel_random_map_k_nearest_for_covariance = 8;
  // voxel_keyframe_submap 専用パラメータ
  // sliding window: registration target に含める直近キーフレーム数
  int submap_window_size = 20;
  // M 個キーフレームごとに 1 個の「確定 submap」スナップショットを内部保存する
  int submap_finalize_size = 20;
  int submap_max_points_per_cell = 32;
  int submap_k_nearest_for_covariance = 8;
  // cell_sampling_mode: "keep_up_to_n" (既定) | "random_one"
  // random_one を選ぶと submap_max_points_per_cell は事実上 1 になり、 メモリと
  // PCD ファイルサイズが大幅に減る。 GICP の対応点も「実点」を保つ。
  std::string submap_cell_sampling_mode = "keep_up_to_n";
  int submap_random_seed = 12345;

  // loop closure (Scan Context detection)
  bool enable_loop_detection = true;
  int loop_num_rings = 60;
  int loop_num_sectors = 20;
  double loop_max_radius_m = 80.0;
  double loop_min_radius_m = 0.5;
  int loop_ring_key_top_k = 10;
  // Scan Context 距離しきい値。 false positive が多いログを観察した結果、
  // 0.2 だと毎 keyframe に近い頻度で誤検出が起きるため 0.10 をデフォルトにする。
  double loop_score_threshold = 0.10;
  int loop_exclude_recent_kf = 50;

  // Loop closure 受け入れゲート (PGO 爆発防止):
  // 1) ICP fitness > 閾値なら採用しない (small_gicp の MSE 等価指標)
  double loop_icp_max_fitness_score = 1.0;
  // 2) ICP 結果が初期推定から大きく跳んだら採用しない (点群が完全に misalign)
  double loop_icp_max_translation_jump_m = 10.0;
  // 3) 同じ match_kf に短期間で複数回マッチしたら cooldown でスキップ
  int loop_match_cooldown_keyframes = 5;

  // pose graph optimization (GTSAM)
  bool enable_pose_graph_optimization = true;
  // PGO トリガ間隔 (keyframe 数)。 加えて loop 検出時にも即トリガ
  int pgo_optimize_every_n_keyframes = 10;
  // loop ICP の voxel 解像度 / 最大対応距離
  double loop_icp_voxel_size_m = 0.5;
  double loop_icp_max_correspondence_m = 5.0;
  int loop_icp_max_iterations = 50;
  // PGO factor のノイズ標準偏差
  double pgo_odometry_sigma_rot_rad = 0.01;
  double pgo_odometry_sigma_trans_m = 0.05;
  double pgo_loop_sigma_rot_rad = 0.05;
  double pgo_loop_sigma_trans_m = 0.1;

  // registration
  std::string registration_name = "small_gicp_vgicp";
    // plain_gicp | small_gicp_gicp | small_gicp_vgicp | sycl
  double registration_max_correspondence_m = 2.0;
  int registration_num_threads = 4;
  int registration_max_iterations = 30;

  // plain_gicp 用: 縮退方向 Tikhonov 正則化 (X-ICP / sycl_points 流)。
  // 廊下や対称的な環境で回転/並進が拘束されないときに姿勢が暴れるのを防ぐ。
  bool enable_degenerate_regularization = false;
  double rotation_eigenvalue_threshold = 10.0;
  double translation_eigenvalue_threshold = 1.0;
  double regularization_base_factor = 1.0;

  // state estimator
  std::string state_estimator_name = "ieskf";    // ieskf | hgo | gicp_only

  // gicp_only 専用パラメータ。 ROS yaml 上は 'gicp_only_*' プレフィックスで宣言する。
  // 各機能は 0 / false / 1.0 などのデフォルトで「無効」になる後方互換設計。
  double gicp_only_max_extrapolation_translation_m = 1.0;
  double gicp_only_max_extrapolation_rotation_rad = 0.5;
  bool gicp_only_enable_static_candidate = false;
  bool gicp_only_enable_acceleration_candidate = false;
  double gicp_only_max_correction_translation_m = 0.0;
  double gicp_only_max_correction_rotation_rad = 0.0;
  double gicp_only_max_jerk_translation_m = 0.0;
  double gicp_only_max_jerk_rotation_rad = 0.0;
  double gicp_only_stationary_translation_threshold_m = 0.0;
  double gicp_only_stationary_rotation_threshold_rad = 0.0;
  int gicp_only_stationary_streak_required = 3;
  double gicp_only_ema_alpha_translation = 1.0;
  double gicp_only_ema_alpha_rotation = 1.0;

  // IMU linear_acceleration の単位補正係数。 sensor_msgs/Imu の規約は m/s² だが、
  // Livox driver 等は実際には g 単位で出してくるケースがある (静止時 acc.z ≈ 1.0)。
  // その場合は 9.80665 を指定して m/s² にスケールアップする。 規約通りの IMU では 1.0。
  double imu_acceleration_scale = 1.0;

  // IESKF / HGO 用: 重力大きさを固定値として持つ (Super-LIO lio.sensor.gravity_norm 同等)。
  // 起動時 IMU 平均から「方向」だけ取り、大きさは常に gravity_norm を使う設計。
  // これにより IMU の単位/取り付け誤差が重力大きさ推定に乗らず、 z drift を抑える。
  double gravity_norm = 9.7946;

  // LiDAR ↔ IMU extrinsic (Super-LIO lio.extrinsic.lidar_imu と同じ定義)。
  // 「LiDAR 点を IMU frame に持ってくる」変換 T_imu_lidar。
  // Mid-360 のデフォルト値は livox_360.yaml と整合: R=I, t≈(-0.011, -0.023, 0.044) m。
  // rotation は行優先 9 要素。 identity が既定。
  double extrinsic_translation_imu_from_lidar_x = -0.011;
  double extrinsic_translation_imu_from_lidar_y = -0.02329;
  double extrinsic_translation_imu_from_lidar_z = 0.04412;
  // 9 個の double を ROS パラメータ (double_array) で受ける。 デフォルトは 3x3 identity。
  // ROS yaml 上は extrinsic_rotation_imu_from_lidar: [1,0,0, 0,1,0, 0,0,1] のように記述。
  // (lio_node 側でパース。 ここでは std::vector<double> に格納)。
  std::vector<double> extrinsic_rotation_imu_from_lidar_row_major = {
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0
  };

  // ================ Static TF (base_link -> lidar / imu) ================
  // lio_node 起動時に StaticTransformBroadcaster で 2 本 publish するための定義。
  // IESKF 内部用の extrinsic_*_imu_from_lidar とは別管理で、 yaml 上で値を合わせる
  // のは利用者の責任 (将来は base_link 起点に一本化する PR を予定)。
  std::string lidar_frame_id = "lidar_frame";
  std::string imu_frame_id = "imu_frame";
  // base_link -> lidar
  double base_link_to_lidar_translation_x = 0.0;
  double base_link_to_lidar_translation_y = 0.0;
  double base_link_to_lidar_translation_z = 0.0;
  std::vector<double> base_link_to_lidar_rotation_row_major = {
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0
  };
  // base_link -> imu
  double base_link_to_imu_translation_x = 0.0;
  double base_link_to_imu_translation_y = 0.0;
  double base_link_to_imu_translation_z = 0.0;
  std::vector<double> base_link_to_imu_rotation_row_major = {
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0
  };
  // false にすると static TF を出さない (外部 URDF / robot_state_publisher が出している前提のとき)
  bool publish_sensor_static_tf = true;

  // extrinsic の取得元:
  //   "config" → yaml の extrinsic_*_imu_from_lidar を IESKF に直接渡す (デフォルト)
  //   "tf"     → 起動時 1 回だけ TF tree から lookup(imu_frame_id, lidar_frame_id) し、
  //              その結果で extrinsic_*_imu_from_lidar を上書き。 失敗時は config に fallback
  // どちらでも IESKF への渡し方は同じ (T_imu_lidar = R + t)。
  std::string extrinsic_source = "config";
  // "tf" モードでの lookup 待機タイムアウト
  double extrinsic_tf_lookup_timeout_s = 2.0;

  // ================ PCD auto save ================
  // keyframe 採択ごとに「形成 map 全体」(active + finalized submap) を PCD ファイルに
  // 上書き保存する。 ノード稼働中、 常に最新の map が指定パスに残る。
  // map=voxel_keyframe_submap 以外では toFullMapPointCloud() が無いため no-op。
  // 重い書き込みなので必要に応じて keyframe 間隔を広げて頻度調整すること。
  bool enable_pcd_auto_save = false;
  // 空文字なら無効化 (enable_pcd_auto_save=true でも save しない)。
  std::string output_pcd_path = "";
  // binary mode (true) なら savePCDFileBinary、 false なら ascii (人間可読だが巨大)。
  bool output_pcd_binary = true;
  // 重力ベクトル (world frame, m/s^2)。ROS 標準は z-up = (0, 0, -9.81)。
  // 起動直後の静止 IMU 平均から自動推定する場合は auto_estimate_gravity=true。
  bool auto_estimate_gravity = true;
  int gravity_estimation_samples = 100;
  double gravity_world_x = 0.0;
  double gravity_world_y = 0.0;
  double gravity_world_z = -9.81;
};

struct LioBackends
{
  IPreprocessorPtr preprocessor;
  IPointCloudMapPtr point_cloud_map;
  IRegistrationPtr registration;
  IStateEstimatorPtr state_estimator;
  std::string summary;  // 1 行サマリ (ログ用)
};

// 文字列名がサポート外なら std::invalid_argument を投げる (起動時に検出させる)。
LioBackends createBackendsFromConfig(const LioBackendConfig & config);

}  // namespace pylot_lio

#endif  // PYLOT_LIO__FACTORY_HPP_
