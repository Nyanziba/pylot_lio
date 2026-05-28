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
  // voxel_random_sampling 用: サンプリング率 0〜1 (GLIM の randomgrid downsampling 相当)。
  // 各 voxel から round(点数 × rate) 点 (最低 1) を保持。 0 で従来の voxel 1 点、 1 で全点。
  double voxel_random_sampling_rate = 0.0;
  // voxel_random_sampling 用: 間引きを GPU (Metal) で行うか。 GPU 無効ビルド / デバイス無
  // では CPU に自動フォールバック。
  bool voxel_random_sampling_use_gpu = true;

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
  // keyframe ごとの [loop_diag] 診断ログを出すか。 切り分け時は true、 通常運用や
  // 高速 bag 処理でログを静かにしたいときは false。
  bool enable_loop_diag = true;
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
  // 並列バックエンド: "omp" (OpenMP) | "tbb" (Intel TBB)。
  // plain_gicp と small_gicp_(v)gicp の両方で共通に使う。 ビルドに TBB が含まれない
  // (PYLOT_LIO_HAS_TBB 未定義) 場合は "tbb" 指定でも OpenMP に自動フォールバックする。
  std::string registration_parallel_backend = "omp";

  // plain_gicp 固有のチューニング (small_gicp 系では無視される)。
  // Gauss-Newton の停止条件と Huber ロバスト重みの転換点を ROS パラメータから触れる。
  double registration_convergence_translation_m = 1e-4;
  double registration_convergence_rotation_rad = 1e-4;
  double registration_huber_threshold = 1.0;

  // plain_gicp 用: source 共分散 C_s (GICP distribution-to-distribution) の計算。
  // 近傍点数 k と平面性正則化の最小固有値 epsilon。 small_gicp 系では無視される。
  int registration_source_covariance_num_neighbors = 10;
  double registration_source_covariance_plane_epsilon = 1e-3;

  // metal_vgicp 用: GPU を使う最小 source 点数。 これ未満は CPU VGICP に自動切替
  // (GPU 起動オーバヘッド回避)。 0 で常に GPU。 他の registration では無視される。
  int registration_metal_gpu_min_points = 50000;

  // metal_vgicp 用: 多重解像度 (coarse-to-fine) VGICP。 levels=1 で単一解像度。
  int registration_metal_voxelmap_levels = 2;
  double registration_metal_voxelmap_scaling_factor = 2.0;

  // metal_vgicp 用: source 共分散推定 (前処理) を GPU で行うか。 true なら k 近傍探索 +
  // 平面正則化を Metal にオフロード (glim 相当の前処理 GPU 化)。 GPU 無効ビルド /
  // デバイス無では CPU 参照 (グリッド kNN) に自動フォールバック。 他 registration では無視。
  bool registration_metal_gpu_source_covariance = true;
  // GPU 共分散推定の近傍探索グリッドのセル一辺 [m]。 source 点群密度に対して
  // 「k 近傍が 27 近傍セルに収まる」 程度に取る。
  double registration_metal_source_covariance_cell_size_m = 0.5;

  // metal_vgicp 用: 地面平面 leveling 拘束。 IMU 重力が使えない (extrinsic 未知) ときに、
  // 各スキャンの地面法線を world-up (0,0,1) に合わせてロール/ピッチを pin し、 遠方地面の
  // お椀化 (ピッチドリフト) を抑える。 平地走行が前提。 他 registration では無視。
  bool registration_metal_enable_ground_constraint = false;
  double registration_metal_ground_constraint_weight = 1.0;
  double registration_metal_ground_band_m = 0.5;
  double registration_metal_ground_max_tilt_deg = 30.0;
  // damping: 1 align あたりの leveling 補正上限 [deg]。 0 で無制限 (hard)。
  double registration_metal_ground_max_correction_per_frame_deg = 1.0;
  // 振動ゲート: 前フレームの地面法線 (body) から角度差 > これ [deg] のフレームは
  // 「車体ピッチ振動中」 とみなして拘束をスキップ。 0 で振動ゲート無効。
  double registration_metal_ground_vibration_threshold_deg = 3.0;

  // plain_gicp 用: 縮退方向 Tikhonov 正則化 (X-ICP / sycl_points 流)。
  // 廊下や対称的な環境で回転/並進が拘束されないときに姿勢が暴れるのを防ぐ。
  bool enable_degenerate_regularization = false;
  double rotation_eigenvalue_threshold = 10.0;
  double translation_eigenvalue_threshold = 1.0;
  double regularization_base_factor = 1.0;

  // state estimator
  std::string state_estimator_name = "ieskf";    // ieskf | hgo | gicp_only

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
  // lidar_frame_id / imu_frame_id は extrinsic_source="tf" のときの TF lookup
  // (imu_frame <- lidar_frame) でのみ使う。 sensor static TF (base_link -> lidar/imu) の
  // publish はこのノードの責務ではない (外部の robot_state_publisher / URDF が出す前提)
  // ため、 base_link_to_lidar/imu_* や publish_sensor_static_tf は持たない。
  std::string lidar_frame_id = "lidar_frame";
  std::string imu_frame_id = "imu_frame";

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
