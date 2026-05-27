// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#ifndef PYLOT_LIO__LIO_NODE_HPP_
#define PYLOT_LIO__LIO_NODE_HPP_

#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/string.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/exceptions.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>

#include "pylot_lio/factory.hpp"
#include "pylot_lio/keyframe/distance_keyframe_selector.hpp"
#include "pylot_lio/livox_conversion.hpp"

#ifdef PYLOT_LIO_HAS_LIVOX_DRIVER
#include <livox_ros_driver2/msg/custom_msg.hpp>
#endif
#include "pylot_lio/loop/loop_closure_icp.hpp"
#include "pylot_lio/loop/pose_graph_optimizer.hpp"
#include "pylot_lio/loop/scan_context_loop_detector.hpp"
#include "pylot_lio/map/voxel_keyframe_submap_map.hpp"

using namespace std::chrono_literals;

namespace pylot_lio
{

class LioNode : public rclcpp::Node
{
public:
  LioNode()
  : rclcpp::Node("pylot_lio")
  {
    declareAndReadParameters();

    // 拡張: extrinsic_source="tf" のとき、 TF tree (例: robot_state_publisher が出している
    // base_link → lidar_frame / imu_frame の static TF) から T_imu_lidar を引き出し、
    // backend_config_ の extrinsic_*_imu_from_lidar を上書きしてから backends を生成する。
    // lookup 失敗時は WARN を出して yaml の値で続行する (= "config" 動作にフォールバック)。
    if (backend_config_.extrinsic_source == "tf") {
      tryOverrideExtrinsicFromTf();
    }

    backends_ = createBackendsFromConfig(backend_config_);
    RCLCPP_INFO(get_logger(), "Backends: %s", backends_.summary.c_str());

    // 初期状態は単位姿勢、速度 0。/initialpose 経由の再初期化は省略 (将来拡張)。
    RobotState initial_state;
    backends_.state_estimator->initialize(initial_state);

    // 高レート publisher (例: 30Hz Velodyne の bag) で SensorDataQoS (depth=5, best_effort)
    // だと callback の処理時間が長くなったとき DDS 側 で配信が止まる現象を実機で確認。
    // 代わりに reliable + 深 buffer を使うことで、処理が遅れても DDS はキューに溜めて配信を続ける。
    rclcpp::QoS sensor_qos(rclcpp::KeepLast(100));
    sensor_qos.reliable();
    sensor_qos.durability_volatile();

    // 入力形式に応じて subscriber を 1 種類だけ生成する。 「両方同時購読」はしないことで
    // 同じセンサからの重複処理を物理的に防ぐ (= 同一 keyframe が二重に挿入される事故を防ぐ)。
    if (input_cloud_format_ == "livox_custom") {
#ifdef PYLOT_LIO_HAS_LIVOX_DRIVER
      livox_custom_subscription_ =
        create_subscription<livox_ros_driver2::msg::CustomMsg>(
          input_cloud_topic_, sensor_qos,
          std::bind(&LioNode::onLivoxCustomCloud, this, std::placeholders::_1));
      RCLCPP_INFO(get_logger(),
        "Cloud format: livox_custom (livox_ros_driver2::msg::CustomMsg)");
#else
      throw std::runtime_error(
        "input_cloud_format='livox_custom' was requested but pylot_lio was "
        "built without livox_ros_driver2 support. "
        "Install livox_ros_driver2 and rebuild, or use 'pointcloud2'.");
#endif
    } else if (input_cloud_format_ == "pointcloud2") {
      cloud_subscription_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        input_cloud_topic_, sensor_qos,
        std::bind(&LioNode::onLidarCloud, this, std::placeholders::_1));
      RCLCPP_INFO(get_logger(),
        "Cloud format: pointcloud2 (sensor_msgs::msg::PointCloud2)");
    } else {
      throw std::invalid_argument(
        "Unknown input_cloud_format: '" + input_cloud_format_ +
        "'. Valid options: 'pointcloud2', 'livox_custom'.");
    }
    if (backends_.state_estimator->usesImu()) {
      imu_subscription_ = create_subscription<sensor_msgs::msg::Imu>(
        input_imu_topic_, sensor_qos,
        std::bind(&LioNode::onImu, this, std::placeholders::_1));
      RCLCPP_INFO(get_logger(),
        "Subscribing: cloud='%s', imu='%s'",
        input_cloud_topic_.c_str(), input_imu_topic_.c_str());
    } else {
      RCLCPP_INFO(get_logger(),
        "Subscribing: cloud='%s' (estimator does not use IMU; skipping IMU sub)",
        input_cloud_topic_.c_str());
    }

    odom_publisher_ = create_publisher<nav_msgs::msg::Odometry>(output_odom_topic_, 10);
    cloud_publisher_ =
      create_publisher<sensor_msgs::msg::PointCloud2>(output_cloud_topic_, 10);
    diag_publisher_ = create_publisher<std_msgs::msg::String>(output_diag_topic_, 10);
    keyframe_path_publisher_ =
      create_publisher<nav_msgs::msg::Path>(output_keyframes_topic_, 10);
    // 「形成したマップ全体」publisher。 finalized submap + active submap を連結した
    // 大きな点群を 5 秒 (デフォルト) に 1 回出す。 サイズが大きいので Reliable QoS
    // + 浅いキュー (= 古い未送信は捨てる) で。
    full_map_publisher_ =
      create_publisher<sensor_msgs::msg::PointCloud2>(output_full_map_topic_, 1);
    if (full_map_publish_period_s_ > 0.0) {
      full_map_timer_ = create_wall_timer(
        std::chrono::milliseconds(
          static_cast<int64_t>(full_map_publish_period_s_ * 1000.0)),
        std::bind(&LioNode::onFullMapTimer, this));
    }
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    // sensor static TF (base_link -> lidar/imu) はこのノードでは publish しない。
    // それを出す責務は外部 (robot_state_publisher / URDF) にあるため。

    // Keyframe selector を生成 (現状 1 種類だけだが、抽象 IF 経由で差し替え可能)。
    DistanceKeyframeSelector::Config keyframe_config;
    keyframe_config.min_translation_m = keyframe_min_translation_m_;
    keyframe_config.min_rotation_rad = keyframe_min_rotation_rad_;
    keyframe_config.max_scans_between_keyframes = keyframe_max_scans_between_;
    keyframe_selector_ =
      std::make_unique<DistanceKeyframeSelector>(keyframe_config);
    RCLCPP_INFO(get_logger(), "Keyframe: %s",
      keyframe_selector_->describe().c_str());

    // Loop closure detector (Scan Context)。 enable_loop_detection=false の場合は
    // 構築しない。 検出 publish 用の publisher は detector の有無にかかわらず作っておく
    // (購読側を変えずに済む)。
    loop_candidates_publisher_ =
      create_publisher<std_msgs::msg::String>(output_loop_candidates_topic_, 10);
    if (backend_config_.enable_loop_detection) {
      ScanContextLoopDetector::Config loop_config;
      loop_config.descriptor.num_rings = backend_config_.loop_num_rings;
      loop_config.descriptor.num_sectors = backend_config_.loop_num_sectors;
      loop_config.descriptor.max_radius_m = backend_config_.loop_max_radius_m;
      loop_config.descriptor.min_radius_m = backend_config_.loop_min_radius_m;
      loop_config.ring_key_top_k = backend_config_.loop_ring_key_top_k;
      loop_config.score_threshold = backend_config_.loop_score_threshold;
      loop_config.exclude_recent_kf = backend_config_.loop_exclude_recent_kf;
      loop_detector_ = std::make_unique<ScanContextLoopDetector>(loop_config);
      RCLCPP_INFO(get_logger(), "Loop: %s", loop_detector_->describe().c_str());
    } else {
      RCLCPP_INFO(get_logger(), "Loop detection: disabled");
    }

    // Pose Graph Optimization (GTSAM)。 起動時に最初の keyframe (id=0) を world に
    // ピン留めする prior factor を入れるが、 まだ最初の keyframe が来ていないので
    // 実際の addKeyframePrior は最初の onLidarCloud 内で 1 回だけ行う。
    if (backend_config_.enable_pose_graph_optimization) {
      pose_graph_optimizer_ = std::make_unique<PoseGraphOptimizer>();
      RCLCPP_INFO(get_logger(), "PGO: %s",
        pose_graph_optimizer_->describe().c_str());
    } else {
      RCLCPP_INFO(get_logger(), "PGO: disabled");
    }

    // 起動から N 秒以内に IMU / 点群が来なかったら警告する。
    watchdog_timer_ = create_wall_timer(
      std::chrono::seconds(watchdog_interval_s_),
      std::bind(&LioNode::onWatchdogTimer, this));
  }

  // ============================================================
  // オフライン rosbag リーダ (lio_rosbag) 用の直接投入入口。
  // DDS を経由せず、 bag から読んだメッセージを通常の購読コールバックと同じ処理経路へ
  // 渡す。 これにより再生レートに依存せず (= ハードが許す限り高速に) 全フレームを
  // 取りこぼしなく処理できる (GLIM の glim_rosbag 相当)。
  // ============================================================
  void injectImu(const sensor_msgs::msg::Imu::SharedPtr & message) { onImu(message); }
  void injectLidarCloud(const sensor_msgs::msg::PointCloud2::SharedPtr & message)
  {
    onLidarCloud(message);
  }
#ifdef PYLOT_LIO_HAS_LIVOX_DRIVER
  void injectLivoxCustomCloud(
    const livox_ros_driver2::msg::CustomMsg::SharedPtr & message)
  {
    onLivoxCustomCloud(message);
  }
#endif

private:
  void declareAndReadParameters()
  {
    // ROS パラメータ → backend_config。ROS 名は kebab-case ではなく snake_case。
    backend_config_.preprocessor_name =
      declare_parameter<std::string>("preprocessor", backend_config_.preprocessor_name);
    backend_config_.voxel_grid_size_m =
      declare_parameter<double>("voxel_grid_size_m", backend_config_.voxel_grid_size_m);
    backend_config_.random_sampling_target_count =
      declare_parameter<int>(
        "random_sampling_target_count", backend_config_.random_sampling_target_count);
    backend_config_.voxel_random_sampling_seed =
      declare_parameter<int>(
        "voxel_random_sampling_seed",
        backend_config_.voxel_random_sampling_seed);

    backend_config_.map_name =
      declare_parameter<std::string>("map", backend_config_.map_name);
    backend_config_.map_voxel_size_m =
      declare_parameter<double>("map_voxel_size_m", backend_config_.map_voxel_size_m);
    backend_config_.map_min_points_per_cell =
      declare_parameter<int>(
        "map_min_points_per_cell", backend_config_.map_min_points_per_cell);
    backend_config_.map_max_total_cells = static_cast<std::size_t>(
      declare_parameter<int>(
        "map_max_total_cells", static_cast<int>(backend_config_.map_max_total_cells)));
    backend_config_.voxel_random_map_max_points_per_voxel =
      declare_parameter<int>(
        "voxel_random_map_max_points_per_voxel",
        backend_config_.voxel_random_map_max_points_per_voxel);
    backend_config_.voxel_random_map_k_nearest_for_covariance =
      declare_parameter<int>(
        "voxel_random_map_k_nearest_for_covariance",
        backend_config_.voxel_random_map_k_nearest_for_covariance);

    backend_config_.submap_window_size =
      declare_parameter<int>("submap_window_size", backend_config_.submap_window_size);
    backend_config_.submap_finalize_size =
      declare_parameter<int>(
        "submap_finalize_size", backend_config_.submap_finalize_size);
    backend_config_.submap_max_points_per_cell =
      declare_parameter<int>(
        "submap_max_points_per_cell", backend_config_.submap_max_points_per_cell);
    backend_config_.submap_k_nearest_for_covariance =
      declare_parameter<int>(
        "submap_k_nearest_for_covariance",
        backend_config_.submap_k_nearest_for_covariance);
    backend_config_.submap_cell_sampling_mode =
      declare_parameter<std::string>(
        "submap_cell_sampling_mode",
        backend_config_.submap_cell_sampling_mode);
    backend_config_.submap_random_seed =
      declare_parameter<int>(
        "submap_random_seed", backend_config_.submap_random_seed);

    backend_config_.enable_loop_detection =
      declare_parameter<bool>(
        "enable_loop_detection", backend_config_.enable_loop_detection);
    backend_config_.loop_num_rings =
      declare_parameter<int>("loop_num_rings", backend_config_.loop_num_rings);
    backend_config_.loop_num_sectors =
      declare_parameter<int>("loop_num_sectors", backend_config_.loop_num_sectors);
    backend_config_.loop_max_radius_m =
      declare_parameter<double>(
        "loop_max_radius_m", backend_config_.loop_max_radius_m);
    backend_config_.loop_min_radius_m =
      declare_parameter<double>(
        "loop_min_radius_m", backend_config_.loop_min_radius_m);
    backend_config_.loop_ring_key_top_k =
      declare_parameter<int>(
        "loop_ring_key_top_k", backend_config_.loop_ring_key_top_k);
    backend_config_.loop_score_threshold =
      declare_parameter<double>(
        "loop_score_threshold", backend_config_.loop_score_threshold);
    backend_config_.loop_exclude_recent_kf =
      declare_parameter<int>(
        "loop_exclude_recent_kf", backend_config_.loop_exclude_recent_kf);
    backend_config_.loop_icp_max_fitness_score =
      declare_parameter<double>(
        "loop_icp_max_fitness_score",
        backend_config_.loop_icp_max_fitness_score);
    backend_config_.loop_icp_max_translation_jump_m =
      declare_parameter<double>(
        "loop_icp_max_translation_jump_m",
        backend_config_.loop_icp_max_translation_jump_m);
    backend_config_.loop_match_cooldown_keyframes =
      declare_parameter<int>(
        "loop_match_cooldown_keyframes",
        backend_config_.loop_match_cooldown_keyframes);
    output_loop_candidates_topic_ =
      declare_parameter<std::string>(
        "output_loop_candidates_topic", "/lio/loop_candidates");

    backend_config_.enable_pose_graph_optimization =
      declare_parameter<bool>(
        "enable_pose_graph_optimization",
        backend_config_.enable_pose_graph_optimization);
    backend_config_.pgo_optimize_every_n_keyframes =
      declare_parameter<int>(
        "pgo_optimize_every_n_keyframes",
        backend_config_.pgo_optimize_every_n_keyframes);
    backend_config_.loop_icp_voxel_size_m =
      declare_parameter<double>(
        "loop_icp_voxel_size_m", backend_config_.loop_icp_voxel_size_m);
    backend_config_.loop_icp_max_correspondence_m =
      declare_parameter<double>(
        "loop_icp_max_correspondence_m",
        backend_config_.loop_icp_max_correspondence_m);
    backend_config_.loop_icp_max_iterations =
      declare_parameter<int>(
        "loop_icp_max_iterations", backend_config_.loop_icp_max_iterations);
    backend_config_.pgo_odometry_sigma_rot_rad =
      declare_parameter<double>(
        "pgo_odometry_sigma_rot_rad",
        backend_config_.pgo_odometry_sigma_rot_rad);
    backend_config_.pgo_odometry_sigma_trans_m =
      declare_parameter<double>(
        "pgo_odometry_sigma_trans_m",
        backend_config_.pgo_odometry_sigma_trans_m);
    backend_config_.pgo_loop_sigma_rot_rad =
      declare_parameter<double>(
        "pgo_loop_sigma_rot_rad", backend_config_.pgo_loop_sigma_rot_rad);
    backend_config_.pgo_loop_sigma_trans_m =
      declare_parameter<double>(
        "pgo_loop_sigma_trans_m", backend_config_.pgo_loop_sigma_trans_m);

    backend_config_.registration_name =
      declare_parameter<std::string>("registration", backend_config_.registration_name);
    backend_config_.registration_max_correspondence_m =
      declare_parameter<double>(
        "registration_max_correspondence_m",
        backend_config_.registration_max_correspondence_m);
    backend_config_.registration_num_threads =
      declare_parameter<int>(
        "registration_num_threads", backend_config_.registration_num_threads);
    backend_config_.registration_max_iterations =
      declare_parameter<int>(
        "registration_max_iterations", backend_config_.registration_max_iterations);
    backend_config_.registration_parallel_backend =
      declare_parameter<std::string>(
        "registration_parallel_backend",
        backend_config_.registration_parallel_backend);
    backend_config_.registration_convergence_translation_m =
      declare_parameter<double>(
        "registration_convergence_translation_m",
        backend_config_.registration_convergence_translation_m);
    backend_config_.registration_convergence_rotation_rad =
      declare_parameter<double>(
        "registration_convergence_rotation_rad",
        backend_config_.registration_convergence_rotation_rad);
    backend_config_.registration_huber_threshold =
      declare_parameter<double>(
        "registration_huber_threshold",
        backend_config_.registration_huber_threshold);
    backend_config_.registration_source_covariance_num_neighbors =
      declare_parameter<int>(
        "registration_source_covariance_num_neighbors",
        backend_config_.registration_source_covariance_num_neighbors);
    backend_config_.registration_source_covariance_plane_epsilon =
      declare_parameter<double>(
        "registration_source_covariance_plane_epsilon",
        backend_config_.registration_source_covariance_plane_epsilon);
    backend_config_.registration_metal_gpu_min_points =
      declare_parameter<int>(
        "registration_metal_gpu_min_points",
        backend_config_.registration_metal_gpu_min_points);
    backend_config_.registration_metal_voxelmap_levels =
      declare_parameter<int>(
        "registration_metal_voxelmap_levels",
        backend_config_.registration_metal_voxelmap_levels);
    backend_config_.registration_metal_voxelmap_scaling_factor =
      declare_parameter<double>(
        "registration_metal_voxelmap_scaling_factor",
        backend_config_.registration_metal_voxelmap_scaling_factor);
    backend_config_.enable_degenerate_regularization =
      declare_parameter<bool>(
        "enable_degenerate_regularization",
        backend_config_.enable_degenerate_regularization);
    backend_config_.rotation_eigenvalue_threshold =
      declare_parameter<double>(
        "rotation_eigenvalue_threshold",
        backend_config_.rotation_eigenvalue_threshold);
    backend_config_.translation_eigenvalue_threshold =
      declare_parameter<double>(
        "translation_eigenvalue_threshold",
        backend_config_.translation_eigenvalue_threshold);
    backend_config_.regularization_base_factor =
      declare_parameter<double>(
        "regularization_base_factor",
        backend_config_.regularization_base_factor);

    backend_config_.state_estimator_name =
      declare_parameter<std::string>(
        "state_estimator", backend_config_.state_estimator_name);

    backend_config_.imu_acceleration_scale =
      declare_parameter<double>(
        "imu_acceleration_scale", backend_config_.imu_acceleration_scale);

    backend_config_.gravity_norm =
      declare_parameter<double>("gravity_norm", backend_config_.gravity_norm);
    backend_config_.extrinsic_translation_imu_from_lidar_x =
      declare_parameter<double>(
        "extrinsic_translation_imu_from_lidar_x",
        backend_config_.extrinsic_translation_imu_from_lidar_x);
    backend_config_.extrinsic_translation_imu_from_lidar_y =
      declare_parameter<double>(
        "extrinsic_translation_imu_from_lidar_y",
        backend_config_.extrinsic_translation_imu_from_lidar_y);
    backend_config_.extrinsic_translation_imu_from_lidar_z =
      declare_parameter<double>(
        "extrinsic_translation_imu_from_lidar_z",
        backend_config_.extrinsic_translation_imu_from_lidar_z);
    backend_config_.extrinsic_rotation_imu_from_lidar_row_major =
      declare_parameter<std::vector<double>>(
        "extrinsic_rotation_imu_from_lidar_row_major",
        backend_config_.extrinsic_rotation_imu_from_lidar_row_major);

    // lidar_frame_id / imu_frame_id は extrinsic_source=tf のときの TF lookup
    // (imu_frame <- lidar_frame) でのみ使う。 sensor static TF (base_link -> lidar/imu)
    // の publish はこのノードの責務ではないため行わない (外部の robot_state_publisher /
    // URDF が出す前提)。 従って base_link_to_lidar/imu_* や publish_sensor_static_tf は持たない。
    backend_config_.lidar_frame_id =
      declare_parameter<std::string>(
        "lidar_frame_id", backend_config_.lidar_frame_id);
    backend_config_.imu_frame_id =
      declare_parameter<std::string>(
        "imu_frame_id", backend_config_.imu_frame_id);

    backend_config_.extrinsic_source =
      declare_parameter<std::string>(
        "extrinsic_source", backend_config_.extrinsic_source);
    backend_config_.extrinsic_tf_lookup_timeout_s =
      declare_parameter<double>(
        "extrinsic_tf_lookup_timeout_s",
        backend_config_.extrinsic_tf_lookup_timeout_s);

    backend_config_.enable_pcd_auto_save =
      declare_parameter<bool>(
        "enable_pcd_auto_save", backend_config_.enable_pcd_auto_save);
    backend_config_.output_pcd_path =
      declare_parameter<std::string>(
        "output_pcd_path", backend_config_.output_pcd_path);
    backend_config_.output_pcd_binary =
      declare_parameter<bool>(
        "output_pcd_binary", backend_config_.output_pcd_binary);

    backend_config_.auto_estimate_gravity =
      declare_parameter<bool>(
        "auto_estimate_gravity", backend_config_.auto_estimate_gravity);
    backend_config_.gravity_estimation_samples =
      declare_parameter<int>(
        "gravity_estimation_samples", backend_config_.gravity_estimation_samples);
    backend_config_.gravity_world_x =
      declare_parameter<double>("gravity_world_x", backend_config_.gravity_world_x);
    backend_config_.gravity_world_y =
      declare_parameter<double>("gravity_world_y", backend_config_.gravity_world_y);
    backend_config_.gravity_world_z =
      declare_parameter<double>("gravity_world_z", backend_config_.gravity_world_z);

    world_frame_id_ = declare_parameter<std::string>("world_frame_id", "world");
    body_frame_id_ = declare_parameter<std::string>("body_frame_id", "base_link");
    input_cloud_topic_ =
      declare_parameter<std::string>("input_cloud_topic", "/livox/lidar");
    // 入力点群フォーマット: "pointcloud2" (デフォルト) | "livox_custom"
    // livox_custom は livox_ros_driver2::msg::CustomMsg を購読し、 LivoxRawPoint
    // 経由で内部 PCL 点群に変換してから既存パイプラインに流す。
    input_cloud_format_ =
      declare_parameter<std::string>("input_cloud_format", "pointcloud2");
    input_imu_topic_ =
      declare_parameter<std::string>("input_imu_topic", "/livox/imu");
    output_odom_topic_ =
      declare_parameter<std::string>("output_odom_topic", "/lio/odom");
    output_cloud_topic_ =
      declare_parameter<std::string>("output_cloud_topic", "/lio/cloud_world");
    output_diag_topic_ = declare_parameter<std::string>("output_diag_topic", "/lio/diag");
    output_keyframes_topic_ =
      declare_parameter<std::string>("output_keyframes_topic", "/lio/keyframes");
    output_full_map_topic_ =
      declare_parameter<std::string>("output_full_map_topic", "/lio/full_map");
    full_map_publish_period_s_ =
      declare_parameter<double>("full_map_publish_period_s", 5.0);
    publish_tf_ = declare_parameter<bool>("publish_tf", true);

    keyframe_min_translation_m_ =
      declare_parameter<double>("keyframe_min_translation_m", 1.0);
    keyframe_min_rotation_rad_ =
      declare_parameter<double>("keyframe_min_rotation_rad", 0.2);
    keyframe_max_scans_between_ =
      declare_parameter<int>("keyframe_max_scans_between_keyframes", 50);
  }

  void onImu(const sensor_msgs::msg::Imu::SharedPtr message)
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    ++imu_callback_count_;
    last_imu_wall_time_ = std::chrono::steady_clock::now();
    if (imu_callback_count_ == 1) {
      const double raw_norm = std::sqrt(
        message->linear_acceleration.x * message->linear_acceleration.x +
        message->linear_acceleration.y * message->linear_acceleration.y +
        message->linear_acceleration.z * message->linear_acceleration.z);
      const double scaled_norm = raw_norm * backend_config_.imu_acceleration_scale;
      RCLCPP_INFO(get_logger(),
        "First IMU sample on '%s': raw acc=(%.3f, %.3f, %.3f), norm=%.3f m/s^2, "
        "gyro=(%.4f, %.4f, %.4f) rad/s, imu_acceleration_scale=%.4f -> scaled_norm=%.3f. "
        "Static-only scaled_norm should be ~9.81; if raw_norm ~= 1.0 set scale=9.80665.",
        input_imu_topic_.c_str(),
        message->linear_acceleration.x, message->linear_acceleration.y,
        message->linear_acceleration.z, raw_norm,
        message->angular_velocity.x, message->angular_velocity.y,
        message->angular_velocity.z,
        backend_config_.imu_acceleration_scale, scaled_norm);
    }
    if (!backends_.state_estimator->isInitialized()) {
      return;
    }
    ImuSample imu_sample;
    imu_sample.timestamp_ns =
      static_cast<int64_t>(message->header.stamp.sec) * 1000000000LL
      + static_cast<int64_t>(message->header.stamp.nanosec);
    // Livox driver 等は acc を g 単位で出してくるため、 ここでスケール補正してから
    // estimator に渡す (imu_acceleration_scale=9.80665 で g → m/s²)。
    imu_sample.linear_acceleration_mps2 = Eigen::Vector3d(
      message->linear_acceleration.x * backend_config_.imu_acceleration_scale,
      message->linear_acceleration.y * backend_config_.imu_acceleration_scale,
      message->linear_acceleration.z * backend_config_.imu_acceleration_scale);
    imu_sample.angular_velocity_rps = Eigen::Vector3d(
      message->angular_velocity.x,
      message->angular_velocity.y,
      message->angular_velocity.z);
    backends_.state_estimator->predictWithImu(imu_sample);
  }

  void appendScanToMapAndKeyframes(
    const PointCloud & scan_cloud_body,
    const Eigen::Isometry3d & pose_world_body,
    const rclcpp::Time & stamp)
  {
    PointCloud scan_in_world;
    scan_in_world.points.reserve(scan_cloud_body.size());
    for (const Point & body_point : scan_cloud_body.points) {
      if (!std::isfinite(body_point.x) || !std::isfinite(body_point.y) ||
          !std::isfinite(body_point.z))
      {
        continue;
      }
      const Eigen::Vector3d body_position(body_point.x, body_point.y, body_point.z);
      const Eigen::Vector3d world_position = pose_world_body * body_position;
      Point world_point;
      world_point.x = static_cast<float>(world_position.x());
      world_point.y = static_cast<float>(world_position.y());
      world_point.z = static_cast<float>(world_position.z());
      world_point.intensity = body_point.intensity;
      scan_in_world.points.push_back(world_point);
    }
    backends_.point_cloud_map->insertScan(scan_in_world, pose_world_body);

    // Path に keyframe を追加して publish。
    geometry_msgs::msg::PoseStamped pose_stamped;
    pose_stamped.header.stamp = stamp;
    pose_stamped.header.frame_id = world_frame_id_;
    pose_stamped.pose.position.x = pose_world_body.translation().x();
    pose_stamped.pose.position.y = pose_world_body.translation().y();
    pose_stamped.pose.position.z = pose_world_body.translation().z();
    const Eigen::Quaterniond quaternion(pose_world_body.linear());
    pose_stamped.pose.orientation.x = quaternion.x();
    pose_stamped.pose.orientation.y = quaternion.y();
    pose_stamped.pose.orientation.z = quaternion.z();
    pose_stamped.pose.orientation.w = quaternion.w();
    keyframe_path_msg_.header.stamp = stamp;
    keyframe_path_msg_.header.frame_id = world_frame_id_;
    keyframe_path_msg_.poses.push_back(pose_stamped);
    keyframe_path_publisher_->publish(keyframe_path_msg_);
  }

  void onLidarCloud(const sensor_msgs::msg::PointCloud2::SharedPtr message)
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    ++cloud_callback_count_;
    last_cloud_wall_time_ = std::chrono::steady_clock::now();
    if (cloud_callback_count_ == 1) {
      RCLCPP_INFO(get_logger(),
        "First point cloud received on '%s' (width=%u, height=%u, fields=%zu)",
        input_cloud_topic_.c_str(),
        message->width, message->height, message->fields.size());
    }
    if (!backends_.state_estimator->isInitialized()) {
      return;
    }

    auto raw_cloud = std::make_shared<PointCloud>();
    pcl::fromROSMsg(*message, *raw_cloud);
    if (raw_cloud->empty()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
        "Converted point cloud is empty (PointCloud2 fields lacked x/y/z?)");
      return;
    }
    processRawCloudLocked(raw_cloud, message->header.stamp);
  }

#ifdef PYLOT_LIO_HAS_LIVOX_DRIVER
  void onLivoxCustomCloud(
    const livox_ros_driver2::msg::CustomMsg::SharedPtr message)
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    ++cloud_callback_count_;
    last_cloud_wall_time_ = std::chrono::steady_clock::now();
    if (cloud_callback_count_ == 1) {
      RCLCPP_INFO(get_logger(),
        "First Livox CustomMsg received on '%s' (point_num=%u)",
        input_cloud_topic_.c_str(), message->point_num);
    }
    if (!backends_.state_estimator->isInitialized()) {
      return;
    }

    // CustomMsg.points を内部の素な型 LivoxRawPoint に詰め替え、 純関数の
    // convertLivoxRawPoints へ渡す。 こうすることで変換のテストが ROS 非依存に書ける。
    std::vector<LivoxRawPoint> raw_points;
    raw_points.reserve(message->points.size());
    for (const auto & livox_point : message->points) {
      LivoxRawPoint raw_point;
      raw_point.x = livox_point.x;
      raw_point.y = livox_point.y;
      raw_point.z = livox_point.z;
      raw_point.reflectivity = livox_point.reflectivity;
      raw_point.offset_time_ns = livox_point.offset_time;
      raw_points.push_back(raw_point);
    }
    auto raw_cloud = std::make_shared<PointCloud>(convertLivoxRawPoints(raw_points));
    if (raw_cloud->empty()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
        "Converted Livox CustomMsg is empty (point_num=0?)");
      return;
    }
    processRawCloudLocked(raw_cloud, message->header.stamp);
  }
#endif

  // PointCloud2 / CustomMsg のいずれから来た点群でも、 「PCL 点群が手元に揃って
  // state_estimator が初期化済」の段階に達したらここに合流する。
  // 呼び出し元が state_mutex_ をロックしている前提。
  void processRawCloudLocked(
    const PointCloudPtr & raw_cloud, const rclcpp::Time & stamp)
  {
    auto preprocessed_cloud = backends_.preprocessor->process(raw_cloud);
    backends_.state_estimator->updateWithScan(
      *preprocessed_cloud, *backends_.point_cloud_map, *backends_.registration,
      stamp.nanoseconds());

    // Keyframe ゲート: 直前 keyframe との SE(3) 差分が閾値を超えたとき (= 初回・大きく移動・回転)
    // だけマップに挿入する。停止中は挿入しないので重複点群の肥大化を防げる。
    const auto current_pose = backends_.state_estimator->getState().pose_world_body;
    if (keyframe_selector_->shouldCreateKeyframe(current_pose)) {
      const uint32_t this_keyframe_id =
        static_cast<uint32_t>(keyframe_selector_->keyframeCount());
      appendScanToMapAndKeyframes(
        *preprocessed_cloud, current_pose, stamp);
      keyframe_selector_->commit(current_pose);

      // Scan Context による loop 検出: body 系の preprocessed_cloud で descriptor を計算。
      // body 系を使うのは yaw 推定を意味のある量にするため。
      bool loop_detected_this_keyframe = false;
      ScanContextLoopDetector::LoopCandidate detected_candidate;
      if (loop_detector_) {
        const auto query_result = loop_detector_->addKeyframeAndQuery(
          this_keyframe_id, *preprocessed_cloud, current_pose);
        if (query_result.detected) {
          publishLoopCandidate(query_result.candidate, stamp);
          loop_detected_this_keyframe = true;
          detected_candidate = query_result.candidate;
        }
        // 「Loop candidate が出ない」ときの切り分け用の診断ログ。 keyframe ごとに 1 行。
        //   - cand_after_filter=0 → 検索対象がゼロ (DB が空 or 全部 recent_kf で除外)
        //       → loop_exclude_recent_kf を下げる、 もしくは走行が浅い (keyframe 不足)
        //   - cand_after_filter>0 で best_dist > threshold → 「似た場所が見つからない」
        //       → loop_score_threshold を緩める (例 0.10 → 0.20)
        //   - best_dist < threshold だが detected=false → ロジックバグ (要報告)
        const char * detected_str = query_result.detected ? "DETECTED" : "no-loop";
        RCLCPP_INFO(get_logger(),
          "[loop_diag] kf=%u db=%zu cand_after_filter=%d "
          "best_dist=%.4f (thr=%.4f) best_match_kf=%u -> %s",
          this_keyframe_id,
          query_result.database_size_after,
          query_result.num_candidates_after_filter,
          query_result.best_distance,
          backend_config_.loop_score_threshold,
          query_result.best_match_kf_id,
          detected_str);
      }

      // PCD 自動保存: keyframe 採択のタイミングで「形成 map 全体」を 1 ファイルに上書き。
      // 重い書き込み (binary でも数十 MB / 数百ms) なので、 走行中に keyframe が高頻度で
      // 採択される設定だと IO 詰まりリスクあり。 必要なら keyframe_min_translation_m を
      // 広げて頻度を下げること。
      if (backend_config_.enable_pcd_auto_save &&
          !backend_config_.output_pcd_path.empty())
      {
        saveFullMapPcd();
      }

      // PGO への factor 投入と必要に応じた optimize。 estimator 結果を odometry factor
      // として積み、 loop 検出時には ICP 精密化 → loop factor 追加 → 即 optimize する。
      if (pose_graph_optimizer_) {
        addKeyframeToPoseGraph(
          this_keyframe_id, current_pose, *preprocessed_cloud);
        bool loop_factor_accepted = false;
        if (loop_detected_this_keyframe) {
          loop_factor_accepted =
            addLoopFactorWithIcpRefinement(detected_candidate, *preprocessed_cloud);
        }
        ++keyframes_since_last_pgo_;
        const bool should_optimize_periodically =
          keyframes_since_last_pgo_ >=
            backend_config_.pgo_optimize_every_n_keyframes;
        // 「Scan Context は当たったが ICP ゲートで弾いた」場合は optimize を急がない。
        // 偽 loop を投入しないので odometry だけで進める方が安定する。
        if (loop_factor_accepted || should_optimize_periodically) {
          runPoseGraphOptimizeAndApply(this_keyframe_id);
          keyframes_since_last_pgo_ = 0;
        }
      }
    }

    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000,
      "scan #%lu: raw=%zu, preprocessed=%zu, map_cells=%zu, keyframes=%zu, time_ms=%.1f",
      static_cast<unsigned long>(cloud_callback_count_),
      raw_cloud->size(),
      preprocessed_cloud->size(),
      backends_.point_cloud_map->size(),
      keyframe_selector_->keyframeCount(),
      backends_.state_estimator->getDiagnostics().processing_time_ms);

    publishOutputs(stamp);
  }

  // 新規 keyframe を pose graph に追加する。
  //  - 最初の keyframe (id=0) は world に固定する prior factor
  //  - 2 つ目以降は前 keyframe との relative transform を odometry factor として追加
  // いずれも optimizer の initial estimate にも入れる。
  void addKeyframeToPoseGraph(
    uint32_t keyframe_id,
    const Eigen::Isometry3d & pose_world_body,
    const PointCloud & scan_body)
  {
    pose_graph_optimizer_->addKeyframeInitialEstimate(keyframe_id, pose_world_body);
    if (!has_first_keyframe_) {
      pose_graph_optimizer_->addKeyframePrior(keyframe_id, pose_world_body);
      has_first_keyframe_ = true;
    } else {
      const Eigen::Isometry3d relative_transform =
        previous_keyframe_pose_.inverse() * pose_world_body;
      PoseGraphOptimizer::NoiseSigmas odometry_sigmas;
      odometry_sigmas.rot_x_rad = backend_config_.pgo_odometry_sigma_rot_rad;
      odometry_sigmas.rot_y_rad = backend_config_.pgo_odometry_sigma_rot_rad;
      odometry_sigmas.rot_z_rad = backend_config_.pgo_odometry_sigma_rot_rad;
      odometry_sigmas.trans_x_m = backend_config_.pgo_odometry_sigma_trans_m;
      odometry_sigmas.trans_y_m = backend_config_.pgo_odometry_sigma_trans_m;
      odometry_sigmas.trans_z_m = backend_config_.pgo_odometry_sigma_trans_m;
      pose_graph_optimizer_->addOdometryFactor(
        previous_keyframe_id_, keyframe_id, relative_transform, odometry_sigmas);
    }
    previous_keyframe_id_ = keyframe_id;
    previous_keyframe_pose_ = pose_world_body;
    keyframe_clouds_body_[keyframe_id] = scan_body;
  }

  // Loop 検出時、 query/match keyframe の body 系点群を ICP で重ね合わせ、 精密化した
  // SE(3) 拘束を pose graph に投入する。 ICP の初期推定は「query/match の world pose
  // 差分」を使う (Scan Context yaw は既に query 姿勢に近い世界推定値に反映されている)。
  // 戻り値: PGO に loop factor を実際に投入したか (= ゲートを全部通ったか)。
  bool addLoopFactorWithIcpRefinement(
    const ScanContextLoopDetector::LoopCandidate & candidate,
    const PointCloud & query_cloud_body)
  {
    auto match_cloud_it = keyframe_clouds_body_.find(candidate.match_kf_id);
    if (match_cloud_it == keyframe_clouds_body_.end()) {
      RCLCPP_WARN(get_logger(),
        "Loop ICP skipped: match_kf=%u cloud not in cache",
        candidate.match_kf_id);
      return false;
    }

    // ゲート (a): 同じ match_kf に短期間で連続マッチしたら cooldown でスキップ。
    // 連続採用すると pose graph が同方向の loop factor を多重投入し、 ISAM2 の
    // linear system が ill-conditioned になりやすい。
    const int cooldown = backend_config_.loop_match_cooldown_keyframes;
    if (cooldown > 0 && last_loop_match_kf_id_.has_value() &&
        *last_loop_match_kf_id_ == candidate.match_kf_id &&
        static_cast<int>(candidate.query_kf_id - last_loop_query_kf_id_) <
        cooldown)
    {
      RCLCPP_INFO(get_logger(),
        "Loop skipped (cooldown): same match_kf=%u within %d keyframes",
        candidate.match_kf_id, cooldown);
      return false;
    }

    const Eigen::Isometry3d initial_transform_query_to_match =
      candidate.match_pose_world_body.inverse() * candidate.query_pose_world_body;

    LoopClosureIcpConfig icp_config;
    icp_config.voxel_resolution_m = backend_config_.loop_icp_voxel_size_m;
    icp_config.max_correspondence_distance_m =
      backend_config_.loop_icp_max_correspondence_m;
    icp_config.max_iterations = backend_config_.loop_icp_max_iterations;
    icp_config.num_threads = backend_config_.registration_num_threads;
    const LoopClosureIcpResult icp_result = refineLoopTransform(
      query_cloud_body, match_cloud_it->second,
      initial_transform_query_to_match, icp_config);

    // ゲート (b): ICP fitness で品質判定。 small_gicp の fitness は target-source の
    // 平均二乗距離相当なので、 1 m² を超えたら点群がまったく合っていない (= 偽 loop)。
    if (icp_result.fitness_score > backend_config_.loop_icp_max_fitness_score) {
      RCLCPP_INFO(get_logger(),
        "Loop rejected (fitness=%.3f > %.3f): query=%u, match=%u",
        icp_result.fitness_score,
        backend_config_.loop_icp_max_fitness_score,
        candidate.query_kf_id, candidate.match_kf_id);
      return false;
    }

    // ゲート (c): ICP 結果が初期推定から大きく跳んだら採用しない。 fitness は良くても
    // 局所最小で別の位置に張り付いている可能性がある。
    const Eigen::Vector3d translation_jump =
      icp_result.transform_match_from_query.translation() -
      initial_transform_query_to_match.translation();
    const double translation_jump_norm = translation_jump.norm();
    if (translation_jump_norm > backend_config_.loop_icp_max_translation_jump_m) {
      RCLCPP_INFO(get_logger(),
        "Loop rejected (translation jump=%.2f m > %.2f m): query=%u, match=%u",
        translation_jump_norm,
        backend_config_.loop_icp_max_translation_jump_m,
        candidate.query_kf_id, candidate.match_kf_id);
      return false;
    }

    RCLCPP_INFO(get_logger(),
      "Loop ICP (%s) accepted: query=%u, match=%u, fitness=%.4f, "
      "translation_jump=%.2f m, converged=%d",
      icp_result.backend_name.c_str(),
      candidate.query_kf_id, candidate.match_kf_id,
      icp_result.fitness_score, translation_jump_norm,
      icp_result.converged ? 1 : 0);

    PoseGraphOptimizer::NoiseSigmas loop_sigmas;
    loop_sigmas.rot_x_rad = backend_config_.pgo_loop_sigma_rot_rad;
    loop_sigmas.rot_y_rad = backend_config_.pgo_loop_sigma_rot_rad;
    loop_sigmas.rot_z_rad = backend_config_.pgo_loop_sigma_rot_rad;
    loop_sigmas.trans_x_m = backend_config_.pgo_loop_sigma_trans_m;
    loop_sigmas.trans_y_m = backend_config_.pgo_loop_sigma_trans_m;
    loop_sigmas.trans_z_m = backend_config_.pgo_loop_sigma_trans_m;
    // BetweenFactor 規約: addLoopFactor(query, match, T_query_match)
    // = T_world_query · T_query_match = T_world_match を満たすようにする。
    pose_graph_optimizer_->addLoopFactor(
      candidate.query_kf_id, candidate.match_kf_id,
      icp_result.transform_match_from_query, loop_sigmas);
    last_loop_match_kf_id_ = candidate.match_kf_id;
    last_loop_query_kf_id_ = candidate.query_kf_id;
    return true;
  }

  // PGO を 1 step 回し、 結果を submap_manager と estimator に反映する。
  // 最新 keyframe の修正姿勢を estimator->setPose() で上書きする。
  void runPoseGraphOptimizeAndApply(uint32_t latest_keyframe_id)
  {
    pose_graph_optimizer_->optimize();
    const auto optimized_poses = pose_graph_optimizer_->allOptimizedPoses();
    if (optimized_poses.empty()) {
      return;
    }
    // submap_manager に修正 pose を流し込み (grid を再構築)。
    // voxel_keyframe_submap_map でない実装の場合は何もしない。
    if (auto * submap_manager = dynamic_cast<VoxelKeyframeSubmapManager *>(
        backends_.point_cloud_map.get()))
    {
      submap_manager->updatePoses(optimized_poses);
    }
    // 最新 keyframe の修正姿勢を estimator に上書き
    for (const auto & [keyframe_id, optimized_pose] : optimized_poses) {
      if (keyframe_id == latest_keyframe_id) {
        backends_.state_estimator->setPose(optimized_pose);
        RCLCPP_INFO(get_logger(),
          "PGO applied: setPose to estimator at kf=%u, "
          "translation=(%.3f, %.3f, %.3f)",
          keyframe_id, optimized_pose.translation().x(),
          optimized_pose.translation().y(), optimized_pose.translation().z());
        // previous_keyframe_pose_ は次の odometry factor 用に最新値で同期
        previous_keyframe_pose_ = optimized_pose;
        break;
      }
    }
  }

  void publishLoopCandidate(
    const ScanContextLoopDetector::LoopCandidate & candidate,
    const rclcpp::Time & stamp)
  {
    // JSON 形式で std_msgs/String に詰める。 専用 msg を作るとパッケージ間依存が
    // 増えるため、 当面は JSON で済ませる (PR3 で正式 msg に置き換える想定)。
    std::ostringstream oss;
    const Eigen::Vector3d query_translation = candidate.query_pose_world_body.translation();
    const Eigen::Vector3d match_translation = candidate.match_pose_world_body.translation();
    const Eigen::Quaterniond query_rotation(candidate.query_pose_world_body.linear());
    const Eigen::Quaterniond match_rotation(candidate.match_pose_world_body.linear());
    oss << "{\"stamp_sec\":" << stamp.seconds()
        << ",\"query_kf_id\":" << candidate.query_kf_id
        << ",\"match_kf_id\":" << candidate.match_kf_id
        << ",\"score\":" << candidate.scan_context_distance
        << ",\"yaw_rad\":" << candidate.estimated_yaw_rad
        << ",\"query_pose\":[" << query_translation.x() << ","
        << query_translation.y() << "," << query_translation.z() << ","
        << query_rotation.x() << "," << query_rotation.y() << ","
        << query_rotation.z() << "," << query_rotation.w() << "]"
        << ",\"match_pose\":[" << match_translation.x() << ","
        << match_translation.y() << "," << match_translation.z() << ","
        << match_rotation.x() << "," << match_rotation.y() << ","
        << match_rotation.z() << "," << match_rotation.w() << "]"
        << "}";
    std_msgs::msg::String loop_message;
    loop_message.data = oss.str();
    loop_candidates_publisher_->publish(loop_message);
    RCLCPP_INFO(get_logger(),
      "Loop candidate: query_kf=%u <-> match_kf=%u, score=%.4f, yaw=%.3f rad",
      candidate.query_kf_id, candidate.match_kf_id,
      candidate.scan_context_distance, candidate.estimated_yaw_rad);
  }

  // 形成 map 全体 (active sliding window + finalized submap 群) を PCD に上書き保存する。
  // keyframe 採択ごとに呼ばれる前提。 map=voxel_keyframe_submap 以外は dynamic_cast が
  // 失敗するので no-op (= 暗黙にスキップ)。
  //
  // 巨大な map (数百 MB) を毎回フル書き出しするのは IO 負荷が高いため、 必要に応じて
  // keyframe_min_translation_m を広げる、 もしくは別途 throttle (例: M keyframe ごと) を
  // 入れる検討余地あり。 現状はユーザー指定通り「毎 keyframe 上書き」。
  void saveFullMapPcd()
  {
    auto * submap_manager = dynamic_cast<VoxelKeyframeSubmapManager *>(
      backends_.point_cloud_map.get());
    if (submap_manager == nullptr) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 30000,
        "PCD auto save: map is not voxel_keyframe_submap; saving skipped.");
      return;
    }
    const auto full_cloud = submap_manager->toFullMapPointCloud();
    if (full_cloud->points.empty()) {
      return;
    }
    int return_code = 0;
    if (backend_config_.output_pcd_binary) {
      return_code = pcl::io::savePCDFileBinary(
        backend_config_.output_pcd_path, *full_cloud);
    } else {
      return_code = pcl::io::savePCDFileASCII(
        backend_config_.output_pcd_path, *full_cloud);
    }
    if (return_code != 0) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
        "PCD auto save failed (code=%d): path='%s'",
        return_code, backend_config_.output_pcd_path.c_str());
    } else {
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000,
        "PCD auto saved: points=%zu, path='%s'",
        full_cloud->points.size(), backend_config_.output_pcd_path.c_str());
    }
  }

  // TF tree から T_imu_lidar = lookupTransform(target=imu_frame, source=lidar_frame) を取得し、
  // backend_config_.extrinsic_*_imu_from_lidar を上書きする。 lookup 失敗時は WARN + 既存値で続行。
  //
  // この関数は backends 生成前に 1 回だけ呼ばれる前提。 一時的に Buffer/Listener を作り、
  // タイムアウト付きで TF が現れるのを待つ。 robot_state_publisher が同時に起動する構成では、
  // robot_state_publisher の static TF が来る前に lio_node が走り始めるとここで wait する。
  void tryOverrideExtrinsicFromTf()
  {
    tf2_ros::Buffer temporary_tf_buffer(get_clock());
    // spin_thread=true で listener 内部に独立スレッドを立てて TF 購読を回す。
    // false にすると本 node の executor が spin するまで TF が buffer に入らず、
    // timeout 付き lookup が無意味になる (tf2_buffer が ERROR を出す典型ケース)。
    tf2_ros::TransformListener temporary_tf_listener(temporary_tf_buffer, this, true);

    const std::string & target_frame = backend_config_.imu_frame_id;
    const std::string & source_frame = backend_config_.lidar_frame_id;
    const auto timeout = tf2::durationFromSec(
      backend_config_.extrinsic_tf_lookup_timeout_s);

    try {
      const auto transform_stamped = temporary_tf_buffer.lookupTransform(
        target_frame, source_frame, tf2::TimePointZero, timeout);
      // translation
      backend_config_.extrinsic_translation_imu_from_lidar_x =
        transform_stamped.transform.translation.x;
      backend_config_.extrinsic_translation_imu_from_lidar_y =
        transform_stamped.transform.translation.y;
      backend_config_.extrinsic_translation_imu_from_lidar_z =
        transform_stamped.transform.translation.z;
      // rotation (quaternion → 3x3 → row-major 9 要素)
      const Eigen::Quaterniond quaternion(
        transform_stamped.transform.rotation.w,
        transform_stamped.transform.rotation.x,
        transform_stamped.transform.rotation.y,
        transform_stamped.transform.rotation.z);
      const Eigen::Matrix3d rotation_matrix = quaternion.toRotationMatrix();
      backend_config_.extrinsic_rotation_imu_from_lidar_row_major.resize(9);
      for (int row_index = 0; row_index < 3; ++row_index) {
        for (int column_index = 0; column_index < 3; ++column_index) {
          backend_config_.extrinsic_rotation_imu_from_lidar_row_major[
            row_index * 3 + column_index] = rotation_matrix(row_index, column_index);
        }
      }
      RCLCPP_INFO(get_logger(),
        "Extrinsic overridden from TF: T_%s_%s = t(%.4f, %.4f, %.4f)",
        target_frame.c_str(), source_frame.c_str(),
        transform_stamped.transform.translation.x,
        transform_stamped.transform.translation.y,
        transform_stamped.transform.translation.z);
    } catch (const tf2::TransformException & exception) {
      RCLCPP_WARN(get_logger(),
        "extrinsic_source=tf but lookupTransform('%s' <- '%s') failed: %s. "
        "Falling back to config (extrinsic_*_imu_from_lidar from yaml).",
        target_frame.c_str(), source_frame.c_str(), exception.what());
    }
  }


  // 5 秒 (パラメータ) ごとに submap_manager の「全体マップ」を取り出し publish する。
  // backends_.point_cloud_map が VoxelKeyframeSubmapManager でない場合は何もしない
  // (他の map 実装は finalized submap 概念を持たないため)。
  void onFullMapTimer()
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    auto * submap_manager = dynamic_cast<VoxelKeyframeSubmapManager *>(
      backends_.point_cloud_map.get());
    if (submap_manager == nullptr) {
      return;
    }
    const auto full_cloud = submap_manager->toFullMapPointCloud();
    if (full_cloud->points.empty()) {
      return;
    }
    sensor_msgs::msg::PointCloud2 cloud_message;
    pcl::toROSMsg(*full_cloud, cloud_message);
    cloud_message.header.stamp = now();
    cloud_message.header.frame_id = world_frame_id_;
    full_map_publisher_->publish(cloud_message);
    RCLCPP_INFO(get_logger(),
      "Full map published: points=%zu (active+finalized submaps=%zu)",
      full_cloud->points.size(), submap_manager->finalizedSubmapCount());
  }

  void onWatchdogTimer()
  {
    const auto now = std::chrono::steady_clock::now();

    if (cloud_callback_count_ == 0) {
      RCLCPP_WARN(get_logger(),
        "No point cloud received on '%s' in the last %d s. "
        "Check: (1) is the bag playing? (2) topic name typo? "
        "Try: ros2 topic hz %s   ros2 topic info %s",
        input_cloud_topic_.c_str(), watchdog_interval_s_,
        input_cloud_topic_.c_str(), input_cloud_topic_.c_str());
    } else {
      const double idle_s =
        std::chrono::duration<double>(now - last_cloud_wall_time_).count();
      if (idle_s > watchdog_interval_s_) {
        RCLCPP_WARN(get_logger(),
          "Point cloud callback STOPPED after %lu scans (%.1fs since last). "
          "Publisher still alive? Try: ros2 topic hz %s   ros2 topic info %s -v",
          static_cast<unsigned long>(cloud_callback_count_), idle_s,
          input_cloud_topic_.c_str(), input_cloud_topic_.c_str());
      } else {
        // 元気に来ているとき: 統計を 1 行だけ出す。
        RCLCPP_INFO(get_logger(),
          "alive: scans=%lu, imu=%lu, idle=%.1fs",
          static_cast<unsigned long>(cloud_callback_count_),
          static_cast<unsigned long>(imu_callback_count_),
          idle_s);
      }
    }
    if (backends_.state_estimator->usesImu()) {
      if (imu_callback_count_ == 0) {
        RCLCPP_WARN(get_logger(),
          "No IMU received on '%s' in the last %d s.",
          input_imu_topic_.c_str(), watchdog_interval_s_);
      } else {
        const double idle_s =
          std::chrono::duration<double>(now - last_imu_wall_time_).count();
        if (idle_s > watchdog_interval_s_) {
          RCLCPP_WARN(get_logger(),
            "IMU callback STOPPED after %lu samples (%.1fs since last).",
            static_cast<unsigned long>(imu_callback_count_), idle_s);
        }
      }
    }
  }

  void publishOutputs(const rclcpp::Time & stamp)
  {
    const RobotState state = backends_.state_estimator->getState();

    nav_msgs::msg::Odometry odom_message;
    odom_message.header.stamp = stamp;
    odom_message.header.frame_id = world_frame_id_;
    odom_message.child_frame_id = body_frame_id_;
    odom_message.pose.pose.position.x = state.pose_world_body.translation().x();
    odom_message.pose.pose.position.y = state.pose_world_body.translation().y();
    odom_message.pose.pose.position.z = state.pose_world_body.translation().z();
    const Eigen::Quaterniond quaternion(state.pose_world_body.linear());
    odom_message.pose.pose.orientation.x = quaternion.x();
    odom_message.pose.pose.orientation.y = quaternion.y();
    odom_message.pose.pose.orientation.z = quaternion.z();
    odom_message.pose.pose.orientation.w = quaternion.w();
    odom_message.twist.twist.linear.x = state.velocity_world.x();
    odom_message.twist.twist.linear.y = state.velocity_world.y();
    odom_message.twist.twist.linear.z = state.velocity_world.z();
    odom_publisher_->publish(odom_message);

    if (publish_tf_) {
      geometry_msgs::msg::TransformStamped transform_stamped;
      transform_stamped.header.stamp = stamp;
      transform_stamped.header.frame_id = world_frame_id_;
      transform_stamped.child_frame_id = body_frame_id_;
      transform_stamped.transform.translation.x = state.pose_world_body.translation().x();
      transform_stamped.transform.translation.y = state.pose_world_body.translation().y();
      transform_stamped.transform.translation.z = state.pose_world_body.translation().z();
      transform_stamped.transform.rotation.x = quaternion.x();
      transform_stamped.transform.rotation.y = quaternion.y();
      transform_stamped.transform.rotation.z = quaternion.z();
      transform_stamped.transform.rotation.w = quaternion.w();
      tf_broadcaster_->sendTransform(transform_stamped);
    }

    // ローカルマップ全体を毎フレーム流すのは重いので、20Hz 中 1Hz 程度に間引く。
    ++cloud_publish_counter_;
    if (cloud_publish_counter_ >= cloud_publish_interval_) {
      cloud_publish_counter_ = 0;
      const auto world_cloud = backends_.point_cloud_map->toPointCloud();
      sensor_msgs::msg::PointCloud2 cloud_message;
      pcl::toROSMsg(*world_cloud, cloud_message);
      cloud_message.header.stamp = stamp;
      cloud_message.header.frame_id = world_frame_id_;
      cloud_publisher_->publish(cloud_message);
    }

    const EstimatorDiagnostics diag = backends_.state_estimator->getDiagnostics();
    std_msgs::msg::String diag_message;
    std::ostringstream oss;
    oss << "{\"iterations\":" << diag.iterations
        << ",\"cost\":" << diag.cost
        << ",\"converged\":" << (diag.converged ? "true" : "false")
        << ",\"processing_time_ms\":" << diag.processing_time_ms
        << ",\"map_size\":" << backends_.point_cloud_map->size()
        << "}";
    diag_message.data = oss.str();
    diag_publisher_->publish(diag_message);
  }

  // パラメータ・トピック
  LioBackendConfig backend_config_;
  std::string world_frame_id_;
  std::string body_frame_id_;
  std::string input_cloud_topic_;
  std::string input_imu_topic_;
  std::string output_odom_topic_;
  std::string output_cloud_topic_;
  std::string output_diag_topic_;
  bool publish_tf_ = true;

  // バックエンド
  LioBackends backends_;
  std::mutex state_mutex_;

  // ROS
  std::string input_cloud_format_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_subscription_;
#ifdef PYLOT_LIO_HAS_LIVOX_DRIVER
  rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr
    livox_custom_subscription_;
#endif
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_subscription_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_publisher_;
  // 形成マップ全体 (finalized + active) を 5 秒に 1 回出す publisher と timer。
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr full_map_publisher_;
  rclcpp::TimerBase::SharedPtr full_map_timer_;
  std::string output_full_map_topic_;
  double full_map_publish_period_s_ = 5.0;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr diag_publisher_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  // 起動時 1 回だけ publish する base_link → lidar / imu の static TF。

  int cloud_publish_counter_ = 0;
  // /lio/cloud_world は全マップを toROSMsg してネットワークに流すので重い。
  // 30Hz 入力に対し ~2Hz (= 15 scan に 1 回) 程度に抑えないと callback がフレーム周期内に終わらない。
  static constexpr int cloud_publish_interval_ = 15;

  // Keyframe ロジック
  std::unique_ptr<IKeyframeSelector> keyframe_selector_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr keyframe_path_publisher_;
  nav_msgs::msg::Path keyframe_path_msg_;
  std::string output_keyframes_topic_;
  double keyframe_min_translation_m_ = 1.0;
  double keyframe_min_rotation_rad_ = 0.2;
  int keyframe_max_scans_between_ = 50;

  // Loop closure detection (Scan Context)。 enable_loop_detection=false の場合は
  // loop_detector_ が nullptr のままになる。
  std::unique_ptr<ScanContextLoopDetector> loop_detector_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr loop_candidates_publisher_;
  std::string output_loop_candidates_topic_;

  // Pose Graph Optimization (GTSAM ISAM2)
  std::unique_ptr<PoseGraphOptimizer> pose_graph_optimizer_;
  bool has_first_keyframe_ = false;
  uint32_t previous_keyframe_id_ = 0;
  Eigen::Isometry3d previous_keyframe_pose_ = Eigen::Isometry3d::Identity();
  int keyframes_since_last_pgo_ = 0;
  // Loop ICP の target 用に body 系の keyframe 点群を 1 個ずつキャッシュ。
  // 大きくなるためメモリ圧迫が問題なら LRU 化を検討。
  std::unordered_map<uint32_t, PointCloud> keyframe_clouds_body_;

  // 連続マッチ抑制用 (cooldown)
  std::optional<uint32_t> last_loop_match_kf_id_;
  uint32_t last_loop_query_kf_id_ = 0;

  // 診断カウンタと watchdog
  uint64_t imu_callback_count_ = 0;
  uint64_t cloud_callback_count_ = 0;
  std::chrono::steady_clock::time_point last_cloud_wall_time_;
  std::chrono::steady_clock::time_point last_imu_wall_time_;
  rclcpp::TimerBase::SharedPtr watchdog_timer_;
  static constexpr int watchdog_interval_s_ = 5;
};

}  // namespace pylot_lio

#endif  // PYLOT_LIO__LIO_NODE_HPP_
