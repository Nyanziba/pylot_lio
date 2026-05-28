// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#include "pylot_lio/factory.hpp"

#include <cstdio>
#include <memory>
#include <sstream>
#include <stdexcept>

#include "pylot_lio/estimator/gicp_only_estimator.hpp"
#include "pylot_lio/estimator/hgo_estimator.hpp"
#include "pylot_lio/estimator/ieskf_estimator.hpp"
#include "pylot_lio/map/kd_tree_map.hpp"
#include "pylot_lio/map/normal_map.hpp"
#include "pylot_lio/map/voxel_map.hpp"
#include "pylot_lio/map/voxel_random_map.hpp"
#include "pylot_lio/map/voxel_keyframe_submap_map.hpp"
#include "pylot_lio/preprocess/random_sampling_preprocessor.hpp"
#include "pylot_lio/preprocess/voxel_grid_preprocessor.hpp"
#include "pylot_lio/preprocess/voxel_random_sampling_preprocessor.hpp"
#include "pylot_lio/registration/plain_gicp_registration.hpp"
#include "pylot_lio/registration/metal_vgicp_registration.hpp"

#ifdef PYLOT_LIO_HAS_SMALL_GICP
#include "pylot_lio/registration/small_gicp_registration.hpp"
#endif

namespace pylot_lio
{

namespace
{

IPreprocessorPtr buildPreprocessor(const LioBackendConfig & config)
{
  if (config.preprocessor_name == "voxel_grid") {
    return std::make_unique<VoxelGridPreprocessor>(config.voxel_grid_size_m);
  }
  if (config.preprocessor_name == "random_sampling") {
    return std::make_unique<RandomSamplingPreprocessor>(
      static_cast<std::size_t>(config.random_sampling_target_count), 12345u);
  }
  if (config.preprocessor_name == "voxel_random_sampling") {
    VoxelRandomSamplingPreprocessor::Config voxel_random_config;
    voxel_random_config.voxel_size_m = config.voxel_grid_size_m;
    voxel_random_config.random_seed = config.voxel_random_sampling_seed;
    voxel_random_config.sampling_rate = config.voxel_random_sampling_rate;
    voxel_random_config.use_gpu = config.voxel_random_sampling_use_gpu;
    return std::make_unique<VoxelRandomSamplingPreprocessor>(voxel_random_config);
  }
  throw std::invalid_argument(
    "Unknown preprocessor_name: " + config.preprocessor_name);
}

IPointCloudMapPtr buildPointCloudMap(const LioBackendConfig & config)
{
  if (config.map_name == "voxel_map") {
    VoxelMap::Config voxel_map_config;
    voxel_map_config.voxel_size_m = config.map_voxel_size_m;
    voxel_map_config.min_points_per_cell_for_covariance = config.map_min_points_per_cell;
    voxel_map_config.max_total_cells = config.map_max_total_cells;
    return std::make_unique<VoxelMap>(voxel_map_config);
  }
  if (config.map_name == "normal_map") {
    NormalMap::Config normal_map_config;
    normal_map_config.voxel_size_m = config.map_voxel_size_m;
    normal_map_config.min_points_per_cell = config.map_min_points_per_cell;
    normal_map_config.max_total_cells = config.map_max_total_cells;
    return std::make_unique<NormalMap>(normal_map_config);
  }
  if (config.map_name == "kd_tree_map") {
    KdTreeMap::Config kd_tree_config;
    kd_tree_config.max_total_points = config.map_max_total_cells;
    return std::make_unique<KdTreeMap>(kd_tree_config);
  }
  if (config.map_name == "voxel_random_map") {
    VoxelRandomMap::Config voxel_random_config;
    voxel_random_config.voxel_size_m = config.map_voxel_size_m;
    voxel_random_config.max_total_cells = config.map_max_total_cells;
    voxel_random_config.max_points_per_voxel = config.voxel_random_map_max_points_per_voxel;
    voxel_random_config.k_nearest_for_covariance =
      config.voxel_random_map_k_nearest_for_covariance;
    return std::make_unique<VoxelRandomMap>(voxel_random_config);
  }
  if (config.map_name == "voxel_keyframe_submap") {
    VoxelKeyframeSubmapManager::Config submap_config;
    submap_config.voxel_size_m = config.map_voxel_size_m;
    submap_config.min_points_per_cell_for_search = config.map_min_points_per_cell;
    submap_config.sliding_window_size = config.submap_window_size;
    submap_config.finalize_size = config.submap_finalize_size;
    submap_config.max_points_per_cell = config.submap_max_points_per_cell;
    submap_config.k_nearest_for_covariance = config.submap_k_nearest_for_covariance;
    // セル内点の保持戦略 (string → enum)。 未知文字列はエラーにせずデフォルトに落とす。
    if (config.submap_cell_sampling_mode == "random_one") {
      submap_config.cell_sampling_mode =
        VoxelKeyframeSubmapManager::CellSamplingMode::RandomOne;
    } else {
      submap_config.cell_sampling_mode =
        VoxelKeyframeSubmapManager::CellSamplingMode::KeepUpToN;
    }
    submap_config.random_seed = static_cast<uint32_t>(config.submap_random_seed);
    return std::make_unique<VoxelKeyframeSubmapManager>(submap_config);
  }
  throw std::invalid_argument("Unknown map_name: " + config.map_name);
}

// plain_gicp の Config を BackendConfig から組む (plain_gicp 本体・sycl/metal の
// フォールバックで共用)。
PlainGicpRegistration::Config makePlainGicpConfig(const LioBackendConfig & config)
{
  PlainGicpRegistration::Config plain_gicp_config;
  plain_gicp_config.max_iterations = config.registration_max_iterations;
  plain_gicp_config.max_correspondence_distance_m =
    config.registration_max_correspondence_m;
  plain_gicp_config.convergence_translation_m =
    config.registration_convergence_translation_m;
  plain_gicp_config.convergence_rotation_rad =
    config.registration_convergence_rotation_rad;
  plain_gicp_config.huber_threshold = config.registration_huber_threshold;
  plain_gicp_config.num_threads = config.registration_num_threads;
  plain_gicp_config.source_covariance_num_neighbors =
    config.registration_source_covariance_num_neighbors;
  plain_gicp_config.source_covariance_plane_epsilon =
    config.registration_source_covariance_plane_epsilon;
  plain_gicp_config.enable_degenerate_regularization =
    config.enable_degenerate_regularization;
  plain_gicp_config.rotation_eigenvalue_threshold = config.rotation_eigenvalue_threshold;
  plain_gicp_config.translation_eigenvalue_threshold =
    config.translation_eigenvalue_threshold;
  plain_gicp_config.regularization_base_factor = config.regularization_base_factor;
  plain_gicp_config.parallel_backend = config.registration_parallel_backend;
  return plain_gicp_config;
}

IRegistrationPtr buildRegistration(const LioBackendConfig & config)
{
  if (config.registration_name == "plain_gicp") {
    return std::make_unique<PlainGicpRegistration>(makePlainGicpConfig(config));
  }
  if (config.registration_name == "metal_vgicp") {
    // Apple Metal GPU VGICP。 非対応ビルド (PYLOT_LIO_HAS_METAL 未定義) では
    // plain_gicp に自動フォールバックする (tbb fallback と同パターン)。
    if (MetalVgicpRegistration::isAvailable()) {
      MetalVgicpRegistration::Config metal_config;
      metal_config.max_iterations = config.registration_max_iterations;
      metal_config.max_correspondence_distance_m = config.registration_max_correspondence_m;
      metal_config.convergence_translation_m = config.registration_convergence_translation_m;
      metal_config.convergence_rotation_rad = config.registration_convergence_rotation_rad;
      metal_config.huber_threshold = config.registration_huber_threshold;
      metal_config.source_covariance_num_neighbors =
        config.registration_source_covariance_num_neighbors;
      metal_config.source_covariance_plane_epsilon =
        config.registration_source_covariance_plane_epsilon;
      metal_config.gpu_min_points = config.registration_metal_gpu_min_points;
      metal_config.voxelmap_levels = config.registration_metal_voxelmap_levels;
      metal_config.voxelmap_scaling_factor =
        config.registration_metal_voxelmap_scaling_factor;
      metal_config.use_gpu_source_covariance =
        config.registration_metal_gpu_source_covariance;
      metal_config.source_covariance_cell_size_m =
        config.registration_metal_source_covariance_cell_size_m;
      // voxel_map / normal_map 以外 (point ベースマップ) を target にするときの
      // ボクセル化設定。 map レイヤーの voxel_size / min_points を流用する。
      metal_config.target_voxel_size_m = config.map_voxel_size_m;
      metal_config.min_points_per_voxel = config.map_min_points_per_cell;
      // 地面 leveling 拘束 (IMU 重力なしのピッチドリフト対策)。
      metal_config.enable_ground_constraint = config.registration_metal_enable_ground_constraint;
      metal_config.ground_constraint_weight = config.registration_metal_ground_constraint_weight;
      metal_config.ground_band_m = config.registration_metal_ground_band_m;
      metal_config.ground_max_tilt_deg = config.registration_metal_ground_max_tilt_deg;
      metal_config.ground_max_correction_per_frame_deg =
        config.registration_metal_ground_max_correction_per_frame_deg;
      metal_config.ground_vibration_threshold_deg =
        config.registration_metal_ground_vibration_threshold_deg;
      return std::make_unique<MetalVgicpRegistration>(metal_config);
    }
    std::fprintf(
      stderr,
      "[factory] registration='metal_vgicp' requested but this build has no Metal "
      "support; falling back to plain_gicp.\n");
    return std::make_unique<PlainGicpRegistration>(makePlainGicpConfig(config));
  }
#ifdef PYLOT_LIO_HAS_SMALL_GICP
  if (config.registration_name == "small_gicp_gicp" ||
      config.registration_name == "small_gicp_vgicp")
  {
    SmallGicpRegistration::Config small_gicp_config;
    small_gicp_config.variant = (config.registration_name == "small_gicp_vgicp")
      ? SmallGicpRegistration::Variant::VGICP
      : SmallGicpRegistration::Variant::GICP;
    small_gicp_config.map_voxel_resolution_m = config.map_voxel_size_m;
    small_gicp_config.max_correspondence_distance_m =
      config.registration_max_correspondence_m;
    small_gicp_config.num_threads = config.registration_num_threads;
    small_gicp_config.max_iterations = config.registration_max_iterations;
    small_gicp_config.parallel_backend = config.registration_parallel_backend;
    return std::make_unique<SmallGicpRegistration>(small_gicp_config);
  }
#endif
  if (config.registration_name == "sycl") {
    // SYCL バックエンドは S4 で実装。CMake 側で stub 経由のフォールバック。
    // ここでは PlainGicp にフォールバックしてビルドを通す。
    PlainGicpRegistration::Config fallback_config;
    fallback_config.max_iterations = config.registration_max_iterations;
    fallback_config.max_correspondence_distance_m =
      config.registration_max_correspondence_m;
    fallback_config.convergence_translation_m =
      config.registration_convergence_translation_m;
    fallback_config.convergence_rotation_rad =
      config.registration_convergence_rotation_rad;
    fallback_config.huber_threshold = config.registration_huber_threshold;
    fallback_config.num_threads = config.registration_num_threads;
    return std::make_unique<PlainGicpRegistration>(fallback_config);
  }
  throw std::invalid_argument(
    "Unknown registration_name: " + config.registration_name);
}

IStateEstimatorPtr buildStateEstimator(const LioBackendConfig & config)
{
  const Eigen::Vector3d gravity_world(
    config.gravity_world_x, config.gravity_world_y, config.gravity_world_z);
  if (config.state_estimator_name == "ieskf") {
    IeskfEstimator::Config ieskf_config;
    // Super-LIO 流: 大きさは gravity_norm 固定、 方向だけ起動時 IMU から推定。
    ieskf_config.gravity_norm = config.gravity_norm;
    ieskf_config.gravity_world_init = gravity_world;
    ieskf_config.auto_estimate_gravity = config.auto_estimate_gravity;
    ieskf_config.gravity_estimation_samples = config.gravity_estimation_samples;
    // LiDAR ↔ IMU extrinsic
    ieskf_config.extrinsic_translation_imu_from_lidar = Eigen::Vector3d(
      config.extrinsic_translation_imu_from_lidar_x,
      config.extrinsic_translation_imu_from_lidar_y,
      config.extrinsic_translation_imu_from_lidar_z);
    const auto & rotation_row_major =
      config.extrinsic_rotation_imu_from_lidar_row_major;
    if (rotation_row_major.size() == 9) {
      Eigen::Matrix3d rotation_matrix;
      for (int row_index = 0; row_index < 3; ++row_index) {
        for (int column_index = 0; column_index < 3; ++column_index) {
          rotation_matrix(row_index, column_index) =
            rotation_row_major[row_index * 3 + column_index];
        }
      }
      ieskf_config.extrinsic_rotation_imu_from_lidar = rotation_matrix;
    }
    return std::make_unique<IeskfEstimator>(ieskf_config);
  }
  if (config.state_estimator_name == "hgo") {
    HgoEstimator::Config hgo_config;
    hgo_config.gravity_world = gravity_world;
    hgo_config.auto_estimate_gravity = config.auto_estimate_gravity;
    hgo_config.gravity_estimation_samples = config.gravity_estimation_samples;
    return std::make_unique<HgoEstimator>(hgo_config);
  }
  if (config.state_estimator_name == "gicp_only") {
    GicpOnlyEstimator::RejectionConfig rejection_config;
    rejection_config.enabled = config.gicp_only_rejection_enabled;
    rejection_config.max_translation_correction_m =
      config.gicp_only_max_translation_correction_m;
    rejection_config.max_rotation_correction_deg =
      config.gicp_only_max_rotation_correction_deg;
    rejection_config.min_correspondences_when_unconverged =
      config.gicp_only_min_correspondences_when_unconverged;
    return std::make_unique<GicpOnlyEstimator>(rejection_config);
  }
  throw std::invalid_argument(
    "Unknown state_estimator_name: " + config.state_estimator_name);
}

}  // namespace

LioBackends createBackendsFromConfig(const LioBackendConfig & config)
{
  LioBackends backends;
  backends.preprocessor = buildPreprocessor(config);
  backends.point_cloud_map = buildPointCloudMap(config);
  backends.registration = buildRegistration(config);
  backends.state_estimator = buildStateEstimator(config);

  std::ostringstream oss;
  oss << "[" << backends.state_estimator->describe()
      << "] + [" << backends.registration->describe()
      << "] + [" << backends.point_cloud_map->describe()
      << "] + [" << backends.preprocessor->describe() << "]";
  backends.summary = oss.str();
  return backends;
}

}  // namespace pylot_lio
