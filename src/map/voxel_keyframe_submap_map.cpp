// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#include "pylot_lio/map/voxel_keyframe_submap_map.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <sstream>
#include <unordered_set>
#include <utility>

#include <Eigen/Eigenvalues>

namespace pylot_lio
{

std::size_t VoxelKeyframeSubmapManager::VoxelKeyHash::operator()(
  const VoxelKey & key) const noexcept
{
  // VoxelMap と同じ混ぜ方。3 つの座標を別々の odd 定数で撹拌してから xor する。
  const auto h1 = std::hash<int64_t>{}(key.x);
  const auto h2 = std::hash<int64_t>{}(key.y);
  const auto h3 = std::hash<int64_t>{}(key.z);
  return h1 ^ (h2 * 0x9E3779B97F4A7C15ULL) ^ (h3 * 0xBF58476D1CE4E5B9ULL);
}

VoxelKeyframeSubmapManager::VoxelKeyframeSubmapManager(const Config & config)
: config_(config),
  random_engine_(
    config.random_seed != 0u ? config.random_seed : std::random_device{}())
{
}

VoxelKeyframeSubmapManager::VoxelKey
VoxelKeyframeSubmapManager::computeKey(const Eigen::Vector3d & position_world) const
{
  const double inverse_voxel_size = 1.0 / config_.voxel_size_m;
  return VoxelKey{
    static_cast<int64_t>(std::floor(position_world.x() * inverse_voxel_size)),
    static_cast<int64_t>(std::floor(position_world.y() * inverse_voxel_size)),
    static_cast<int64_t>(std::floor(position_world.z() * inverse_voxel_size))
  };
}

void VoxelKeyframeSubmapManager::evictKeyframeFromGrid(uint32_t keyframe_id_to_evict)
{
  // grid 全セルを走査し、退場対象 keyframe_id のエントリだけを erase-remove する。
  // セルが空になったらセル自体も grid から削除する。
  // O(セル数 + 全エントリ数) だが、退場は keyframe ごとに高々 1 回なので許容。
  for (auto cell_iterator = grid_.begin(); cell_iterator != grid_.end(); ) {
    auto & entries = cell_iterator->second;
    entries.erase(
      std::remove_if(
        entries.begin(), entries.end(),
        [keyframe_id_to_evict](const PointEntry & entry) {
          return entry.keyframe_id == keyframe_id_to_evict;
        }),
      entries.end());
    if (entries.empty()) {
      cell_iterator = grid_.erase(cell_iterator);
    } else {
      ++cell_iterator;
    }
  }
}

void VoxelKeyframeSubmapManager::maybeFinalizeSubmap()
{
  // 「最後に finalize したときからのキーフレーム差」が finalize_size 以上なら確定。
  // ループでぐるりと回せば、たとえば次のキーフレームで一気に複数 submap を確定させる
  // ことになっても辻褄が合う (現実には 1 回でループは抜ける)。
  const uint32_t finalize_size = static_cast<uint32_t>(std::max(1, config_.finalize_size));
  while (next_keyframe_id_ - keyframes_at_last_finalize_ >= finalize_size) {
    const uint32_t start_kf = keyframes_at_last_finalize_;
    const uint32_t end_kf_exclusive = start_kf + finalize_size;

    Submap submap;
    submap.submap_id = next_submap_id_++;
    submap.keyframe_ids.reserve(finalize_size);
    submap.merged_cloud_world.points.clear();

    // 末尾 anchor pose は range の最後のキーフレームの姿勢
    for (uint32_t kf_id = start_kf; kf_id < end_kf_exclusive; ++kf_id) {
      submap.keyframe_ids.push_back(kf_id);
      // 同じ kf_id のレコードが keyframe_records_[kf_id] にあるはず (push順 == id)。
      const KeyframeRecord & record = keyframe_records_[kf_id];
      const PointCloud & cloud_world = keyframe_clouds_world_[kf_id];
      submap.merged_cloud_world.points.insert(
        submap.merged_cloud_world.points.end(),
        cloud_world.points.begin(), cloud_world.points.end());
      submap.anchor_pose_world_body = record.pose_world_body;
    }
    submap.merged_cloud_world.width =
      static_cast<uint32_t>(submap.merged_cloud_world.points.size());
    submap.merged_cloud_world.height = 1;
    submap.merged_cloud_world.is_dense = true;

    finalized_submaps_.push_back(std::move(submap));
    keyframes_at_last_finalize_ = end_kf_exclusive;
  }
}

void VoxelKeyframeSubmapManager::insertScan(
  const PointCloud & scan_in_world,
  const Eigen::Isometry3d & pose_world_body)
{
  const uint32_t my_keyframe_id = next_keyframe_id_++;

  // 1) keyframe record と cloud を蓄える (確定 submap 構築用)
  KeyframeRecord record;
  record.keyframe_id = my_keyframe_id;
  record.pose_world_body = pose_world_body;
  keyframe_records_.push_back(record);

  PointCloud cloud_world_copy;
  cloud_world_copy.points.reserve(scan_in_world.points.size());

  // 2) grid に挿入 (NaN を弾く + max_points_per_cell でセル内点数を上限)
  for (const Point & input_point : scan_in_world.points) {
    if (!std::isfinite(input_point.x) ||
        !std::isfinite(input_point.y) ||
        !std::isfinite(input_point.z))
    {
      continue;
    }
    const Eigen::Vector3d position_world(input_point.x, input_point.y, input_point.z);
    const VoxelKey key = computeKey(position_world);
    auto & entries = grid_[key];
    if (config_.cell_sampling_mode == CellSamplingMode::RandomOne) {
      // セルに必ず 1 点だけ持つ。 既に 1 点ある場合は「50% で置換」する単純な
      // reservoir sampling (= サンプル数 1 の reservoir)。 厳密な per-cell count を
      // 持たない近似だが、 各 voxel に届く点は数〜数百でほぼ気にならない範囲。
      if (entries.empty()) {
        entries.push_back(PointEntry{position_world, my_keyframe_id});
      } else {
        std::uniform_int_distribution<int> replace_distribution(0, 1);
        if (replace_distribution(random_engine_) == 0) {
          entries.front() = PointEntry{position_world, my_keyframe_id};
        }
      }
    } else {
      // KeepUpToN モード (既定、 後方互換)
      if (config_.max_points_per_cell <= 0 ||
          static_cast<int>(entries.size()) < config_.max_points_per_cell)
      {
        entries.push_back(PointEntry{position_world, my_keyframe_id});
      }
    }
    cloud_world_copy.points.push_back(input_point);
  }
  cloud_world_copy.width = static_cast<uint32_t>(cloud_world_copy.points.size());
  cloud_world_copy.height = 1;
  cloud_world_copy.is_dense = true;
  keyframe_clouds_world_.push_back(std::move(cloud_world_copy));

  // 3) sliding window 更新 (古い keyframe を grid から追い出す)
  active_keyframe_ids_.push_back(my_keyframe_id);
  const int window_size = std::max(1, config_.sliding_window_size);
  while (static_cast<int>(active_keyframe_ids_.size()) > window_size) {
    const uint32_t oldest_id = active_keyframe_ids_.front();
    active_keyframe_ids_.pop_front();
    evictKeyframeFromGrid(oldest_id);
  }

  // 4) 確定 submap のスナップショット (M 個ごと)
  maybeFinalizeSubmap();
}

PointCorrespondence
VoxelKeyframeSubmapManager::findNearestNeighbor(const Eigen::Vector3d & query_world) const
{
  PointCorrespondence correspondence;
  correspondence.source_point_world = query_world;
  correspondence.valid = false;
  correspondence.squared_distance = std::numeric_limits<double>::infinity();

  const VoxelKey base_key = computeKey(query_world);
  const int search_radius = config_.neighbor_search_radius_voxels;

  // 最近傍点の探索: 探索半径内の全セルから最近点を 1 つ選ぶ。
  // 共分散は近傍点 K 個から自前で計算する (entries は raw point の集まりなので
  // VoxelMap のような online 統計は持っていない)。
  std::vector<Eigen::Vector3d> neighbor_points;
  neighbor_points.reserve(64);

  for (int delta_x = -search_radius; delta_x <= search_radius; ++delta_x) {
    for (int delta_y = -search_radius; delta_y <= search_radius; ++delta_y) {
      for (int delta_z = -search_radius; delta_z <= search_radius; ++delta_z) {
        const VoxelKey neighbor_key{
          base_key.x + delta_x, base_key.y + delta_y, base_key.z + delta_z};
        const auto found = grid_.find(neighbor_key);
        if (found == grid_.end()) {
          continue;
        }
        if (static_cast<int>(found->second.size()) <
            config_.min_points_per_cell_for_search)
        {
          continue;
        }
        for (const PointEntry & entry : found->second) {
          neighbor_points.push_back(entry.point_world);
          const double squared_distance =
            (entry.point_world - query_world).squaredNorm();
          if (squared_distance < correspondence.squared_distance) {
            correspondence.squared_distance = squared_distance;
            correspondence.target_point_world = entry.point_world;
            correspondence.valid = true;
          }
        }
      }
    }
  }

  if (!correspondence.valid) {
    return correspondence;
  }

  // 共分散: 最近傍 K 個の点で簡易推定。
  // K 未満しかなければ全点を使う。空 (=1点しかない) なら等方の床値で埋める。
  const int k_target =
    std::max(1, config_.k_nearest_for_covariance);
  if (static_cast<int>(neighbor_points.size()) > k_target) {
    std::nth_element(
      neighbor_points.begin(), neighbor_points.begin() + k_target,
      neighbor_points.end(),
      [&query_world](const Eigen::Vector3d & lhs, const Eigen::Vector3d & rhs) {
        return (lhs - query_world).squaredNorm() < (rhs - query_world).squaredNorm();
      });
    neighbor_points.resize(k_target);
  }

  Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
  if (neighbor_points.size() >= 2) {
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (const Eigen::Vector3d & point : neighbor_points) {
      mean += point;
    }
    mean /= static_cast<double>(neighbor_points.size());
    for (const Eigen::Vector3d & point : neighbor_points) {
      const Eigen::Vector3d deviation = point - mean;
      covariance += deviation * deviation.transpose();
    }
    covariance /= static_cast<double>(neighbor_points.size() - 1);
  }

  // 退化方向の暴走を防ぐため、最小固有値を floor で持ち上げる。
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
  Eigen::Vector3d eigenvalues = solver.eigenvalues();
  for (int axis = 0; axis < 3; ++axis) {
    if (eigenvalues(axis) < config_.covariance_eigen_floor) {
      eigenvalues(axis) = config_.covariance_eigen_floor;
    }
  }
  correspondence.target_covariance =
    solver.eigenvectors() * eigenvalues.asDiagonal() * solver.eigenvectors().transpose();
  return correspondence;
}

std::size_t VoxelKeyframeSubmapManager::size() const
{
  return grid_.size();
}

PointCloudPtr VoxelKeyframeSubmapManager::toPointCloud() const
{
  auto output_cloud = std::make_shared<PointCloud>();
  std::size_t total_points = 0;
  for (const auto & [key, entries] : grid_) {
    (void)key;
    total_points += entries.size();
  }
  output_cloud->points.reserve(total_points);
  for (const auto & [key, entries] : grid_) {
    (void)key;
    for (const PointEntry & entry : entries) {
      Point output_point;
      output_point.x = static_cast<float>(entry.point_world.x());
      output_point.y = static_cast<float>(entry.point_world.y());
      output_point.z = static_cast<float>(entry.point_world.z());
      output_point.intensity = static_cast<float>(entry.keyframe_id);
      output_cloud->points.push_back(output_point);
    }
  }
  output_cloud->width = static_cast<uint32_t>(output_cloud->points.size());
  output_cloud->height = 1;
  output_cloud->is_dense = true;
  return output_cloud;
}

std::string VoxelKeyframeSubmapManager::describe() const
{
  std::ostringstream oss;
  oss << "voxel_keyframe_submap:voxel=" << config_.voxel_size_m
      << ",window=" << config_.sliding_window_size
      << ",finalize=" << config_.finalize_size
      << ",cells=" << grid_.size()
      << ",active_kf=" << active_keyframe_ids_.size()
      << ",finalized_submaps=" << finalized_submaps_.size();
  return oss.str();
}

std::vector<uint32_t> VoxelKeyframeSubmapManager::activeKeyframeIds() const
{
  return std::vector<uint32_t>(active_keyframe_ids_.begin(), active_keyframe_ids_.end());
}

const std::vector<VoxelKeyframeSubmapManager::Submap> &
VoxelKeyframeSubmapManager::finalizedSubmaps() const
{
  return finalized_submaps_;
}

PointCloudPtr VoxelKeyframeSubmapManager::toFullMapPointCloud() const
{
  auto output_cloud = std::make_shared<PointCloud>();
  // 容量を予約: active grid のエントリ数 + finalized submap の点数合計
  std::size_t total_points = 0;
  for (const auto & [key, entries] : grid_) {
    (void)key;
    total_points += entries.size();
  }
  for (const Submap & submap : finalized_submaps_) {
    total_points += submap.merged_cloud_world.points.size();
  }
  output_cloud->points.reserve(total_points);

  // 1) active sliding window 部分 (grid) を吐き出す
  for (const auto & [key, entries] : grid_) {
    (void)key;
    for (const PointEntry & entry : entries) {
      Point output_point;
      output_point.x = static_cast<float>(entry.point_world.x());
      output_point.y = static_cast<float>(entry.point_world.y());
      output_point.z = static_cast<float>(entry.point_world.z());
      // intensity に keyframe_id を入れて RViz で色分け可視化可能にする
      output_point.intensity = static_cast<float>(entry.keyframe_id);
      output_cloud->points.push_back(output_point);
    }
  }

  // 2) finalized submap の点群を連結
  for (const Submap & submap : finalized_submaps_) {
    for (const Point & input_point : submap.merged_cloud_world.points) {
      output_cloud->points.push_back(input_point);
    }
  }
  output_cloud->width = static_cast<uint32_t>(output_cloud->points.size());
  output_cloud->height = 1;
  output_cloud->is_dense = true;
  return output_cloud;
}

void VoxelKeyframeSubmapManager::updatePoses(
  const std::vector<std::pair<uint32_t, Eigen::Isometry3d>> & updated_poses)
{
  // PGO で 1 回に修正される keyframe は通常 sliding window 内が中心。
  // ここでは「修正された keyframe 全てに対し、 grid 上の点を T_new · T_old^{-1} で
  // 写し直す」処理を行う。 アクティブでない keyframe は keyframe_records_ の pose
  // 更新のみ。
  //
  // 注意: keyframe_clouds_world_ も「修正前の world 座標」のままなので、 修正対象
  // については新 pose に合わせて再回転 (旧 world → body → 新 world) する必要が
  // ある。 そうしないと sliding window から退場したあと 確定 submap として
  // 切り出される際にズレた cloud を蓄えてしまう。

  // 1) 旧 pose をスナップショットし、 keyframe_records_ を新 pose に上書き
  std::vector<std::pair<uint32_t, Eigen::Isometry3d>> old_to_new_transform_per_keyframe;
  old_to_new_transform_per_keyframe.reserve(updated_poses.size());
  for (const auto & [keyframe_id, new_pose] : updated_poses) {
    if (keyframe_id >= keyframe_records_.size()) {
      continue;
    }
    const Eigen::Isometry3d old_pose = keyframe_records_[keyframe_id].pose_world_body;
    keyframe_records_[keyframe_id].pose_world_body = new_pose;
    // 旧 world → 新 world の剛体変換 T_new · T_old^{-1}
    const Eigen::Isometry3d transform_old_world_to_new_world = new_pose * old_pose.inverse();
    old_to_new_transform_per_keyframe.emplace_back(
      keyframe_id, transform_old_world_to_new_world);
  }

  // 2) keyframe_clouds_world_ を新 world に書き換え。 後段の確定 submap でも整合する。
  for (const auto & [keyframe_id, transform_old_to_new] :
       old_to_new_transform_per_keyframe)
  {
    PointCloud & cloud_world = keyframe_clouds_world_[keyframe_id];
    for (Point & point : cloud_world.points) {
      const Eigen::Vector3d old_world_position(point.x, point.y, point.z);
      const Eigen::Vector3d new_world_position =
        transform_old_to_new * old_world_position;
      point.x = static_cast<float>(new_world_position.x());
      point.y = static_cast<float>(new_world_position.y());
      point.z = static_cast<float>(new_world_position.z());
    }
  }

  // 3) アクティブ keyframe の grid を作り直す: 退場済みの keyframe を grid に
  // 入れ直すと巨大化するため、 「現在 active な keyframe だけ」を新 cloud で
  // 再投入する。
  std::unordered_set<uint32_t> active_id_set;
  active_id_set.reserve(active_keyframe_ids_.size());
  for (uint32_t active_id : active_keyframe_ids_) {
    active_id_set.insert(active_id);
  }

  grid_.clear();
  for (uint32_t active_id : active_keyframe_ids_) {
    const PointCloud & cloud_world = keyframe_clouds_world_[active_id];
    for (const Point & input_point : cloud_world.points) {
      if (!std::isfinite(input_point.x) || !std::isfinite(input_point.y) ||
          !std::isfinite(input_point.z))
      {
        continue;
      }
      const Eigen::Vector3d position_world(input_point.x, input_point.y, input_point.z);
      const VoxelKey key = computeKey(position_world);
      auto & entries = grid_[key];
      if (config_.cell_sampling_mode == CellSamplingMode::RandomOne) {
        if (entries.empty()) {
          entries.push_back(PointEntry{position_world, active_id});
        } else {
          std::uniform_int_distribution<int> replace_distribution(0, 1);
          if (replace_distribution(random_engine_) == 0) {
            entries.front() = PointEntry{position_world, active_id};
          }
        }
      } else {
        if (config_.max_points_per_cell <= 0 ||
            static_cast<int>(entries.size()) < config_.max_points_per_cell)
        {
          entries.push_back(PointEntry{position_world, active_id});
        }
      }
    }
  }
}

}  // namespace pylot_lio
