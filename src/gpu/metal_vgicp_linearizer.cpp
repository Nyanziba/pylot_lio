// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#include "pylot_lio/gpu/metal_vgicp_linearizer.hpp"

#include <cmath>
#include <cstdint>

#include <Eigen/Eigenvalues>

// metal-cpp の「実装マクロ」(NS_PRIVATE_IMPLEMENTATION 等) は
// src/gpu/metal_compute_probe.cpp の唯一の TU で既に定義済み。 ここで再定義すると
// シンボル重複でリンクエラーになるため、 マクロ無しで宣言だけ取り込む。
// (必ずファイルスコープで include すること: namespace 内に置くと <functional> が
//  pylot_lio::gpu::std として解釈され libc++ 内部が壊れる。)
#ifdef PYLOT_LIO_HAS_METAL
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#endif

namespace pylot_lio::gpu
{

namespace
{

// CPU/GPU で完全に同一でなければならないボクセルハッシュ (spatial hashing)。
// int32 を uint32 に解釈して乗算 → XOR。 capacity は 2 の冪なので & (capacity-1)。
inline std::uint32_t hashVoxel(
  std::int32_t x, std::int32_t y, std::int32_t z, int capacity)
{
  const std::uint32_t hash =
    (static_cast<std::uint32_t>(x) * 73856093u) ^
    (static_cast<std::uint32_t>(y) * 19349663u) ^
    (static_cast<std::uint32_t>(z) * 83492791u);
  return hash & static_cast<std::uint32_t>(capacity - 1);
}

// world 座標 → ボクセル整数座標 (floor)。 CPU/GPU で同一にする。
inline Eigen::Vector3i voxelCoord(const Eigen::Vector3d & world, double voxel_size)
{
  return Eigen::Vector3i(
    static_cast<int>(std::floor(world.x() / voxel_size)),
    static_cast<int>(std::floor(world.y() / voxel_size)),
    static_cast<int>(std::floor(world.z() / voxel_size)));
}

// 対称 3x3 を adjugate/det で逆行列化 (fp64)。 det が小さい時は floor で持ち上げる。
// Metal カーネルも同一アルゴリズム (fp32) を使い、 比較を「fp32 丸めの差」だけに絞る。
Eigen::Matrix3d inverse3x3WithFloor(const Eigen::Matrix3d & matrix, double det_floor)
{
  const double a = matrix(0, 0), b = matrix(0, 1), c = matrix(0, 2);
  const double d = matrix(1, 0), e = matrix(1, 1), f = matrix(1, 2);
  const double g = matrix(2, 0), h = matrix(2, 1), i = matrix(2, 2);
  const double cof0 = e * i - f * h;
  const double cof1 = c * h - b * i;
  const double cof2 = b * f - c * e;
  double det = a * cof0 + d * cof1 + g * cof2;
  if (std::abs(det) < det_floor) {
    det = (det < 0.0 ? -det_floor : det_floor);
  }
  const double inv_det = 1.0 / det;
  Eigen::Matrix3d result;
  result(0, 0) = cof0 * inv_det;
  result(0, 1) = cof1 * inv_det;
  result(0, 2) = cof2 * inv_det;
  result(1, 0) = (f * g - d * i) * inv_det;
  result(1, 1) = (a * i - c * g) * inv_det;
  result(1, 2) = (c * d - a * f) * inv_det;
  result(2, 0) = (d * h - e * g) * inv_det;
  result(2, 1) = (b * g - a * h) * inv_det;
  result(2, 2) = (a * e - b * d) * inv_det;
  return result;
}

inline Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d & v)
{
  Eigen::Matrix3d skew;
  skew << 0.0, -v.z(), v.y(),
    v.z(), 0.0, -v.x(),
    -v.y(), v.x(), 0.0;
  return skew;
}

constexpr double kCovarianceDetFloor = 1e-9;

}  // namespace

VgicpVoxelTable VgicpVoxelTable::build(
  float voxel_size_m,
  const std::vector<Eigen::Vector3i> & voxel_coords,
  const std::vector<Eigen::Vector3d> & means,
  const std::vector<Eigen::Matrix3d> & covariances)
{
  VgicpVoxelTable table;
  table.voxel_size_m = voxel_size_m;

  const std::size_t num_voxels = voxel_coords.size();
  // capacity = num_voxels*2 以上の最小 2 冪 (最低 8)。 負荷率 <= 0.5 で probe 短く保つ。
  int capacity = 8;
  while (static_cast<std::size_t>(capacity) < num_voxels * 2) {
    capacity <<= 1;
  }
  table.capacity = capacity;
  table.occupied.assign(capacity, 0);
  table.voxel_keys_xyz.assign(static_cast<std::size_t>(capacity) * 3, 0);
  table.means_xyz.assign(static_cast<std::size_t>(capacity) * 3, 0.0f);
  table.covariances.assign(static_cast<std::size_t>(capacity) * 9, 0.0f);

  for (std::size_t voxel_index = 0; voxel_index < num_voxels; ++voxel_index) {
    const Eigen::Vector3i & coord = voxel_coords[voxel_index];
    std::uint32_t slot = hashVoxel(coord.x(), coord.y(), coord.z(), capacity);
    // 線形プロービングで空きスロットを探す (負荷率 <= 0.5 なので必ず空きがある)。
    while (table.occupied[slot] != 0) {
      slot = (slot + 1) & static_cast<std::uint32_t>(capacity - 1);
    }
    table.occupied[slot] = 1;
    table.voxel_keys_xyz[slot * 3 + 0] = coord.x();
    table.voxel_keys_xyz[slot * 3 + 1] = coord.y();
    table.voxel_keys_xyz[slot * 3 + 2] = coord.z();
    table.means_xyz[slot * 3 + 0] = static_cast<float>(means[voxel_index].x());
    table.means_xyz[slot * 3 + 1] = static_cast<float>(means[voxel_index].y());
    table.means_xyz[slot * 3 + 2] = static_cast<float>(means[voxel_index].z());
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        table.covariances[slot * 9 + r * 3 + c] =
          static_cast<float>(covariances[voxel_index](r, c));
      }
    }
  }
  return table;
}

VgicpLinearization linearizeVgicpCpu(
  const std::vector<Eigen::Vector3d> & source_points_body,
  const std::vector<Eigen::Matrix3d> & source_covariances_body,
  const VgicpVoxelTable & voxel_table,
  const Eigen::Isometry3d & transform_world_body,
  const VgicpLinearizeConfig & config)
{
  VgicpLinearization result;
  const double voxel_size = voxel_table.voxel_size_m;
  const double max_corr_sq =
    static_cast<double>(config.max_correspondence_distance_m) *
    static_cast<double>(config.max_correspondence_distance_m);
  const double huber = config.huber_threshold;
  const Eigen::Matrix3d rotation = transform_world_body.linear();

  const std::size_t num_points = source_points_body.size();
  for (std::size_t point_index = 0; point_index < num_points; ++point_index) {
    const Eigen::Vector3d & source_body = source_points_body[point_index];
    if (!source_body.allFinite()) {
      continue;
    }
    const Eigen::Vector3d source_world = transform_world_body * source_body;

    // 近傍 (2*radius+1)^3 ボクセルを走査し、 mean が最も近い有効ボクセルを target にする。
    // 各近傍について: ハッシュ → 線形プロービングで一致 key を探す (空きで打ち切り)。
    const Eigen::Vector3i center = voxelCoord(source_world, voxel_size);
    const std::uint32_t mask = static_cast<std::uint32_t>(voxel_table.capacity - 1);
    const int radius = config.search_radius_voxels;
    bool found = false;
    double best_distance_sq = max_corr_sq;
    Eigen::Vector3d target_mean = Eigen::Vector3d::Zero();
    Eigen::Matrix3d target_cov = Eigen::Matrix3d::Identity();

    for (int dz = -radius; dz <= radius; ++dz) {
      for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
          const Eigen::Vector3i coord(center.x() + dx, center.y() + dy, center.z() + dz);
          std::uint32_t slot = hashVoxel(coord.x(), coord.y(), coord.z(), voxel_table.capacity);
          for (int probe = 0; probe < voxel_table.capacity; ++probe) {
            if (voxel_table.occupied[slot] == 0) {
              break;  // 空きスロット = この voxel は未登録
            }
            if (voxel_table.voxel_keys_xyz[slot * 3 + 0] == coord.x() &&
                voxel_table.voxel_keys_xyz[slot * 3 + 1] == coord.y() &&
                voxel_table.voxel_keys_xyz[slot * 3 + 2] == coord.z())
            {
              const Eigen::Vector3d candidate_mean(
                voxel_table.means_xyz[slot * 3 + 0],
                voxel_table.means_xyz[slot * 3 + 1],
                voxel_table.means_xyz[slot * 3 + 2]);
              const double distance_sq = (candidate_mean - source_world).squaredNorm();
              if (distance_sq < best_distance_sq) {
                best_distance_sq = distance_sq;
                target_mean = candidate_mean;
                for (int r = 0; r < 3; ++r) {
                  for (int c = 0; c < 3; ++c) {
                    target_cov(r, c) = voxel_table.covariances[slot * 9 + r * 3 + c];
                  }
                }
                found = true;
              }
              break;  // この近傍 voxel は見つけた (key 一致) ので probe 終了
            }
            slot = (slot + 1) & mask;
          }
        }
      }
    }
    if (!found) {
      continue;
    }
    // best_distance_sq は探索時に max_corr_sq 未満であることを既に保証済み。
    const Eigen::Vector3d residual = target_mean - source_world;

    const Eigen::Matrix3d source_cov_world =
      rotation * source_covariances_body[point_index] * rotation.transpose();
    const Eigen::Matrix3d combined = target_cov + source_cov_world;
    const Eigen::Matrix3d information = inverse3x3WithFloor(combined, kCovarianceDetFloor);

    const double mahalanobis = residual.transpose() * information * residual;
    double weight = 1.0;
    if (mahalanobis > huber * huber) {
      weight = huber / std::sqrt(mahalanobis);
    }

    Eigen::Matrix<double, 3, 6> jacobian;
    jacobian.block<3, 3>(0, 0) = skewSymmetric(source_world);
    jacobian.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();

    result.hessian.noalias() += weight * jacobian.transpose() * information * jacobian;
    result.gradient.noalias() -= weight * jacobian.transpose() * information * residual;
    result.cost += weight * mahalanobis;
    result.valid_correspondences += 1;
  }
  result.gpu_used = false;
  return result;
}

#ifndef PYLOT_LIO_HAS_METAL

VgicpLinearization linearizeVgicpMetal(
  const std::vector<Eigen::Vector3d> & source_points_body,
  const std::vector<Eigen::Matrix3d> & source_covariances_body,
  const VgicpVoxelTable & voxel_table,
  const Eigen::Isometry3d & transform_world_body,
  const VgicpLinearizeConfig & config)
{
  // Metal 非対応ビルド: CPU 参照にフォールバックしつつ gpu_used=false を明示。
  VgicpLinearization result = linearizeVgicpCpu(
    source_points_body, source_covariances_body, voxel_table,
    transform_world_body, config);
  result.gpu_used = false;
  result.error_message = "Metal support not compiled in (PYLOT_LIO_HAS_METAL undefined)";
  return result;
}

#else  // PYLOT_LIO_HAS_METAL

namespace
{

// カーネルに渡すスカラ群。 Metal 側の struct とレイアウトを完全一致させる
// (全フィールド 4 byte、 padding 無し)。
struct VgicpKernelParams
{
  float rotation[9];      // R_world_body (row-major)
  float translation[3];   // t_world_body
  float voxel_size;
  float inv_voxel_size;
  float huber_threshold;
  float max_corr_sq;
  std::uint32_t capacity;
  std::uint32_t num_points;
  std::int32_t search_radius;  // 近傍探索半径 (ボクセル単位、 0=自ボクセルのみ)
  std::int32_t padding;        // 16 byte 境界揃え (Metal constant の安全側)
};

// 1 threadgroup = 128 スレッド固定。 部分和は 29 float / threadgroup。
//   [0..20] : H 上三角 (row-major: (0,0)(0,1)..(0,5)(1,1)..(5,5))
//   [21..26]: b[0..5]
//   [27]    : cost
//   [28]    : count
constexpr int kThreadgroupSize = 128;
constexpr int kPartialStride = 29;

const char * kVgicpKernelSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct VgicpKernelParams {
  float rotation[9];
  float translation[3];
  float voxel_size;
  float inv_voxel_size;
  float huber_threshold;
  float max_corr_sq;
  uint  capacity;
  uint  num_points;
  int   search_radius;
  int   padding;
};

inline uint hash_voxel(int x, int y, int z, uint capacity) {
  uint h = (uint(x) * 73856093u) ^ (uint(y) * 19349663u) ^ (uint(z) * 83492791u);
  return h & (capacity - 1u);
}

// 対称 3x3 の adjugate/det 逆行列 (CPU 参照と同一アルゴリズム)。
inline float3x3 inverse3x3_floored(float3x3 m, float det_floor) {
  float a = m[0][0], b = m[1][0], c = m[2][0];
  float d = m[0][1], e = m[1][1], f = m[2][1];
  float g = m[0][2], h = m[1][2], i = m[2][2];
  // 注: metal の float3x3 は列優先。 ここでは m[col][row] でアクセスし、
  //     CPU の row-major と整合するよう a..i を「数学的な行列要素」に合わせて読む。
  float cof0 = e * i - f * h;
  float cof1 = c * h - b * i;
  float cof2 = b * f - c * e;
  float det = a * cof0 + d * cof1 + g * cof2;
  if (fabs(det) < det_floor) {
    det = (det < 0.0f ? -det_floor : det_floor);
  }
  float inv_det = 1.0f / det;
  // 数学的な逆行列 (row-major) を float3x3 (col-major) に詰め直す: out[col][row]
  float3x3 out;
  out[0][0] = cof0 * inv_det;            out[1][0] = cof1 * inv_det;            out[2][0] = cof2 * inv_det;
  out[0][1] = (f * g - d * i) * inv_det; out[1][1] = (a * i - c * g) * inv_det; out[2][1] = (c * d - a * f) * inv_det;
  out[0][2] = (d * h - e * g) * inv_det; out[1][2] = (b * g - a * h) * inv_det; out[2][2] = (a * e - b * d) * inv_det;
  return out;
}

kernel void vgicp_linearize(
    device const float * source_points     [[buffer(0)]],  // num_points * 3
    device const float * source_covs        [[buffer(1)]],  // num_points * 9 (row-major)
    device const uchar * voxel_occupied     [[buffer(2)]],  // capacity
    device const int   * voxel_keys         [[buffer(3)]],  // capacity * 3
    device const float * voxel_means        [[buffer(4)]],  // capacity * 3
    device const float * voxel_covs         [[buffer(5)]],  // capacity * 9 (row-major)
    constant VgicpKernelParams & params     [[buffer(6)]],
    device float       * partials           [[buffer(7)]],  // num_groups * 29
    uint  global_id   [[thread_position_in_grid]],
    uint  local_id    [[thread_position_in_threadgroup]],
    uint  group_id    [[threadgroup_position_in_grid]])
{
  const uint kStride = 29u;
  threadgroup float tg[128][29];

  // --- 1 点ぶんの寄与を計算 (範囲外 / 対応無しは全ゼロ) ---
  float contrib[29];
  for (uint k = 0; k < kStride; ++k) { contrib[k] = 0.0f; }

  if (global_id < params.num_points) {
    float3 p_body = float3(
      source_points[global_id * 3 + 0],
      source_points[global_id * 3 + 1],
      source_points[global_id * 3 + 2]);

    // 非有限 (NaN/Inf) の source 点は寄与ゼロでスキップ (CPU 参照の allFinite と同じ)。
    if (isfinite(p_body.x) && isfinite(p_body.y) && isfinite(p_body.z)) {

    // R (row-major) を float3x3 (col-major) に: Rc[col][row] = rotation[row*3+col]
    float3x3 R;
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {
        R[col][row] = params.rotation[row * 3 + col];
      }
    }
    float3 t = float3(params.translation[0], params.translation[1], params.translation[2]);
    float3 p_world = R * p_body + t;

    int cx = int(floor(p_world.x * params.inv_voxel_size));
    int cy = int(floor(p_world.y * params.inv_voxel_size));
    int cz = int(floor(p_world.z * params.inv_voxel_size));

    uint mask = params.capacity - 1u;
    int radius = params.search_radius;
    bool found = false;
    float best_dist_sq = params.max_corr_sq;
    uint best_slot = 0u;

    // 近傍 (2*radius+1)^3 ボクセルを走査し、 mean が最も近い有効ボクセルを選ぶ。
    for (int dz = -radius; dz <= radius; ++dz) {
      for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
          int vx = cx + dx, vy = cy + dy, vz = cz + dz;
          uint slot = hash_voxel(vx, vy, vz, params.capacity);
          for (uint probe = 0; probe < params.capacity; ++probe) {
            if (voxel_occupied[slot] == 0) { break; }
            if (voxel_keys[slot * 3 + 0] == vx &&
                voxel_keys[slot * 3 + 1] == vy &&
                voxel_keys[slot * 3 + 2] == vz) {
              float3 cand_mu = float3(
                voxel_means[slot * 3 + 0], voxel_means[slot * 3 + 1], voxel_means[slot * 3 + 2]);
              float3 cd = cand_mu - p_world;
              float dist_sq = dot(cd, cd);
              if (dist_sq < best_dist_sq) {
                best_dist_sq = dist_sq;
                best_slot = slot;
                found = true;
              }
              break;
            }
            slot = (slot + 1u) & mask;
          }
        }
      }
    }

    if (found) {
      uint slot = best_slot;
      float3 mu_t = float3(
        voxel_means[slot * 3 + 0], voxel_means[slot * 3 + 1], voxel_means[slot * 3 + 2]);
      float3 d = mu_t - p_world;
      {
        // C_t (row-major) → float3x3
        float3x3 Ct;
        for (int row = 0; row < 3; ++row) {
          for (int col = 0; col < 3; ++col) {
            Ct[col][row] = voxel_covs[slot * 9 + row * 3 + col];
          }
        }
        // C_s (body, row-major) → float3x3
        float3x3 Cs;
        for (int row = 0; row < 3; ++row) {
          for (int col = 0; col < 3; ++col) {
            Cs[col][row] = source_covs[global_id * 9 + row * 3 + col];
          }
        }
        float3x3 Cs_world = R * Cs * transpose(R);
        float3x3 combined = Ct + Cs_world;
        float3x3 Omega = inverse3x3_floored(combined, 1e-9f);

        // mahalanobis = dᵀ Ω d
        float3 Od = Omega * d;
        float maha = dot(d, Od);
        float weight = 1.0f;
        float huber = params.huber_threshold;
        if (maha > huber * huber) {
          weight = huber / sqrt(maha);
        }

        // J = [ skew(p_world) | -I ] (3x6)。 行 r の 6 成分を作る。
        // skew(p): [[0,-z,y],[z,0,-x],[-y,x,0]]
        float3 sk0 = float3(0.0f, -p_world.z, p_world.y);
        float3 sk1 = float3(p_world.z, 0.0f, -p_world.x);
        float3 sk2 = float3(-p_world.y, p_world.x, 0.0f);
        // J 行ベクトル (各 6 次元): col 0..2 = skew 行, col 3..5 = -I 行
        float J[3][6];
        J[0][0] = sk0.x; J[0][1] = sk0.y; J[0][2] = sk0.z; J[0][3] = -1.0f; J[0][4] = 0.0f; J[0][5] = 0.0f;
        J[1][0] = sk1.x; J[1][1] = sk1.y; J[1][2] = sk1.z; J[1][3] = 0.0f; J[1][4] = -1.0f; J[1][5] = 0.0f;
        J[2][0] = sk2.x; J[2][1] = sk2.y; J[2][2] = sk2.z; J[2][3] = 0.0f; J[2][4] = 0.0f; J[2][5] = -1.0f;

        // Omega を配列で扱う (row-major 数学行列)。 Omega[col][row] なので O(r,c)=Omega[c][r]
        float Om[3][3];
        for (int r = 0; r < 3; ++r) {
          for (int c = 0; c < 3; ++c) {
            Om[r][c] = Omega[c][r];
          }
        }

        // M = Ω J  (3x6)
        float M[3][6];
        for (int r = 0; r < 3; ++r) {
          for (int col = 0; col < 6; ++col) {
            float s = 0.0f;
            for (int k = 0; k < 3; ++k) { s += Om[r][k] * J[k][col]; }
            M[r][col] = s;
          }
        }
        // H = w Jᵀ M (6x6, 上三角のみ)、 b = -w Jᵀ (Ω d)
        int idx = 0;
        for (int a = 0; a < 6; ++a) {
          for (int b2 = a; b2 < 6; ++b2) {
            float s = 0.0f;
            for (int r = 0; r < 3; ++r) { s += J[r][a] * M[r][b2]; }
            contrib[idx] = weight * s;
            ++idx;
          }
        }
        // b[a] = -w Σ_r J[r][a] * Od[r]
        float Odarr[3] = { Od.x, Od.y, Od.z };
        for (int a = 0; a < 6; ++a) {
          float s = 0.0f;
          for (int r = 0; r < 3; ++r) { s += J[r][a] * Odarr[r]; }
          contrib[21 + a] = -weight * s;
        }
        contrib[27] = weight * maha;
        contrib[28] = 1.0f;
      }
    }
    }  // close isfinite(p_body) guard
  }

  // --- threadgroup 部分和: log-step ツリー reduction ---
  // thread 0 直列ではなく、 半分ずつ畳む。 木構造が固定なので毎回同一順序 = 決定的。
  for (uint k = 0; k < kStride; ++k) { tg[local_id][k] = contrib[k]; }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = 64u; stride > 0u; stride >>= 1u) {
    if (local_id < stride) {
      for (uint k = 0; k < kStride; ++k) {
        tg[local_id][k] += tg[local_id + stride][k];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (local_id == 0) {
    for (uint k = 0; k < kStride; ++k) {
      partials[group_id * kStride + k] = tg[0][k];
    }
  }
}
)METAL";

}  // namespace

VgicpLinearization linearizeVgicpMetal(
  const std::vector<Eigen::Vector3d> & source_points_body,
  const std::vector<Eigen::Matrix3d> & source_covariances_body,
  const VgicpVoxelTable & voxel_table,
  const Eigen::Isometry3d & transform_world_body,
  const VgicpLinearizeConfig & config)
{
  VgicpLinearization result;
  const std::size_t num_points = source_points_body.size();
  if (num_points == 0 || source_covariances_body.size() != num_points) {
    result.error_message = "empty source or covariance count mismatch";
    return result;
  }
  if (voxel_table.capacity <= 0) {
    result.error_message = "empty voxel table";
    return result;
  }

  NS::AutoreleasePool * pool = NS::AutoreleasePool::alloc()->init();

  MTL::Device * device = MTL::CreateSystemDefaultDevice();
  if (device == nullptr) {
    result.error_message = "no Metal device";
    pool->release();
    // CPU フォールバック (gpu_used=false のまま)。
    VgicpLinearization cpu = linearizeVgicpCpu(
      source_points_body, source_covariances_body, voxel_table, transform_world_body, config);
    cpu.error_message = result.error_message;
    return cpu;
  }

  NS::Error * compile_error = nullptr;
  NS::String * source = NS::String::string(kVgicpKernelSource, NS::UTF8StringEncoding);
  MTL::Library * library = device->newLibrary(source, nullptr, &compile_error);
  if (library == nullptr) {
    result.error_message = "kernel compile failed: ";
    if (compile_error != nullptr) {
      result.error_message += compile_error->localizedDescription()->utf8String();
    }
    device->release();
    pool->release();
    return result;
  }
  MTL::Function * function =
    library->newFunction(NS::String::string("vgicp_linearize", NS::UTF8StringEncoding));
  NS::Error * pso_error = nullptr;
  MTL::ComputePipelineState * pso = device->newComputePipelineState(function, &pso_error);
  if (pso == nullptr) {
    result.error_message = "pipeline creation failed";
    function->release();
    library->release();
    device->release();
    pool->release();
    return result;
  }

  // --- 入力をフラット fp32 配列に詰める ---
  std::vector<float> source_points_flat(num_points * 3);
  std::vector<float> source_covs_flat(num_points * 9);
  for (std::size_t i = 0; i < num_points; ++i) {
    source_points_flat[i * 3 + 0] = static_cast<float>(source_points_body[i].x());
    source_points_flat[i * 3 + 1] = static_cast<float>(source_points_body[i].y());
    source_points_flat[i * 3 + 2] = static_cast<float>(source_points_body[i].z());
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        source_covs_flat[i * 9 + r * 3 + c] =
          static_cast<float>(source_covariances_body[i](r, c));
      }
    }
  }

  VgicpKernelParams params{};
  const Eigen::Matrix3d rotation = transform_world_body.linear();
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      params.rotation[r * 3 + c] = static_cast<float>(rotation(r, c));
    }
  }
  params.translation[0] = static_cast<float>(transform_world_body.translation().x());
  params.translation[1] = static_cast<float>(transform_world_body.translation().y());
  params.translation[2] = static_cast<float>(transform_world_body.translation().z());
  params.voxel_size = voxel_table.voxel_size_m;
  params.inv_voxel_size = 1.0f / voxel_table.voxel_size_m;
  params.huber_threshold = config.huber_threshold;
  params.max_corr_sq =
    config.max_correspondence_distance_m * config.max_correspondence_distance_m;
  params.capacity = static_cast<std::uint32_t>(voxel_table.capacity);
  params.num_points = static_cast<std::uint32_t>(num_points);
  params.search_radius = config.search_radius_voxels;
  params.padding = 0;

  const int num_groups =
    static_cast<int>((num_points + kThreadgroupSize - 1) / kThreadgroupSize);

  auto make_buffer = [&](const void * data, std::size_t bytes) {
    return device->newBuffer(data, bytes, MTL::ResourceStorageModeShared);
  };
  MTL::Buffer * buf_points =
    make_buffer(source_points_flat.data(), source_points_flat.size() * sizeof(float));
  MTL::Buffer * buf_covs =
    make_buffer(source_covs_flat.data(), source_covs_flat.size() * sizeof(float));
  MTL::Buffer * buf_occupied =
    make_buffer(voxel_table.occupied.data(), voxel_table.occupied.size());
  MTL::Buffer * buf_keys =
    make_buffer(voxel_table.voxel_keys_xyz.data(),
      voxel_table.voxel_keys_xyz.size() * sizeof(std::int32_t));
  MTL::Buffer * buf_means =
    make_buffer(voxel_table.means_xyz.data(), voxel_table.means_xyz.size() * sizeof(float));
  MTL::Buffer * buf_vcovs =
    make_buffer(voxel_table.covariances.data(), voxel_table.covariances.size() * sizeof(float));
  MTL::Buffer * buf_params = make_buffer(&params, sizeof(VgicpKernelParams));
  MTL::Buffer * buf_partials = device->newBuffer(
    static_cast<std::size_t>(num_groups) * kPartialStride * sizeof(float),
    MTL::ResourceStorageModeShared);

  MTL::CommandQueue * queue = device->newCommandQueue();
  MTL::CommandBuffer * command_buffer = queue->commandBuffer();
  MTL::ComputeCommandEncoder * encoder = command_buffer->computeCommandEncoder();
  encoder->setComputePipelineState(pso);
  encoder->setBuffer(buf_points, 0, 0);
  encoder->setBuffer(buf_covs, 0, 1);
  encoder->setBuffer(buf_occupied, 0, 2);
  encoder->setBuffer(buf_keys, 0, 3);
  encoder->setBuffer(buf_means, 0, 4);
  encoder->setBuffer(buf_vcovs, 0, 5);
  encoder->setBuffer(buf_params, 0, 6);
  encoder->setBuffer(buf_partials, 0, 7);
  const MTL::Size grid = MTL::Size::Make(
    static_cast<NS::UInteger>(num_groups), 1, 1);
  const MTL::Size tg = MTL::Size::Make(kThreadgroupSize, 1, 1);
  encoder->dispatchThreadgroups(grid, tg);
  encoder->endEncoding();
  command_buffer->commit();
  command_buffer->waitUntilCompleted();

  // --- threadgroup 部分和を CPU で固定順序合算 ---
  const float * partials = static_cast<const float *>(buf_partials->contents());
  double accum[kPartialStride];
  for (int k = 0; k < kPartialStride; ++k) { accum[k] = 0.0; }
  for (int group = 0; group < num_groups; ++group) {
    for (int k = 0; k < kPartialStride; ++k) {
      accum[k] += static_cast<double>(partials[group * kPartialStride + k]);
    }
  }

  // 上三角 21 個から対称 H を復元。
  int idx = 0;
  for (int a = 0; a < 6; ++a) {
    for (int b = a; b < 6; ++b) {
      const double value = accum[idx];
      result.hessian(a, b) = value;
      result.hessian(b, a) = value;
      ++idx;
    }
  }
  for (int a = 0; a < 6; ++a) {
    result.gradient(a) = accum[21 + a];
  }
  result.cost = accum[27];
  result.valid_correspondences = static_cast<int>(std::lround(accum[28]));
  result.gpu_used = true;

  buf_partials->release();
  buf_params->release();
  buf_vcovs->release();
  buf_means->release();
  buf_keys->release();
  buf_occupied->release();
  buf_covs->release();
  buf_points->release();
  queue->release();
  pso->release();
  function->release();
  library->release();
  device->release();
  pool->release();

  return result;
}

#endif  // PYLOT_LIO_HAS_METAL

}  // namespace pylot_lio::gpu
