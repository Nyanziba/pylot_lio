// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#include "pylot_lio/gpu/metal_covariance_estimator.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <unordered_map>

#include <Eigen/Geometry>
#include <Eigen/LU>

// metal-cpp の「実装マクロ」は src/gpu/metal_compute_probe.cpp の唯一の TU で定義済み。
// ここでは宣言だけ取り込む (必ずファイルスコープで include する: namespace 内に置くと
// <functional> 等が pylot_lio::gpu::std として解釈され libc++ 内部が壊れる)。
#ifdef PYLOT_LIO_HAS_METAL
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#endif

namespace pylot_lio::gpu
{

namespace
{

// 近傍探索で 1 点が抱えられる候補 k の上限。 num_neighbors はこれにクランプする。
// GPU カーネルの thread-local 配列サイズと一致させること。
constexpr int kMaxNeighbors = 32;

// CPU/GPU で完全に同一でなければならないセルハッシュ (linearizer と同一定数)。
inline std::uint32_t hashCell(
  std::int32_t x, std::int32_t y, std::int32_t z, int capacity)
{
  const std::uint32_t hash =
    (static_cast<std::uint32_t>(x) * 73856093u) ^
    (static_cast<std::uint32_t>(y) * 19349663u) ^
    (static_cast<std::uint32_t>(z) * 83492791u);
  return hash & static_cast<std::uint32_t>(capacity - 1);
}

// body 座標 → セル整数座標 (floor)。 CPU/GPU で同一。
inline Eigen::Vector3i cellCoord(const Eigen::Vector3d & point, double cell_size)
{
  return Eigen::Vector3i(
    static_cast<int>(std::floor(point.x() / cell_size)),
    static_cast<int>(std::floor(point.y() / cell_size)),
    static_cast<int>(std::floor(point.z() / cell_size)));
}

// 対称 3x3 の最小固有値に対応する単位固有ベクトル (= 平面の法線) を closed-form で。
// 反復ソルバを使わない (GPU スレッドで決定的にするため)。 CPU 参照は fp64、 GPU は fp32
// で同一アルゴリズムを実装する。 ok=false のときは法線が定まらない (等方・縮退)。
Eigen::Vector3d smallestEigenvectorSymmetric(const Eigen::Matrix3d & a, bool & ok)
{
  ok = false;
  // 非対角成分の二乗和。 0 なら対角行列。
  const double p1 = a(0, 1) * a(0, 1) + a(0, 2) * a(0, 2) + a(1, 2) * a(1, 2);
  const double trace = a(0, 0) + a(1, 1) + a(2, 2);
  const double q = trace / 3.0;

  double eig_min = 0.0;
  if (p1 < 1e-18) {
    // 対角行列: 固有値は対角成分。 最小成分の軸が法線。
    int min_axis = 0;
    double min_value = a(0, 0);
    if (a(1, 1) < min_value) { min_value = a(1, 1); min_axis = 1; }
    if (a(2, 2) < min_value) { min_value = a(2, 2); min_axis = 2; }
    ok = true;
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
    normal(min_axis) = 1.0;
    return normal;
  }

  const double p2 =
    (a(0, 0) - q) * (a(0, 0) - q) +
    (a(1, 1) - q) * (a(1, 1) - q) +
    (a(2, 2) - q) * (a(2, 2) - q) + 2.0 * p1;
  const double p = std::sqrt(p2 / 6.0);
  if (p < 1e-12) {
    return Eigen::Vector3d::UnitZ();  // 実質等方。 ok=false のまま。
  }

  // B = (A - qI) / p の行列式 → r = det(B)/2 ∈ [-1, 1]。
  Eigen::Matrix3d b = (a - q * Eigen::Matrix3d::Identity()) / p;
  double r = b.determinant() / 2.0;
  r = std::max(-1.0, std::min(1.0, r));
  const double phi = std::acos(r) / 3.0;

  const double eig_max = q + 2.0 * p * std::cos(phi);
  eig_min = q + 2.0 * p * std::cos(phi + 2.0 * M_PI / 3.0);
  // 中間固有値 = 3q - eig_max - eig_min (使わないが定義のため)。
  (void)eig_max;

  // 法線 = (A - eig_min I) の零空間。 3 行のうち 2 行の外積で最大ノルムのものを選ぶ。
  const Eigen::Matrix3d m = a - eig_min * Eigen::Matrix3d::Identity();
  const Eigen::Vector3d row0 = m.row(0);
  const Eigen::Vector3d row1 = m.row(1);
  const Eigen::Vector3d row2 = m.row(2);
  const Eigen::Vector3d cross01 = row0.cross(row1);
  const Eigen::Vector3d cross12 = row1.cross(row2);
  const Eigen::Vector3d cross20 = row2.cross(row0);
  const double norm01 = cross01.squaredNorm();
  const double norm12 = cross12.squaredNorm();
  const double norm20 = cross20.squaredNorm();

  Eigen::Vector3d normal = cross01;
  double best = norm01;
  if (norm12 > best) { best = norm12; normal = cross12; }
  if (norm20 > best) { best = norm20; normal = cross20; }
  if (best < 1e-18) {
    return Eigen::Vector3d::UnitZ();  // 退化。 ok=false。
  }
  ok = true;
  return normal.normalized();
}

// k 近傍点の集合から平面正則化共分散 I − (1−ε)·n·nᵀ を組む (CPU 参照)。
// neighbor_points は body 座標。 count < 3 や法線が定まらない場合は ε·I を返す。
Eigen::Matrix3d regularizedCovariance(
  const std::vector<Eigen::Vector3d> & neighbor_points,
  double plane_epsilon)
{
  const int count = static_cast<int>(neighbor_points.size());
  if (count < 3) {
    return Eigen::Matrix3d::Identity() * plane_epsilon;
  }
  Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  for (const Eigen::Vector3d & point : neighbor_points) {
    mean += point;
  }
  mean /= static_cast<double>(count);

  Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
  for (const Eigen::Vector3d & point : neighbor_points) {
    const Eigen::Vector3d deviation = point - mean;
    covariance.noalias() += deviation * deviation.transpose();
  }
  covariance /= static_cast<double>(count);

  bool ok = false;
  const Eigen::Vector3d normal = smallestEigenvectorSymmetric(covariance, ok);
  if (!ok) {
    return Eigen::Matrix3d::Identity() * plane_epsilon;
  }
  // I − (1−ε)·n·nᵀ : 法線方向の分散を ε に、 平面に沿う 2 方向を 1 に。
  return Eigen::Matrix3d::Identity() -
    (1.0 - plane_epsilon) * (normal * normal.transpose());
}

}  // namespace

PointHashGrid PointHashGrid::build(
  float cell_size_m,
  const std::vector<Eigen::Vector3d> & points_body)
{
  PointHashGrid grid;
  grid.cell_size_m = cell_size_m;
  grid.num_points = static_cast<int>(points_body.size());
  grid.points_xyz.assign(static_cast<std::size_t>(grid.num_points) * 3, 0.0f);

  // 有限な点だけをセルに割り当てる。 非有限点は points_xyz にそのまま入れておくが
  // (index 整合のため)、 どのセルにも属さない (= 近傍候補にならない)。
  struct CellEntry
  {
    Eigen::Vector3i coord;
    std::vector<std::int32_t> point_indices;
  };
  // セルキー (x,y,z) → エントリ。 順序非依存 (ハッシュ表で key 検索するため)。
  std::unordered_map<std::int64_t, CellEntry> cells;
  cells.reserve(points_body.size());

  auto packKey = [](const Eigen::Vector3i & c) -> std::int64_t {
    // 21bit ずつにパックした衝突しにくいキー (unordered_map 用、 GPU 探索には無関係)。
    const std::int64_t x = static_cast<std::int64_t>(c.x()) & 0x1FFFFF;
    const std::int64_t y = static_cast<std::int64_t>(c.y()) & 0x1FFFFF;
    const std::int64_t z = static_cast<std::int64_t>(c.z()) & 0x1FFFFF;
    return (x << 42) | (y << 21) | z;
  };

  for (int point_index = 0; point_index < grid.num_points; ++point_index) {
    const Eigen::Vector3d & point = points_body[point_index];
    grid.points_xyz[point_index * 3 + 0] = static_cast<float>(point.x());
    grid.points_xyz[point_index * 3 + 1] = static_cast<float>(point.y());
    grid.points_xyz[point_index * 3 + 2] = static_cast<float>(point.z());
    if (!point.allFinite()) {
      continue;
    }
    const Eigen::Vector3i coord = cellCoord(point, cell_size_m);
    CellEntry & entry = cells[packKey(coord)];
    entry.coord = coord;
    entry.point_indices.push_back(point_index);
  }

  const std::size_t num_cells = cells.size();
  // capacity = num_cells*2 以上の最小 2 冪 (最低 8)。 負荷率 <= 0.5。
  int capacity = 8;
  while (static_cast<std::size_t>(capacity) < num_cells * 2) {
    capacity <<= 1;
  }
  grid.capacity = capacity;
  grid.occupied.assign(capacity, 0);
  grid.cell_keys_xyz.assign(static_cast<std::size_t>(capacity) * 3, 0);
  grid.cell_point_offset.assign(capacity, 0);
  grid.cell_point_count.assign(capacity, 0);
  grid.point_indices.reserve(grid.num_points);

  for (const auto & key_entry : cells) {
    const CellEntry & entry = key_entry.second;
    std::uint32_t slot = hashCell(entry.coord.x(), entry.coord.y(), entry.coord.z(), capacity);
    while (grid.occupied[slot] != 0) {
      slot = (slot + 1) & static_cast<std::uint32_t>(capacity - 1);
    }
    grid.occupied[slot] = 1;
    grid.cell_keys_xyz[slot * 3 + 0] = entry.coord.x();
    grid.cell_keys_xyz[slot * 3 + 1] = entry.coord.y();
    grid.cell_keys_xyz[slot * 3 + 2] = entry.coord.z();
    grid.cell_point_offset[slot] = static_cast<std::int32_t>(grid.point_indices.size());
    grid.cell_point_count[slot] = static_cast<std::int32_t>(entry.point_indices.size());
    for (const std::int32_t point_index : entry.point_indices) {
      grid.point_indices.push_back(point_index);
    }
  }
  return grid;
}

std::vector<Eigen::Matrix3d> estimateSourceCovariancesGridCpu(
  const std::vector<Eigen::Vector3d> & points_body,
  const CovarianceEstimateConfig & config)
{
  const int num_points = static_cast<int>(points_body.size());
  std::vector<Eigen::Matrix3d> covariances(
    num_points, Eigen::Matrix3d::Identity() * static_cast<double>(config.plane_epsilon));
  if (num_points < 3) {
    return covariances;
  }

  const PointHashGrid grid = PointHashGrid::build(config.cell_size_m, points_body);
  const double cell_size = config.cell_size_m;
  const double plane_epsilon = config.plane_epsilon;
  const int radius = config.search_radius_cells;
  const int k = std::min(config.num_neighbors, kMaxNeighbors);
  const std::uint32_t mask = static_cast<std::uint32_t>(grid.capacity - 1);

  for (int point_index = 0; point_index < num_points; ++point_index) {
    const Eigen::Vector3d & query = points_body[point_index];
    if (!query.allFinite()) {
      continue;  // 既定の ε·I のまま。
    }
    const Eigen::Vector3i center = cellCoord(query, cell_size);

    // 27 近傍セルの候補点から k 近傍を距離で選ぶ (挿入による部分選択)。
    std::array<double, kMaxNeighbors> best_dist;
    std::array<int, kMaxNeighbors> best_index;
    int found = 0;
    best_dist.fill(0.0);

    for (int dz = -radius; dz <= radius; ++dz) {
      for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
          const Eigen::Vector3i coord(center.x() + dx, center.y() + dy, center.z() + dz);
          std::uint32_t slot = hashCell(coord.x(), coord.y(), coord.z(), grid.capacity);
          for (int probe = 0; probe < grid.capacity; ++probe) {
            if (grid.occupied[slot] == 0) {
              break;  // 空きスロット = このセルは未登録。
            }
            if (grid.cell_keys_xyz[slot * 3 + 0] == coord.x() &&
                grid.cell_keys_xyz[slot * 3 + 1] == coord.y() &&
                grid.cell_keys_xyz[slot * 3 + 2] == coord.z())
            {
              const int offset = grid.cell_point_offset[slot];
              const int count = grid.cell_point_count[slot];
              for (int local = 0; local < count; ++local) {
                const int candidate = grid.point_indices[offset + local];
                const double cx = grid.points_xyz[candidate * 3 + 0];
                const double cy = grid.points_xyz[candidate * 3 + 1];
                const double cz = grid.points_xyz[candidate * 3 + 2];
                const double ddx = cx - query.x();
                const double ddy = cy - query.y();
                const double ddz = cz - query.z();
                const double distance_sq = ddx * ddx + ddy * ddy + ddz * ddz;
                // k 近傍配列に挿入: 満杯なら最大より小さいときだけ末尾を置換。
                if (found < k) {
                  int pos = found++;
                  while (pos > 0 && best_dist[pos - 1] > distance_sq) {
                    best_dist[pos] = best_dist[pos - 1];
                    best_index[pos] = best_index[pos - 1];
                    --pos;
                  }
                  best_dist[pos] = distance_sq;
                  best_index[pos] = candidate;
                } else if (distance_sq < best_dist[k - 1]) {
                  int pos = k - 1;
                  while (pos > 0 && best_dist[pos - 1] > distance_sq) {
                    best_dist[pos] = best_dist[pos - 1];
                    best_index[pos] = best_index[pos - 1];
                    --pos;
                  }
                  best_dist[pos] = distance_sq;
                  best_index[pos] = candidate;
                }
              }
              break;  // key 一致セルを処理したので probe 終了。
            }
            slot = (slot + 1) & mask;
          }
        }
      }
    }

    if (found < 3) {
      continue;  // 既定の ε·I のまま。
    }
    std::vector<Eigen::Vector3d> neighbors;
    neighbors.reserve(found);
    for (int neighbor = 0; neighbor < found; ++neighbor) {
      const int candidate = best_index[neighbor];
      neighbors.emplace_back(
        grid.points_xyz[candidate * 3 + 0],
        grid.points_xyz[candidate * 3 + 1],
        grid.points_xyz[candidate * 3 + 2]);
    }
    covariances[point_index] = regularizedCovariance(neighbors, plane_epsilon);
  }
  return covariances;
}

#ifndef PYLOT_LIO_HAS_METAL

CovarianceEstimation estimateSourceCovariancesMetal(
  const std::vector<Eigen::Vector3d> & points_body,
  const CovarianceEstimateConfig & config)
{
  CovarianceEstimation result;
  result.covariances = estimateSourceCovariancesGridCpu(points_body, config);
  result.gpu_used = false;
  result.error_message = "Metal support not compiled in (PYLOT_LIO_HAS_METAL undefined)";
  return result;
}

struct MetalCovarianceEngine::Impl {};
MetalCovarianceEngine::MetalCovarianceEngine() : impl_(nullptr) {}
MetalCovarianceEngine::~MetalCovarianceEngine() = default;
bool MetalCovarianceEngine::isValid() const { return false; }
CovarianceEstimation MetalCovarianceEngine::estimate(
  const std::vector<Eigen::Vector3d> & points_body,
  const CovarianceEstimateConfig & config)
{
  CovarianceEstimation result;
  result.covariances = estimateSourceCovariancesGridCpu(points_body, config);
  result.gpu_used = false;
  result.error_message = "Metal support not compiled in (PYLOT_LIO_HAS_METAL undefined)";
  return result;
}

#else  // PYLOT_LIO_HAS_METAL

namespace
{

// カーネルに渡すスカラ群。 Metal 側 struct とレイアウトを完全一致させる (全 4 byte)。
struct CovKernelParams
{
  float cell_size;
  float inv_cell_size;
  float plane_epsilon;
  std::int32_t search_radius;
  std::uint32_t capacity;
  std::uint32_t num_points;
  std::int32_t num_neighbors;
  std::int32_t padding;
};

constexpr int kThreadgroupSize = 128;

const char * kCovKernelSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

constant int kMaxNeighbors = 32;

struct CovKernelParams {
  float cell_size;
  float inv_cell_size;
  float plane_epsilon;
  int   search_radius;
  uint  capacity;
  uint  num_points;
  int   num_neighbors;
  int   padding;
};

inline uint hash_cell(int x, int y, int z, uint capacity) {
  uint h = (uint(x) * 73856093u) ^ (uint(y) * 19349663u) ^ (uint(z) * 83492791u);
  return h & (capacity - 1u);
}

// 対称 3x3 の最小固有値に対応する単位固有ベクトル (法線) を closed-form で。
// CPU 参照 smallestEigenvectorSymmetric と同一アルゴリズム (fp32)。 ok=false で退化。
inline float3 smallest_eigenvector(float3x3 A, thread bool & ok) {
  ok = false;
  float a00 = A[0][0], a11 = A[1][1], a22 = A[2][2];
  float a01 = A[1][0], a02 = A[2][0], a12 = A[2][1];  // 対称: A[col][row]
  float p1 = a01 * a01 + a02 * a02 + a12 * a12;
  float tr = a00 + a11 + a22;
  float q = tr / 3.0f;

  if (p1 < 1e-18f) {
    // 対角行列: 最小対角成分の軸が法線。
    int min_axis = 0;
    float min_value = a00;
    if (a11 < min_value) { min_value = a11; min_axis = 1; }
    if (a22 < min_value) { min_value = a22; min_axis = 2; }
    ok = true;
    return float3(min_axis == 0 ? 1.0f : 0.0f,
                  min_axis == 1 ? 1.0f : 0.0f,
                  min_axis == 2 ? 1.0f : 0.0f);
  }

  float p2 = (a00 - q) * (a00 - q) + (a11 - q) * (a11 - q) + (a22 - q) * (a22 - q)
             + 2.0f * p1;
  float p = sqrt(p2 / 6.0f);
  if (p < 1e-12f) { return float3(0.0f, 0.0f, 1.0f); }

  // B = (A - qI)/p, r = det(B)/2 ∈ [-1,1]
  float3x3 B = (A - q * float3x3(1.0f)) * (1.0f / p);
  float detB =
      B[0][0] * (B[1][1] * B[2][2] - B[2][1] * B[1][2])
    - B[1][0] * (B[0][1] * B[2][2] - B[2][1] * B[0][2])
    + B[2][0] * (B[0][1] * B[1][2] - B[1][1] * B[0][2]);
  float r = detB / 2.0f;
  r = clamp(r, -1.0f, 1.0f);
  float phi = acos(r) / 3.0f;
  float eig_min = q + 2.0f * p * cos(phi + 2.0f * 3.14159265358979323846f / 3.0f);

  // 法線 = (A - eig_min I) の零空間 (行ペアの外積で最大ノルム)。
  float3x3 M = A - eig_min * float3x3(1.0f);
  // M[col][row] 表現の「行ベクトル」を取り出す。
  float3 row0 = float3(M[0][0], M[1][0], M[2][0]);
  float3 row1 = float3(M[0][1], M[1][1], M[2][1]);
  float3 row2 = float3(M[0][2], M[1][2], M[2][2]);
  float3 c01 = cross(row0, row1);
  float3 c12 = cross(row1, row2);
  float3 c20 = cross(row2, row0);
  float n01 = dot(c01, c01);
  float n12 = dot(c12, c12);
  float n20 = dot(c20, c20);
  float3 normal = c01;
  float best = n01;
  if (n12 > best) { best = n12; normal = c12; }
  if (n20 > best) { best = n20; normal = c20; }
  if (best < 1e-18f) { return float3(0.0f, 0.0f, 1.0f); }
  ok = true;
  return normalize(normal);
}

kernel void estimate_covariance(
    device const float * points        [[buffer(0)]],  // num_points * 3
    device const uchar * occupied      [[buffer(1)]],  // capacity
    device const int   * cell_keys     [[buffer(2)]],  // capacity * 3
    device const int   * cell_offset   [[buffer(3)]],  // capacity
    device const int   * cell_count    [[buffer(4)]],  // capacity
    device const int   * point_indices [[buffer(5)]],  // num_points
    constant CovKernelParams & params  [[buffer(6)]],
    device float       * out_covs      [[buffer(7)]],  // num_points * 9 (row-major)
    uint gid [[thread_position_in_grid]])
{
  if (gid >= params.num_points) { return; }

  float eps = params.plane_epsilon;
  // 既定値: ε·I。
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      out_covs[gid * 9 + r * 3 + c] = (r == c) ? eps : 0.0f;
    }
  }

  float3 query = float3(points[gid * 3 + 0], points[gid * 3 + 1], points[gid * 3 + 2]);
  if (!(isfinite(query.x) && isfinite(query.y) && isfinite(query.z))) { return; }

  int cx = int(floor(query.x * params.inv_cell_size));
  int cy = int(floor(query.y * params.inv_cell_size));
  int cz = int(floor(query.z * params.inv_cell_size));

  int k = min(params.num_neighbors, kMaxNeighbors);
  uint mask = params.capacity - 1u;
  int radius = params.search_radius;

  float best_dist[kMaxNeighbors];
  float3 best_pt[kMaxNeighbors];
  int found = 0;

  for (int dz = -radius; dz <= radius; ++dz) {
    for (int dy = -radius; dy <= radius; ++dy) {
      for (int dx = -radius; dx <= radius; ++dx) {
        int vx = cx + dx, vy = cy + dy, vz = cz + dz;
        uint slot = hash_cell(vx, vy, vz, params.capacity);
        for (uint probe = 0; probe < params.capacity; ++probe) {
          if (occupied[slot] == 0) { break; }
          if (cell_keys[slot * 3 + 0] == vx &&
              cell_keys[slot * 3 + 1] == vy &&
              cell_keys[slot * 3 + 2] == vz) {
            int offset = cell_offset[slot];
            int count = cell_count[slot];
            for (int local = 0; local < count; ++local) {
              int candidate = point_indices[offset + local];
              float3 cp = float3(points[candidate * 3 + 0],
                                 points[candidate * 3 + 1],
                                 points[candidate * 3 + 2]);
              float3 diff = cp - query;
              float dist_sq = dot(diff, diff);
              if (found < k) {
                int pos = found++;
                while (pos > 0 && best_dist[pos - 1] > dist_sq) {
                  best_dist[pos] = best_dist[pos - 1];
                  best_pt[pos] = best_pt[pos - 1];
                  --pos;
                }
                best_dist[pos] = dist_sq;
                best_pt[pos] = cp;
              } else if (dist_sq < best_dist[k - 1]) {
                int pos = k - 1;
                while (pos > 0 && best_dist[pos - 1] > dist_sq) {
                  best_dist[pos] = best_dist[pos - 1];
                  best_pt[pos] = best_pt[pos - 1];
                  --pos;
                }
                best_dist[pos] = dist_sq;
                best_pt[pos] = cp;
              }
            }
            break;
          }
          slot = (slot + 1u) & mask;
        }
      }
    }
  }

  if (found < 3) { return; }  // 既定の ε·I のまま。

  float3 mean = float3(0.0f);
  for (int n = 0; n < found; ++n) { mean += best_pt[n]; }
  mean /= float(found);

  // 共分散 (母): Σ (p-μ)(p-μ)ᵀ / found。 float3x3 は col-major。
  float3x3 cov = float3x3(0.0f);
  for (int n = 0; n < found; ++n) {
    float3 dv = best_pt[n] - mean;
    cov[0][0] += dv.x * dv.x; cov[1][0] += dv.x * dv.y; cov[2][0] += dv.x * dv.z;
    cov[0][1] += dv.y * dv.x; cov[1][1] += dv.y * dv.y; cov[2][1] += dv.y * dv.z;
    cov[0][2] += dv.z * dv.x; cov[1][2] += dv.z * dv.y; cov[2][2] += dv.z * dv.z;
  }
  float inv_n = 1.0f / float(found);
  for (int c = 0; c < 3; ++c) {
    for (int r = 0; r < 3; ++r) { cov[c][r] *= inv_n; }
  }

  bool ok = false;
  float3 normal = smallest_eigenvector(cov, ok);
  if (!ok) { return; }  // 退化: ε·I のまま。

  // I − (1−ε)·n·nᵀ を row-major で書き出す。
  float w = 1.0f - eps;
  for (int r = 0; r < 3; ++r) {
    float nr = (r == 0) ? normal.x : ((r == 1) ? normal.y : normal.z);
    for (int c = 0; c < 3; ++c) {
      float nc = (c == 0) ? normal.x : ((c == 1) ? normal.y : normal.z);
      float identity = (r == c) ? 1.0f : 0.0f;
      out_covs[gid * 9 + r * 3 + c] = identity - w * nr * nc;
    }
  }
}
)METAL";

}  // namespace

struct MetalCovarianceEngine::Impl
{
  MTL::Device * device = nullptr;
  MTL::CommandQueue * queue = nullptr;
  MTL::Library * library = nullptr;
  MTL::Function * function = nullptr;
  MTL::ComputePipelineState * pso = nullptr;
  MTL::Buffer * buf_params = nullptr;

  static void releaseBuffer(MTL::Buffer *& buffer)
  {
    if (buffer != nullptr) {
      buffer->release();
      buffer = nullptr;
    }
  }

  ~Impl()
  {
    releaseBuffer(buf_params);
    if (pso != nullptr) { pso->release(); }
    if (function != nullptr) { function->release(); }
    if (library != nullptr) { library->release(); }
    if (queue != nullptr) { queue->release(); }
    if (device != nullptr) { device->release(); }
  }
};

MetalCovarianceEngine::MetalCovarianceEngine() : impl_(std::make_unique<Impl>())
{
  NS::AutoreleasePool * pool = NS::AutoreleasePool::alloc()->init();
  impl_->device = MTL::CreateSystemDefaultDevice();
  if (impl_->device != nullptr) {
    NS::Error * compile_error = nullptr;
    NS::String * source = NS::String::string(kCovKernelSource, NS::UTF8StringEncoding);
    impl_->library = impl_->device->newLibrary(source, nullptr, &compile_error);
    if (impl_->library != nullptr) {
      impl_->function = impl_->library->newFunction(
        NS::String::string("estimate_covariance", NS::UTF8StringEncoding));
      NS::Error * pso_error = nullptr;
      impl_->pso = impl_->device->newComputePipelineState(impl_->function, &pso_error);
      impl_->queue = impl_->device->newCommandQueue();
      impl_->buf_params = impl_->device->newBuffer(
        sizeof(CovKernelParams), MTL::ResourceStorageModeShared);
    }
  }
  pool->release();
}

MetalCovarianceEngine::~MetalCovarianceEngine() = default;

bool MetalCovarianceEngine::isValid() const
{
  return impl_ && impl_->device != nullptr && impl_->pso != nullptr &&
    impl_->queue != nullptr && impl_->buf_params != nullptr;
}

CovarianceEstimation MetalCovarianceEngine::estimate(
  const std::vector<Eigen::Vector3d> & points_body,
  const CovarianceEstimateConfig & config)
{
  CovarianceEstimation result;
  const int num_points = static_cast<int>(points_body.size());
  if (!isValid() || num_points < 3) {
    // デバイス無 / 点が少ない: CPU 参照にフォールバック。
    result.covariances = estimateSourceCovariancesGridCpu(points_body, config);
    result.gpu_used = false;
    if (!isValid()) {
      result.error_message = "engine not valid (no Metal device / PSO); CPU fallback";
    }
    return result;
  }

  NS::AutoreleasePool * pool = NS::AutoreleasePool::alloc()->init();

  const PointHashGrid grid = PointHashGrid::build(config.cell_size_m, points_body);

  auto make_buffer = [&](const void * data, std::size_t bytes) {
    return impl_->device->newBuffer(data, bytes, MTL::ResourceStorageModeShared);
  };
  MTL::Buffer * buf_points =
    make_buffer(grid.points_xyz.data(), grid.points_xyz.size() * sizeof(float));
  MTL::Buffer * buf_occupied =
    make_buffer(grid.occupied.data(), grid.occupied.size());
  MTL::Buffer * buf_keys =
    make_buffer(grid.cell_keys_xyz.data(), grid.cell_keys_xyz.size() * sizeof(std::int32_t));
  MTL::Buffer * buf_offset =
    make_buffer(grid.cell_point_offset.data(), grid.cell_point_offset.size() * sizeof(std::int32_t));
  MTL::Buffer * buf_count =
    make_buffer(grid.cell_point_count.data(), grid.cell_point_count.size() * sizeof(std::int32_t));
  MTL::Buffer * buf_pidx =
    make_buffer(grid.point_indices.data(), grid.point_indices.size() * sizeof(std::int32_t));
  MTL::Buffer * buf_out = impl_->device->newBuffer(
    static_cast<std::size_t>(num_points) * 9 * sizeof(float), MTL::ResourceStorageModeShared);

  CovKernelParams * params = static_cast<CovKernelParams *>(impl_->buf_params->contents());
  params->cell_size = config.cell_size_m;
  params->inv_cell_size = 1.0f / config.cell_size_m;
  params->plane_epsilon = config.plane_epsilon;
  params->search_radius = config.search_radius_cells;
  params->capacity = static_cast<std::uint32_t>(grid.capacity);
  params->num_points = static_cast<std::uint32_t>(num_points);
  params->num_neighbors = std::min(config.num_neighbors, kMaxNeighbors);
  params->padding = 0;

  MTL::CommandBuffer * command_buffer = impl_->queue->commandBuffer();
  MTL::ComputeCommandEncoder * encoder = command_buffer->computeCommandEncoder();
  encoder->setComputePipelineState(impl_->pso);
  encoder->setBuffer(buf_points, 0, 0);
  encoder->setBuffer(buf_occupied, 0, 1);
  encoder->setBuffer(buf_keys, 0, 2);
  encoder->setBuffer(buf_offset, 0, 3);
  encoder->setBuffer(buf_count, 0, 4);
  encoder->setBuffer(buf_pidx, 0, 5);
  encoder->setBuffer(impl_->buf_params, 0, 6);
  encoder->setBuffer(buf_out, 0, 7);
  const int num_groups = (num_points + kThreadgroupSize - 1) / kThreadgroupSize;
  const MTL::Size grid_size = MTL::Size::Make(static_cast<NS::UInteger>(num_groups), 1, 1);
  const MTL::Size tg_size = MTL::Size::Make(kThreadgroupSize, 1, 1);
  encoder->dispatchThreadgroups(grid_size, tg_size);
  encoder->endEncoding();
  command_buffer->commit();
  command_buffer->waitUntilCompleted();

  const float * out = static_cast<const float *>(buf_out->contents());
  result.covariances.resize(num_points);
  for (int point_index = 0; point_index < num_points; ++point_index) {
    Eigen::Matrix3d covariance;
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        covariance(r, c) = static_cast<double>(out[point_index * 9 + r * 3 + c]);
      }
    }
    result.covariances[point_index] = covariance;
  }
  result.gpu_used = true;

  Impl::releaseBuffer(buf_points);
  Impl::releaseBuffer(buf_occupied);
  Impl::releaseBuffer(buf_keys);
  Impl::releaseBuffer(buf_offset);
  Impl::releaseBuffer(buf_count);
  Impl::releaseBuffer(buf_pidx);
  Impl::releaseBuffer(buf_out);
  pool->release();
  return result;
}

CovarianceEstimation estimateSourceCovariancesMetal(
  const std::vector<Eigen::Vector3d> & points_body,
  const CovarianceEstimateConfig & config)
{
  CovarianceEstimation result;
  if (points_body.size() < 3) {
    result.covariances = estimateSourceCovariancesGridCpu(points_body, config);
    result.gpu_used = false;
    return result;
  }
  // 単発用途: 一時エンジンを作って 1 回だけ走らせる (PSO コンパイルを毎回含む)。
  // 毎フレーム使うなら MetalCovarianceEngine を保持して使い回すこと。
  MetalCovarianceEngine engine;
  return engine.estimate(points_body, config);
}

#endif  // PYLOT_LIO_HAS_METAL

}  // namespace pylot_lio::gpu
