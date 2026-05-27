# pylot_lio アーキテクチャ

設計の意図と、4 つの抽象インタフェース + Factory パターンがどう動くかを説明します。

## 設計目標 (前提)

1. **比較容易性**: 同じ rosbag に対して 4 つのレイヤーすべてを差し替えて結果が変わるさまを観察できる
2. **テスト容易性**: 各レイヤーは ROS 抜きの純ロジックとして単体テストできる
3. **依存最小化**: small_gicp や SYCL コンパイラが無い環境でもコア機能はビルドできる
4. **読みやすさ**: 学生やジュニアエンジニアが「LIO の各レイヤーがそれぞれ何をするか」を 1 ファイル読めば把握できる

## 抽象インタフェース 4 つ

| 名前 | ファイル | 責務 |
|------|----------|------|
| `IPreprocessor` | `include/pylot_lio/preprocess/i_preprocessor.hpp` | 1 スキャン点群の代表点抽出 |
| `IPointCloudMap` | `include/pylot_lio/map/i_point_cloud_map.hpp` | world フレームの累積マップ + 最近傍検索 |
| `IRegistration` | `include/pylot_lio/registration/i_registration.hpp` | source 点群 vs マップの姿勢最適化 |
| `IStateEstimator` | `include/pylot_lio/estimator/i_state_estimator.hpp` | IMU 予測 + スキャン更新で T_world_body を推定 |

すべて `std::unique_ptr` で持ち、`pylot_lio::LioNode` はインタフェースに対してしか書かれていません。

## データフロー

```
 sensor_msgs::Imu          sensor_msgs::PointCloud2
   ── onImu() ─┐               ── onLidarCloud() ─┐
              ▼                                  ▼
        IStateEstimator                      IPreprocessor
        ::predictWithImu                     ::process
                                                  ▼
                                            (downsampled scan)
                                                  ▼
        ┌────────────────────────────────────────┘
        ▼
   IStateEstimator::updateWithScan
        ├── (IESKF/HGO)  → IPointCloudMap::findNearestNeighbor を反復で叩く
        │                 → 共分散付き観測残差から ESKF 更新
        │                 → IPointCloudMap::insertScan で蓄積
        └── (GICP-only)  → IRegistration::align を 1 回呼ぶだけ
                          → 結果を current_state にコピー
                          → IPointCloudMap::insertScan で蓄積
        ▼
   publishOutputs(stamp)
        ├── nav_msgs::Odometry → /lio/odom
        ├── PointCloud2        → /lio/cloud_world (10 フレーム毎)
        ├── std_msgs::String   → /lio/diag (1 スキャン毎)
        └── TF: world → base_link
```

> **TF の責務**: lio_node が publish する TF は `world → base_link`（オドメトリ結果）のみ。
> センサ取り付けの static TF（`base_link → lidar` / `base_link → imu`）は**出さない** ──
> それを出す責務は外部の `robot_state_publisher` / URDF にあるため。lio_node は
> `extrinsic_source="tf"` のとき TF tree から `imu_frame ← lidar_frame` を **lookup（消費）** するだけで、
> `extrinsic_source="config"`（既定）なら yaml の `extrinsic_*_imu_from_lidar` を直接使う。

並列バックエンド（OpenMP / Intel TBB）の切替と性能比較は [PARALLEL_BACKENDS.md](PARALLEL_BACKENDS.md) を参照。

## Factory パターン

`include/pylot_lio/factory.hpp` の `createBackendsFromConfig(LioBackendConfig)` が単一エントリポイント。

- `preprocessor_name`, `map_name`, `registration_name`, `state_estimator_name` の 4 つの文字列で組み合わせを指定
- 未知の文字列は `std::invalid_argument` を投げる → ノード起動時に検出できる
- `small_gicp` が build 時に無いと `small_gicp_*` は登録されない → factory 側で例外
- `sycl` バックエンドは macOS では stub なので、生成時は `plain_gicp` にフォールバックする

利点:
- ノード本体 (`src/lio_node.cpp`) はバックエンドの具象クラスを 1 行も知らない
- テストでは `LioBackendConfig` を直接組み立てれば factory が ROS 抜きで動く (`test/test_factory.cpp`)

## CMake 構造

`CMakeLists.txt` のレイアウト:

1. `pylot_lio_core` SHARED ライブラリに全レイヤー実装を含める
2. `find_package(small_gicp QUIET)` が成功した場合のみ `SmallGicpRegistration` をソースに加え、`PYLOT_LIO_HAS_SMALL_GICP` を定義
3. `CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM"` の場合のみ `SyclRegistration` 本体を加え、それ以外は `sycl_registration_stub.cpp` を代わりに含める
4. `lio_node` 実行体は `pylot_lio_core` だけにリンク

これにより:
- 「small_gicp 未インストール」「SYCL 未インストール」のいずれでも `pylot_lio_core` は必ずビルドできる
- 環境差は CMake 段階で吸収され、`lio_node.cpp` は無条件にコンパイル可能

## テスト戦略

| テスト | 目的 |
|--------|------|
| `test_lie_algebra` | SO(3) exp/log/Jacobian の数値検証 |
| `test_voxel_grid_preprocessor` | ボクセル境界・空入力・NaN 処理 |
| `test_random_sampling_preprocessor` | 目標点数・再現性 (seed) ・空入力 |
| `test_voxel_map` | オンライン共分散更新が batch と一致するか、共分散 floor 適用 |
| `test_kd_tree_map` | 最近傍探索の基本動作 |
| `test_ieskf_estimator` | 重力相殺の確認、scan 更新で map に挿入が走ること |
| `test_gicp_only_estimator` | predictWithImu が no-op であること、初期化動作 |
| `test_factory` | 既知名で生成成功、未知名で例外 |
| `test_small_gicp_registration` | (small_gicp あれば) 自己照合で単位変換に近づく |

すべて ROS を起動しない純粋 gtest。`BUILD_TESTING=ON` で `cbt pylot_lio` から実行。

## 拡張ポイント

新しい状態推定アルゴリズムを試したい場合:

1. `IStateEstimator` を継承するクラスを作る
2. `src/factory.cpp` の `buildStateEstimator` に case を追加
3. `config/presets/<your_name>.yaml` を作る
4. テストを書く

他レイヤーも同じパターン。1 ファイル増えるだけで `lio_node.cpp` は変えなくて良い。
