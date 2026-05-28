# pylot_lio

Livox Mid-360 向けの **比較可能な LiDAR-Inertial Odometry**。
4 つのレイヤー (前処理 / マップ / マッチング / 状態推定) すべてを ROS パラメータで差し替えられ、同じ rosbag 上で複数手法を公平に比較できます。

- インタフェース: `/livox/lidar` (sensor_msgs/PointCloud2 または Livox CustomMsg) + `/livox/imu` (sensor_msgs/Imu) を購読し、`/lio/odom` (nav_msgs/Odometry) と `/lio/cloud_world` (sensor_msgs/PointCloud2) を publish
- 出力 frame: `world -> base_link` TF
- ROS 2 Jazzy 想定、macOS arm64 (pixi/robostack) と Linux で動作
- マッチングの並列バックエンドを **OpenMP / Intel TBB / Apple Metal GPU** で切替可能
- **オフライン rosbag リーダ** (`lio_rosbag`): bag を直読みし再生レートに依存せず全フレームを処理 (GLIM の glim_rosbag 相当)

## 起動方法

```bash
cd /Users/nyanziba/hobby/library
source ./activate_pixi_env.sh
cb --packages-select pylot_lio
source install/setup.zsh

# プリセット指定 (config/presets/*.yaml に対応)
ros2 launch pylot_lio lio.launch.py preset:=ieskf_smallgicp_voxel
ros2 launch pylot_lio lio.launch.py preset:=hgo_plaingicp_normal
ros2 launch pylot_lio lio.launch.py preset:=gicp_only_kdtree

# 別端末で rosbag 再生
ros2 bag play /path/to/mid360_loop.bag
```

## 用意されているプリセット

| Preset | 状態推定 | マッチング | マップ | 用途 |
|--------|----------|-----------|--------|------|
| `ieskf_smallgicp_voxel` | IESKF | small_gicp VGICP | VoxelMap | 標準推奨 |
| `ieskf_smallgicp_voxelrandom` | IESKF | small_gicp VGICP | VoxelRandomMap | voxel 内ランダム間引き版 |
| `hgo_plaingicp_normal` | HGO | plain GICP | NormalMap | 外部依存なしで動く |
| `gicp_only_kdtree` | GICP-only | plain GICP | KdTreeMap | 最小ベースライン |
| `gicp_only_voxel` | GICP-only | small_gicp VGICP | VoxelMap | GICP-only の voxel 版 |
| `gicp_only_degeneracy_robust` | GICP-only | plain GICP (縮退正則化) | VoxelMap | 廊下など縮退環境向け |
| `plain_gicp_openmp` | GICP-only | plain GICP (**OpenMP**) | VoxelMap | 外部依存ゼロ + マルチスレッド |
| `plain_gicp_tbb` | GICP-only | plain GICP (**Intel TBB**) | VoxelMap | TBB 並列 (omp と公平比較用) |
| `metal_vgicp` | GICP-only | **Metal GPU VGICP** | VoxelMap | Apple GPU。大規模点群で高速 |
| `mid360_metal_full` | GICP-only | **Metal GPU 全部のせ** | VoxelMap | Mid-360 で前処理〜registration を全部 GPU |
| `velodyne_metal_full` | GICP-only | **Metal GPU 全部のせ** | VoxelMap | Velodyne で前処理〜registration を全部 GPU |
| `velodyne_xsens` | GICP-only | small_gicp GICP | VoxelKeyframeSubmap | Velodyne + xsens rosbag 向け |

すべて `config/mid360.yaml` に既定値があり、preset YAML が個別パラメータを上書きします。

### Apple Metal GPU バックエンド (`metal_vgicp`)

`glim` / `gtsam_points` の GaussianVoxelMap + VGICP を Metal (metal-cpp) に移植したもの。**前処理 (source 共分散推定) と各 Gauss-Newton 反復の正規方程式 (H, b) 構築の両方**を GPU にオフロードします。

- **どのマップでも使えます** (マップ非依存)。`voxel_map` / `normal_map` は保存済みのボクセルガウス分布を直接、`voxel_random_map` / `voxel_keyframe_submap` / `kd_tree_map` は `toPointCloud()` の生点群を `map_voxel_size_m` でボクセル化してガウス分布を復元し GPU へ送ります。`voxel_keyframe_submap` を選べば **loop closure / PGO / PCD フル保存を維持したまま GPU registration** が使えます。
- fp32 精度 (Apple GPU は fp64 不可) ですが、CPU 版 (fp64) と **mm / mrad オーダーで一致**。
- **前処理 GPU 化** (`registration_metal_gpu_source_covariance`、既定 `true`): 生点群の k 近傍探索 + 平面正則化を Metal で実行 (これまで CPU の PCL KdTree が担っていた最大のボトルネック)。近傍探索は KdTree ではなく空間ハッシュグリッド (glim と同じ妥協)。平面正則化は固有値分解を避け、最小固有値方向の法線 `n` だけを closed-form で求め `I−(1−ε)nnᵀ` を組む。Apple M4 ベンチ: **2万点 5.1x / 9万点 8.2x / 41万点 11.6x 高速** (vs 逐次 CPU)。
- registration の線形化: 点数に応じ CPU/GPU を自動選択 (`registration_metal_gpu_min_points`、既定 5万点)。`metal_vgicp.yaml` は GPU-only (`0`)。
- Apple M4 ベンチ (align end-to-end): 5万点 2.6x / 20万点 2.9x 高速 (vs 逐次 plain_gicp)。
- Metal 非対応ビルド / デバイス無では自動的に CPU (前処理はグリッド kNN、registration は `plain_gicp`) にフォールバック。

```bash
ros2 launch pylot_lio lio.launch.py preset:=metal_vgicp
```

### Metal GPU ボクセルダウンサンプリング (`voxel_random_sampling`)

前処理レイヤーの `voxel_random_sampling` も Metal GPU で高速化できます (GLIM の randomgrid downsampling 相当)。

- **サンプリング率 0〜1** (`voxel_random_sampling_rate`): 各ボクセルから `m = max(1, round(点数 × rate))` 点を保持。`0` で従来の「各ボクセル 1 点」(後方互換)、`1` で全点。空間的に均一に間引きつつ密集ボクセルほど多く残せます。
- **GPU 化** (`voxel_random_sampling_use_gpu`、既定 `true`): ボクセル振り分け + reservoir sampling を Metal で実行。出力点数が可変なので、各ボクセルの保持数の prefix sum で出力位置を確定させ、各 GPU スレッドが衝突なく書き込みます。
- 整数演算 (reservoir + xorshift32) のみなので **GPU と CPU の結果は完全一致** (fp 誤差なし)。GPU 無効ビルド / デバイス無では同じグリッド方式を CPU で実行。

## オフライン rosbag 処理 (高速・取りこぼしなし)

`lio_rosbag` は rosbag2 を **直接読み**、点群 / IMU を記録順に LIO pipeline へ投入します。`ros2 bag play` / DDS を経由しないため再生レートに縛られず、**ハードが許す限り高速に全フレームを処理**します (GLIM の `glim_rosbag` 相当)。`ros2 bag play` 経由だと処理が間に合わないとキュー溢れでフレームが落ちますが、こちらは原理的に取りこぼしません。

```bash
# Livox Mid-360 の bag を GPU-only metal_vgicp でオフライン処理
ros2 launch pylot_lio lio_rosbag.launch.py \
    bag:=/path/to/livox_bag \
    preset:=metal_vgicp \
    input_cloud_topic:=/livox/lidar \
    input_imu_topic:=/livox/imu \
    input_cloud_format:=livox_custom

# Velodyne + xsens の bag (PointCloud2)
ros2 launch pylot_lio lio_rosbag.launch.py \
    bag:=/path/to/velodyne_bag \
    preset:=velodyne_xsens \
    input_cloud_topic:=/velodyne_points \
    input_imu_topic:=/imu/data \
    input_cloud_format:=pointcloud2
```

別端末で RViz を開き `/lio/cloud_world` `/lio/odom` を表示すれば結果が確認できます。

## ベンチマーク

```bash
# 1. 各プリセットで rosbag を順次走らせ、軌跡を保存
python3 scripts/run_benchmark.py \
    --bag /path/to/mid360_loop.bag \
    --output_dir /tmp/pylot_lio_bench \
    --presets ieskf_smallgicp_voxel hgo_plaingicp_normal gicp_only_kdtree

# 2. 手法間の相対指標を計算
python3 scripts/compute_relative_metrics.py \
    --input_dir /tmp/pylot_lio_bench \
    --reference ieskf_smallgicp_voxel \
    --output_json /tmp/pylot_lio_bench/metrics.json

# 3. HTML ダッシュボードに集約 (Plotly 任意)
python3 scripts/make_dashboard.py \
    --input_dir /tmp/pylot_lio_bench \
    --metrics /tmp/pylot_lio_bench/metrics.json \
    --output_html /tmp/pylot_lio_bench/dashboard.html
```

Mid-360 単体では Ground Truth が無いため、評価は「手法間の相対 RMSE」と「ループドリフト (始点と末尾の距離)」の 2 軸です。RTK や Motion Capture を後付けすれば `compute_relative_metrics.py` を拡張して絶対 ATE/RPE が計算できます。

## ドキュメント

| ファイル | 内容 |
|----------|------|
| [docs/ALGORITHMS.md](docs/ALGORITHMS.md) | **線形代数を知らない人向け** のアルゴリズム解説。LIO 全体像、4 レイヤーの役割、各手法の長所短所を日常語で説明 |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 抽象インタフェース + Factory パターンの設計意図、データフロー、CMake 構造、拡張ポイント |
| [docs/PARALLEL_BACKENDS.md](docs/PARALLEL_BACKENDS.md) | registration の並列バックエンド (OpenMP / Intel TBB) の切替方法・実装・実 bag での性能比較 (計測環境含む) |
| [docs/OVERVIEW.md](docs/OVERVIEW.md) / [docs/MATH.md](docs/MATH.md) | 全体像 / 数式の詳細 |

## ビルド要件

必須:
- ROS 2 (Jazzy 推奨)
- Eigen3, PCL (common, io, filters, kdtree)
- C++17 コンパイラ

任意 (見つからなければ該当バックエンドが無効化される):
- **small_gicp**: `small_gicp_gicp` / `small_gicp_vgicp` バックエンド用。
  システムインストール (conda-forge / robostack) を `find_package` で優先し、
  見つからなければ `3rdparty/small_gicp` (submodule, v1.0.0 に pin) を
  `add_subdirectory` で取り込みます。
- **Intel oneAPI (icpx)**: `sycl` バックエンド用 (Linux のみ)。
  検出時は `3rdparty/sycl_points` (submodule, header-only, Apache-2.0) も
  自動で include path に追加されます。
- **Intel TBB (oneTBB)**: `registration_parallel_backend: "tbb"` 用。
  見つからなければ "tbb" 指定でも OpenMP/逐次にフォールバック。
- **Apple Metal**: `metal_vgicp` バックエンド用 (macOS のみ)。
  `3rdparty/metal-cpp` (vendor 済みヘッダオンリー, Apache-2.0) と
  Metal/Foundation フレームワークを使用。非 Apple では `plain_gicp` にフォールバック。
- **rosbag2_cpp**: オフライン rosbag リーダ `lio_rosbag` 用。
  見つからなければ実行ファイル `lio_rosbag` をビルドしません (ライブ `lio_node` は影響なし)。

### 3rdparty submodule の初期化

最初の clone 直後は以下を実行してください:

```bash
git submodule update --init --recursive
```

| パス | 用途 | pin |
|---|---|---|
| `3rdparty/small_gicp` | VGICP/GICP の実装 (koide3/small_gicp) | tag `v1.0.0` |
| `3rdparty/sycl_points` | SYCL ベースの点群ユーティリティ (fateshelled/sycl_points, Apache-2.0) | `main` HEAD |
| `3rdparty/metal-cpp` | Apple Metal の C++ ラッパ (vendor 済みヘッダ, Apache-2.0) | macOS SDK 同梱版 |

システムに小型 GICP がインストールされている環境では submodule のビルドは走らない (find_package が先に見つけるため)。submodule は **オフラインビルド / システムインストール不能環境** のためのフォールバックです。

macOS arm64 + pixi (robostack) 環境では `activate_pixi_env.sh` の手順を踏まないと Apple Clang が選ばれてビルドが落ちます。詳しくは workspace ルートの [`CLAUDE.md`](../../CLAUDE.md) を参照してください。

## テスト

```bash
cbt pylot_lio
```

純ロジックの単体テスト (Lie 代数、各前処理 / マップ、IESKF、GICP-only、Factory、plain_gicp 縮退正則化、Scan Context、PGO、Livox 変換、Metal compute プローブ・VGICP 線形化・registration・source 共分散推定・voxel ダウンサンプリング など)。ROS ノードは起動しないので CI で安全に流せます。Metal 対応ビルドでは GPU=CPU の数値一致や GPU-only/CPU フォールバックも検証します (Metal 無しビルドでは該当テストはスタブ)。

## 参考にしたもの

実装は標準教科書 (Sola 2017, Barfoot 2017, Koide VGICP 2021, FAST-LIO2 2022) の数式から自前で書き起こしています。動作概念のみ参考にしたプロジェクト:

- [small_gicp](https://github.com/koide3/small_gicp) (MIT) — VGICP/GICP の参照実装。`SmallGicpRegistration` で直接リンク
- [glim](https://github.com/koide3/glim) / gtsam_points — GPU VGICP (GaussianVoxelMap + coarse-to-fine) と offline rosbag 処理の **定式化** を参考に Metal へ移植 (コードは CUDA のため流用せず)
- plain_slam_ros2 — Hierarchical Geometric Observer / NormalMap の概念
- Super-LIO — OctVoxMap (VoxelMap の元ネタ) の概念
- sycl_points — SYCL 連携の手法

## ライセンス

Apache-2.0
