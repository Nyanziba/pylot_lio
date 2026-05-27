# pylot_lio

Livox Mid-360 向けの **比較可能な LiDAR-Inertial Odometry**。
4 つのレイヤー (前処理 / マップ / マッチング / 状態推定) すべてを ROS パラメータで差し替えられ、同じ rosbag 上で複数手法を公平に比較できます。

- インタフェース: `/livox/lidar` (sensor_msgs/PointCloud2) + `/livox/imu` (sensor_msgs/Imu) を購読し、`/lio/odom` (nav_msgs/Odometry) と `/lio/cloud_world` (sensor_msgs/PointCloud2) を publish
- 出力 frame: `world -> base_link` TF
- ROS 2 Jazzy 想定、macOS arm64 (pixi/robostack) と Linux で動作

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

| Preset | 状態推定 | マッチング | マップ | 前処理 | 用途 |
|--------|----------|-----------|--------|--------|------|
| `ieskf_smallgicp_voxel` | IESKF | small_gicp VGICP | VoxelMap | voxel_grid | 標準推奨 |
| `hgo_plaingicp_normal` | HGO | plain GICP | NormalMap | voxel_grid | 外部依存なしで動く |
| `gicp_only_kdtree` | GICP-only | plain GICP | KdTreeMap | random_sampling | 最小ベースライン |

すべて `config/mid360.yaml` に既定値があり、preset YAML が個別パラメータを上書きします。

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

### 3rdparty submodule の初期化

最初の clone 直後は以下を実行してください:

```bash
git submodule update --init --recursive
```

| パス | 用途 | pin |
|---|---|---|
| `3rdparty/small_gicp` | VGICP/GICP の実装 (koide3/small_gicp) | tag `v1.0.0` |
| `3rdparty/sycl_points` | SYCL ベースの点群ユーティリティ (fateshelled/sycl_points, Apache-2.0) | `main` HEAD |

システムに小型 GICP がインストールされている環境では submodule のビルドは走らない (find_package が先に見つけるため)。submodule は **オフラインビルド / システムインストール不能環境** のためのフォールバックです。

macOS arm64 + pixi (robostack) 環境では `activate_pixi_env.sh` の手順を踏まないと Apple Clang が選ばれてビルドが落ちます。詳しくは workspace ルートの [`CLAUDE.md`](../../CLAUDE.md) を参照してください。

## テスト

```bash
cbt pylot_lio
```

純ロジックの単体テスト 9 本 (Lie 代数、ボクセルグリッド、ランダムサンプリング、VoxelMap、KdTreeMap、IESKF、GICP-only、Factory、+ small_gicp あれば SmallGicpRegistration)。ROS は起動しないので CI で安全に流せます。

## 参考にしたもの

実装は標準教科書 (Sola 2017, Barfoot 2017, Koide VGICP 2021, FAST-LIO2 2022) の数式から自前で書き起こしています。動作概念のみ参考にしたプロジェクト:

- [small_gicp](https://github.com/koide3/small_gicp) (MIT) — VGICP/GICP の参照実装。`SmallGicpRegistration` で直接リンク
- plain_slam_ros2 — Hierarchical Geometric Observer / NormalMap の概念
- Super-LIO — OctVoxMap (VoxelMap の元ネタ) の概念
- sycl_points — SYCL 連携の手法

## ライセンス

Apache-2.0
