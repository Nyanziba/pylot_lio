# 並列バックエンド (OpenMP / Intel TBB) と性能比較

pylot_lio の registration（点群マッチング）は、点ごとの計算を **OpenMP** または **Intel TBB (oneTBB)** で並列化できる。
どちらを使うかは ROS パラメータ `registration_parallel_backend` で**実行時に**切り替えられるため、
同じ rosbag・同じ前処理・同じ推定器のまま、並列バックエンドだけを差し替えた公平な比較ができる。

- 対象: `plain_gicp`（外部依存ゼロの自前 GICP）と `small_gicp_gicp` / `small_gicp_vgicp`
- 切替パラメータ: `registration_parallel_backend: "omp" | "tbb"`（既定 `"omp"`）
- TBB 非対応ビルド（`PYLOT_LIO_HAS_TBB` 未定義）では `"tbb"` 指定でも自動的に OpenMP へフォールバックし、初回のみ警告を出す。

---

## 1. 使い方

### プリセット
比較用に backend だけが異なる 2 つの preset を用意してある:

| preset | registration | backend |
|---|---|---|
| `config/presets/plain_gicp_openmp.yaml` | plain_gicp | omp |
| `config/presets/plain_gicp_tbb.yaml` | plain_gicp | tbb |

```bash
ros2 launch pylot_lio lio.launch.py preset:=plain_gicp_openmp
ros2 launch pylot_lio lio.launch.py preset:=plain_gicp_tbb
```

### パラメータ直接指定
任意の preset に対して backend だけ上書きもできる:

```bash
ros2 run pylot_lio lio_node --ros-args \
  --params-file <preset>.yaml \
  -p registration_parallel_backend:=tbb
```

---

## 2. 実装の要点

### plain_gicp（`src/registration/plain_gicp_registration.cpp`）
点ごとの Hessian / gradient 累算カーネルを共通ラムダ `linearizeOnePoint` に抽出し、
backend で並列化方式を切り替える:

- **omp**: スレッドごとにローカル accumulator へ書き込み、最後に合算（per-thread accumulate → reduction）。
- **tbb**: `tbb::parallel_reduce` で点範囲を分割し、部分和（H / b / cost / count）を join で畳み込む。

Gauss-Newton 反復間は時系列依存のため並列化しない（並列化するのは 1 反復内の点ループのみ）。

### small_gicp（`src/registration/small_gicp_registration.cpp`）
従来は `RegistrationPCL` を使っていたが、これは内部で `ParallelReductionOMP` を**ハードコード**しており、
`BUILD_WITH_TBB=ON` で再ビルドしても最適化ループは OpenMP のままだった。
そこで低レベル API `small_gicp::Registration<GICPFactor, ParallelReductionOMP / ParallelReductionTBB>` を
直接使う形に書き換え、backend で reduction 型を切り替えられるようにした。
backend ごとに異なる「KdTree ビルダ（`KdTreeBuilderOMP/TBB`）」「共分散推定（`estimate_covariances_omp/tbb`）」
「reduction のスレッド数指定の有無」は `BackendTraits<Reduction>` のテンプレート特殊化で吸収している。

> small_gicp 自体も `BUILD_WITH_TBB=ON` で再ビルドしておくこと（ビルド手順は本パッケージの
> ビルドドキュメント / リポジトリルートの `activate_pixi_env.sh` を参照）。

---

## 3. 性能比較

### 3.1 計測環境

| 項目 | 値 |
|---|---|
| マシン | Apple Mac16,1 (`Mac16,1`) |
| CPU | Apple M4 — 物理 10 コア / 論理 10 |
| メモリ | 24 GiB |
| OS | macOS 26.5 (arm64) |
| コンパイラ | clang 19.1.7 (conda-forge / pixi global env) |
| 並列ランタイム | libomp (LLVM OpenMP) / oneTBB 12.17 (libtbb.12.17) |
| ROS | ROS 2 Jazzy (robostack / conda-forge, `pixi global`) |

### 3.2 計測方法

- rosbag: `2023-03-01-12-07-3`（Livox Mid-360, `/livox/lidar` = `livox_ros_driver2/msg/CustomMsg` 10Hz, `/livox/imu` 200Hz, 全長 421s）
- 各構成で bag を 60 秒分再生し、`/lio/diag`（JSON）の `processing_time_ms` を収集。
- 推定器は `gicp_only`（処理時間がほぼ registration に支配される構成）、`registration_num_threads = 4`。
- 統計は warmup 数スキャンを除外して算出。

`processing_time_ms` は前処理（voxel_grid）+ マップ挿入 + registration の**合計**であり、registration 単体ではない点に注意
（直列部分は両 backend で共通なので、差分の符号は backend 差を反映する）。

### 3.3 結果

| registration | backend | N | mean (ms) | median (ms) | p95 (ms) | min | max |
|---|---|---|---|---|---|---|---|
| plain_gicp | **omp** | 1795 | **62.5** | **62.2** | 91.2 | 20.0 | 178 |
| plain_gicp | tbb | 2395 | 77.6 | 72.1 | 144.5 | 13.1 | 301 |
| small_gicp_vgicp | **omp** | 2328 | **111.2** | **96.4** | 213 | 8.5 | 473 |
| small_gicp_vgicp | tbb | 2188 | 141.8 | 120.8 | 365 | 0.1 | 817 |

### 3.4 結論

このワークロード（1 スキャン数万点・**4 スレッド**・Apple M4 / macOS arm64）では **OpenMP の方が速い**。

- plain_gicp: median で TBB が約 **16% 遅い**（62 → 72 ms）
- small_gicp_vgicp: median で TBB が約 **25% 遅い**（96 → 121 ms）。p95 / max の悪化が顕著（外れ値が多い）。

**解釈**: 「1 反復ごとに `parallel_reduce` を 1 回」という使い方では、TBB のタスクスケジューリング /
ワークスティーリングのオーバヘッドが、OpenMP の静的 for 分割の軽さを上回ったと考えられる。
TBB が真価を発揮するのは flow graph による多段パイプライン・多スレッド（〜128）構成であり、今回の使い方では恩恵が出ない。
TBB 側で p95 / max が大きく荒れているのもスケジューラ起因のばらつきと整合する。

### 3.5 注意・今後

- **N（処理スキャン数）が構成間で異なる**。マップが時間とともに育ち後半スキャンほど重くなるため、処理数が違うと mean が偏る。数値は方向性として扱うこと。
- より厳密にやるなら、registration の `align()` だけを固定入力で N 回叩くマイクロベンチ（gtest）に切り出すとよい。
- スレッド数を変えた掃引（1 / 2 / 4 / 8 / 10）も有用。コア数（10）に対して 4 スレッドでの比較である点に留意。
