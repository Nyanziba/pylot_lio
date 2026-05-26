# pylot_lio 総合解説 — アーキテクチャ / Config / アルゴリズム / 数学

本ドキュメントは `pylot_lio` を 1 ファイルで把握するための総合解説です。レイヤー設計、設定パラメータ、各アルゴリズムの中身、必要な数学を順に説明します。個別の深堀りは [ARCHITECTURE.md](ARCHITECTURE.md) と [ALGORITHMS.md](ALGORITHMS.md) を参照してください。

---

## 1. アーキテクチャ

### 1.1 設計コンセプト

「同じ rosbag に対して **LIO の構成要素を差し替えて結果を比較する**」ことが第一目的。そのために LIO 全体を **4 つの抽象インタフェース** に分割し、ROS パラメータで実装を選べるようにしています。

### 1.2 4 つのレイヤー (Strategy パターン)

| レイヤー | インタフェース | 役割 | 入出力 |
|---|---|---|---|
| 1. Preprocessor | `IPreprocessor` | 1 スキャンの間引き (代表点抽出) | 点群 → 点群 (より少ない点) |
| 2. Map | `IPointCloudMap` | world フレーム累積マップ + 最近傍検索 | 点群を `insertScan` で蓄積、`findNearestNeighbor` で問い合わせ |
| 3. Registration | `IRegistration` | source 点群 vs map の姿勢最適化 | 初期姿勢 → 収束姿勢 |
| 4. StateEstimator | `IStateEstimator` | IMU 予測 + Scan 更新で `T_world_body` を出力 | IMU と Scan を吸い込み、姿勢を吐く |

具象クラス (factory に登録されている文字列名):

- Preprocessor: `voxel_grid` / `random_sampling` / `voxel_random_sampling`
- Map: `voxel_map` / `normal_map` / `kd_tree_map` / `voxel_random_map` / `voxel_keyframe_submap`
- Registration: `plain_gicp` / `small_gicp_gicp` / `small_gicp_vgicp` / `sycl`
- Estimator: `ieskf` / `hgo` / `gicp_only`

### 1.3 Factory + CMake による依存吸収

[factory.hpp](../include/pylot_lio/factory.hpp) の `createBackendsFromConfig(LioBackendConfig)` がただ一つのエントリ。`lio_node.cpp` は具象を一切知らない。

CMake 側で:

- `find_package(small_gicp QUIET)` → 成功時のみ `SmallGicpRegistration` をビルド対象に追加 + `PYLOT_LIO_HAS_SMALL_GICP` を define
- `CMAKE_CXX_COMPILER_ID == IntelLLVM` のときだけ `SyclRegistration` 本体、それ以外は `sycl_registration_stub.cpp`

これで「small_gicp 無し」「SYCL 無し」のどちらでもコアはビルドできます。**現状はビルド環境を問わず** `sycl` 指定 → factory が `plain_gicp` にフォールバックします (本来は IntelLLVM + Linux でのみ SYCL 実装が選ばれる設計ですが、現在 SYCL 実装は未配線で、`src/registration/sycl_registration_stub.cpp` がリンクされる)。

### 1.4 データフロー

```
/livox/imu ─► onImu() ─► IStateEstimator::predictWithImu  (200Hz)
                                  │
/livox/lidar ─► onLidarCloud() ─► IPreprocessor::process
                                  ▼
                          IStateEstimator::updateWithScan
                            ├ IESKF/HGO: 反復で IPointCloudMap::findNearestNeighbor → ESKF 更新
                            └ gicp_only: IRegistration::align を 1 回
                                  ▼
                          IPointCloudMap::insertScan
                                  ▼
                  /lio/odom (nav_msgs::Odometry)
                  /lio/cloud_world (cloud_publish_interval_ = 15 frames ごと)
                  TF: world → base_link
```

---

## 2. Config 設定

### 2.1 階層

- **[config/mid360.yaml](../config/mid360.yaml)** — Livox Mid-360 を前提とした「全パラメータの既定値」。I/O トピック、preprocessor、map、registration、estimator、IMU スケール、重力、extrinsic、loop/PGO までフルセット。
- **[config/presets/*.yaml](../config/presets/)** — 各プリセットは **mid360.yaml の差分** のみ書く。launch ファイルが両方を順に読み込む形。

### 2.2 主要パラメータの読み方

| パラメータ | 意味 | 典型値 |
|---|---|---|
| `imu_acceleration_scale` | Livox IMU は g 単位 (acc.z≈1.0)、xsens 等は m/s²。前者なら 9.80665、後者なら 1.0 | 9.80665 / 1.0 |
| `gravity_norm` | 重力の **大きさ** を固定。方向だけ起動時 IMU 平均から決める (z ドリフト主因を消す Super-LIO 流) | 9.7946 |
| `extrinsic_*_imu_from_lidar` | `T_imu_lidar`: LiDAR 点を IMU frame に持ってくる変換 | Mid-360 内蔵 IMU: R=I, t=(-0.011, -0.023, 0.044) |
| `extrinsic_source` | `"config"` ならこの YAML、`"tf"` なら起動時に TF tree から `lookupTransform(imu_frame, lidar_frame)` | config |
| `voxel_grid_size_m` | 前処理ボクセル一辺 [m] | 0.3 (Mid-360) / 0.4 (Velodyne) |
| `map_voxel_size_m` | マップボクセル一辺。前処理より大きく取る (代表点が複数入る粒度に) | 0.5 |
| `map_min_points_per_cell` | この点数を満たさないボクセルは対応点に使わない。`mid360.yaml` 既定は **5** (定常運用で外れ値ボクセルを排除)。`gicp_only_voxel` など IMU 不使用プリセットでは初期数スキャンで対応点ゼロ → マップが原点に積み重なるのを避けるため **1** に下げている | mid360 既定 5 / gicp_only 系 1 |
| `registration_max_correspondence_m` | source→map 対応の最大許容距離。これより遠い対応は外れ値扱い | 2.0〜3.0 |
| `enable_degenerate_regularization` | 縮退方向 Tikhonov 正則化 (廊下/対称構造で発散防止) | preset 依存 |
| `keyframe_min_translation_m` / `keyframe_min_rotation_rad` | keyframe 採択トリガ | 0.5m / 0.1rad 程度 |
| `enable_loop_detection` + Scan Context 系 | リング/セクター数、閾値、cooldown | 60×20, 0.10 |
| `enable_pose_graph_optimization` + PGO 系 | GTSAM ISAM2、odometry/loop の sigma | sigma_rot 0.01〜0.05 |

### 2.3 用意済みプリセット

| プリセット | preprocessor | map | registration | estimator | 用途 |
|---|---|---|---|---|---|
| `ieskf_smallgicp_voxel` | voxel_grid | voxel_map | small_gicp_vgicp | ieskf | **標準推奨** (FAST-LIO2 系) |
| `ieskf_smallgicp_voxelrandom` | voxel_grid | voxel_random_map | small_gicp_vgicp | ieskf | 生点リザーバ保持の効果検証 |
| `hgo_plaingicp_normal` | voxel_grid | normal_map | plain_gicp | hgo | **外部依存ゼロ** (plain_slam_ros2 風) |
| `gicp_only_voxel` | voxel_grid | voxel_random_map | small_gicp_vgicp | gicp_only | IMU 不使用ベースライン |
| `gicp_only_kdtree` | random_sampling | kd_tree_map | plain_gicp | gicp_only | 最素朴な実装 (比較用底辺) |
| `gicp_only_degeneracy_robust` | voxel_grid | voxel_map | small_gicp_vgicp | gicp_only | **Tikhonov 正則化 ON** (廊下対策) |
| `velodyne_xsens` | voxel_random_sampling | voxel_keyframe_submap | small_gicp_gicp | gicp_only | Velodyne+xsens 用パラメータ束 |

---

## 3. 各レイヤーのアルゴリズム

### 3.1 Preprocessor — 代表点抽出

- **`voxel_grid`**: 空間を一辺 `voxel_grid_size_m` の立方体に区切り、同じセルに入った点を **重心** に置き換える。出力点数は環境密度に依存。
- **`random_sampling`**: 単純に `target_count` 個を一様乱択 (seed あり)。点数固定 → 計算量一定。
- **`voxel_random_sampling`**: voxel_grid と同じ粒度でセルに割り当てるが、**重心ではなく実点を 1 つランダム選択**。ring 状 LiDAR (Velodyne) で「重心化が法線推定を壊す」問題への処方。

### 3.2 Map — 累積マップと最近傍

- **`voxel_map`**: 各ボクセルに `(平均 μ, 共分散 Σ, 点数 n)` をオンライン更新で保持。Welford 風漸化式で `O(1)/点`。クエリは近傍ボクセル群の (μ, Σ) を返す。
- **`normal_map`**: voxel_map と同じだが、Σ の固有値を平面側に拡大・法線側に縮小して **point-to-plane の重みを埋め込む**。
- **`voxel_random_map`**: 各ボクセルに **最大 K 個の実点**を Vitter Algorithm R (リザーバサンプリング) で保持。生点が残るので 1 ボクセルに 2 面が偶然入っても両方の情報が消えない。
- **`kd_tree_map`**: 全点を KdTree。教科書通り、遅い (構築コスト)。
- **`voxel_keyframe_submap`**: keyframe ごとに submap を切る。直近 N keyframe を sliding window で registration target に、確定 submap は PGO/loop closure 用に snapshot。

### 3.3 Registration — 姿勢最適化

中核は **GICP (Generalized ICP)**。「点と点の単純距離」ではなく **共分散で重み付けした Mahalanobis 距離** を最小化します。

cost 関数 (source 点 $p_i$, target 点 $q_i$, source/target 共分散 $C_i^s, C_i^t$):

$$
E(T) = \sum_i d_i^T \big(C_i^t + R\, C_i^s R^T\big)^{-1} d_i, \quad d_i = q_i - (R p_i + t)
$$

具象:

- **`plain_gicp`**: 自前実装。外部依存ゼロ。
- **`small_gicp_gicp` / `small_gicp_vgicp`**: Koide 実装ラッパ。VGICP はボクセル単位ガウスでマッチ (point-to-distribution)。OpenMP 並列。
- **`sycl`**: GPU 並列版 (Intel oneAPI、Linux + IntelLLVM 環境を想定)。**現状は未実装** — `src/registration/sycl_registration_stub.cpp` のスタブがリンクされ、factory は `sycl` 指定時 OS を問わず `plain_gicp` にフォールバックします。

#### 縮退正則化 (X-ICP / Tuna 2024)

廊下や対称構造では Hessian の一部固有値がゼロに近づき解が暴れます。処方:

```
1. 6×6 Hessian H を [回転 3×3 | 並進 3×3] のブロック対角に分解、固有値分解
2. inlier 数で正規化した固有値 < threshold の方向 = 縮退方向
3. 縮退方向だけ Tikhonov ペナルティを追加
   (H + H_pen) Δx = -g - H_pen · δ_twist
4. well-conditioned 方向はオリジナル GICP と完全一致 (精度を犠牲にしない)
```

### 3.4 StateEstimator — 推定アルゴリズム

#### IESKF (Iterated Error-State Kalman Filter)

**FAST-LIO2 系**。15 次元状態 [位置 3, 速度 3, 姿勢誤差 3, バイアス加速度 3, バイアスジャイロ 3]。

- **Predict (IMU, 200Hz)**:
  - $p_{k+1} = p_k + v_k \Delta t + \tfrac{1}{2}(R_k(a_m - b_a) + g)\, \Delta t^2$
  - $v_{k+1} = v_k + (R_k(a_m - b_a) + g)\, \Delta t$
  - $R_{k+1} = R_k \cdot \mathrm{Exp}\big((\omega_m - b_g)\, \Delta t\big)$
  - 共分散 $P_{k+1} = F P_k F^T + G Q G^T$

- **Update (Scan, 10Hz)**: registration の結果 $T_{\mathrm{obs}}$ を観測として、誤差状態 $\delta x$ について反復線形化:
  - $\delta x = K (z - h(\hat{x}))$、$K = P H^T (H P H^T + R)^{-1}$
  - 状態に **boxplus** (リー群上の更新)、共分散に $(I - K H)$
  - これを `max_iteration_per_scan` (既定 4) 回繰り返す

**Iterated** = 線形化点をスキャンごとに数回更新 / **Error-State** = 状態そのものでなく誤差を Kalman 変数にする (誤差はゼロ近傍なので線形化が効く)。

#### HGO (Hierarchical Geometric Observer)

Kalman の代わりに **比例制御**:

$$
\hat{T} \leftarrow \hat{T}_{\mathrm{pred}} \boxplus K \cdot (T_{\mathrm{obs}} \boxminus \hat{T}_{\mathrm{pred}})
$$

ゲイン K を `translation_gain` / `rotation_gain` で手動チューニング。共分散を持たない分軽い。

#### GICP-only

IMU 不使用。`predictWithImu` は no-op、registration の結果が即姿勢。ベースライン。

---

## 4. 必要な数学

> 線形代数の基礎 (ベクトル・行列・固有値) から積み上げた詳細版は [MATH.md](MATH.md) を参照。この節は要点のみ。

### 4.1 リー群 SO(3) — なぜ必要か

3D の向きは「軸+角度」で 3 自由度だが、**普通の足し算が成立しない** (回転は非可換: $R_1 R_2 \ne R_2 R_1$)。誤差状態を素直にベクトル空間で扱うと、大きな回転で精度が落ちる/特異点が出ます。

そこで **接空間 (リー代数 $\mathfrak{so}(3) \cong \mathbb{R}^3$) の上で線形化** し、群への戻りに指数写像を使います。

### 4.2 Rodrigues 公式 — `expSO3`

角速度ベクトル $\phi = \theta \hat{u}$ (軸 $\hat{u}$ × 角度 $\theta$) を回転行列に:

$$
\mathrm{Exp}(\phi) = I + \frac{\sin\theta}{\theta}[\phi]_\times + \frac{1 - \cos\theta}{\theta^2}[\phi]_\times^2
$$

- $[\phi]_\times$ は **歪対称行列 (skew)**。$[\phi]_\times v = \phi \times v$ (外積を行列で書いたもの)。
- $\theta \to 0$ で Taylor 展開フォールバック (数値安定化)。

逆写像 `logSO3`: $\theta = \arccos\big(\tfrac{\mathrm{tr}(R) - 1}{2}\big)$、軸は $R - R^T$ から。

### 4.3 boxplus / boxminus — リー群上の加減算

ESKF で状態更新を書くための演算子:

- $T \boxplus \delta = T \cdot \mathrm{Exp}(\delta)$  (誤差を群に乗せる)
- $T_1 \boxminus T_2 = \mathrm{Log}(T_2^{-1} T_1)$  (2 つの姿勢のリー代数差)

### 4.4 右ヤコビアン $J_r$ — `rightJacobianSO3`

「リー代数の差」と「群の差」の橋渡し:

$$
\mathrm{Exp}(\phi + \delta\phi) \approx \mathrm{Exp}(\phi) \cdot \mathrm{Exp}(J_r(\phi)\, \delta\phi)
$$

ESKF の **状態遷移 Jacobian** $F$ や、**観測 Jacobian** $H$ を組むときに必要。逆行列 $J_r^{-1}$ は共分散の boxplus 後の伝播に使います。

### 4.5 Mahalanobis 距離と共分散

GICP の cost に出てくる $(C_t + R C_s R^T)^{-1}$ は **方向別の重み**:

- 平面上の点群 → $C$ の固有値は (大, 大, 小)。平面方向の誤差は無視、法線方向だけ罰する。
- 棒状の点群 → 軸方向のみ大。

`NormalMap` は固有値分解で意図的に (1, 1, ε) に整形して point-to-plane を強制します。

### 4.6 カルマン更新の本体

観測モデル $z = h(x) + \nu$, $\nu \sim N(0, R)$ について:

$$
K = P H^T (H P H^T + R)^{-1}, \quad
\delta x = K (z - h(\hat{x})), \quad
P^+ = (I - K H)\, P
$$

**Iterated** では右辺の $\hat{x}$ を更新後の値で再線形化して繰り返す (≒ Gauss-Newton)。

### 4.7 Tikhonov 正則化と固有値分解

縮退検出のために 6×6 Hessian を回転/並進ブロックに分けて固有値分解 $H = U \Lambda U^T$。$\lambda_i / n_{\mathrm{inlier}} < \mathrm{threshold}$ なる方向 $u_i$ に対して:

$$
H_{\mathrm{pen}} = \alpha \sum_{i \in \text{degenerate}} u_i u_i^T
$$

を $H$ に足す。**well-conditioned 方向の固有値は変えない** 点が普通の Tikhonov (一律に $H + \lambda I$) との違いです。

### 4.8 リザーバサンプリング (Algorithm R, Vitter 1985)

`voxel_random_map` の生点保持で使用。サイズ K の容器に対し:

- 1〜K 番目の要素はそのまま入れる
- $n$ 番目 ($n > K$) は確率 $K/n$ で「採用」、採用時は既存 K 個から一様乱択で 1 個と入れ替え

これで「全 n 個から K 個を一様乱択」と等価分布が、**ストリーム 1 パスかつ $O(K)$ メモリ** で実現できる。

---

## 5. 参考文献

- J. Sola, "Quaternion kinematics for the error-state Kalman filter" (2017) — リー代数と ESKF の決定版
- T. D. Barfoot, "State Estimation for Robotics" (Cambridge, 2017) — リー群上の確率推論の総覧
- Koide et al., "Voxelized GICP for Fast and Accurate 3D Point Cloud Registration" (ICRA 2021) — VGICP の元論文
- Xu & Zhang, "FAST-LIO2: Fast Direct LiDAR-Inertial Odometry" (T-RO 2022) — IESKF + Iterative KF の最新形
- Tuna et al., "X-ICP" (2024, arXiv:2408.11809) — 縮退方向の Tikhonov 正則化
- Vitter, "Random Sampling with a Reservoir" (ACM TOMS 1985) — Algorithm R
