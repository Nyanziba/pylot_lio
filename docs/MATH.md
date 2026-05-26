# pylot_lio で使う数学 — 線形代数の基礎から積み上げる詳細版

このドキュメントは「線形代数を一度もちゃんと学んでいない人」が pylot_lio のコードを読めるところまで連れていくことを目的にしています。各節は前の節だけを前提に書いています。途中で詰まったらその節をもう一度読んでください。

目次:

0. [なぜ数学が必要か](#0-なぜ数学が必要か)
1. [ベクトル — 矢印と数の列](#1-ベクトル--矢印と数の列)
2. [内積 — 「どれくらい同じ向きか」](#2-内積--どれくらい同じ向きか)
3. [外積 — 「どれくらい垂直か」と回転軸](#3-外積--どれくらい垂直か)
4. [行列 — ベクトルを変換する箱](#4-行列--ベクトルを変換する箱)
5. [行列の積 — 変換の合成](#5-行列の積--変換の合成)
6. [逆行列と転置](#6-逆行列と転置)
7. [固有値・固有ベクトル — 行列の「素性」](#7-固有値固有ベクトル--行列の素性)
8. [対称行列と正定値性](#8-対称行列と正定値性)
9. [共分散 — ばらつきの形](#9-共分散--ばらつきの形)
10. [回転行列 SO(3)](#10-回転行列-so3)
11. [リー代数 so(3) と歪対称行列](#11-リー代数-so3-と歪対称行列)
12. [指数写像 Exp と Rodrigues 公式](#12-指数写像-exp-と-rodrigues-公式)
13. [対数写像 Log と右ヤコビアン](#13-対数写像-log-と右ヤコビアン)
14. [boxplus / boxminus](#14-boxplus--boxminus)
15. [剛体変換 SE(3)](#15-剛体変換-se3)
16. [最小二乗法と Gauss-Newton](#16-最小二乗法と-gauss-newton)
17. [Mahalanobis 距離と GICP のコスト](#17-mahalanobis-距離と-gicp-のコスト)
18. [カルマンフィルタの式](#18-カルマンフィルタの式)
19. [Error-State Kalman Filter (ESKF)](#19-error-state-kalman-filter-eskf)
20. [Iterated ESKF](#20-iterated-eskf)
21. [Tikhonov 正則化と縮退対応](#21-tikhonov-正則化と縮退対応)
22. [リザーバサンプリング Algorithm R](#22-リザーバサンプリング-algorithm-r)

---

## 0. なぜ数学が必要か

「ロボットが部屋のどこにいて、どっちを向いているか」を、毎秒何十回も計算したい。これに必要なのは:

- **空間の中の点や向きを数字で書き表す方法** → ベクトル・行列
- **「ここからあっちへ動いた」を数字の操作で書く方法** → 行列の積、回転行列
- **「センサーの値はノイズで揺れる」を扱う方法** → 共分散、確率分布
- **「ズレを最小化する答え」を求める方法** → 最小二乗法、Gauss-Newton
- **「速いセンサー (IMU) と遅いセンサー (LiDAR) を混ぜる方法」** → カルマンフィルタ

これら全部の道具が pylot_lio の中で使われています。順に積み上げます。

---

## 1. ベクトル — 矢印と数の列

### 1.1 直感

3D 空間の「位置」や「向き」は、3 つの数字の組で書けます:

$$
\mathbf{v} = \begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix}
$$

これは「x 方向に 1、y 方向に 2、z 方向に 3 だけ進んだ場所」を表す矢印です。ベクトルとは要するに **「数字を縦に並べたもの」** と思っていればよい。

コード中では `Eigen::Vector3d`。

### 1.2 足し算とスカラー倍

- 足し算: $\begin{pmatrix}1\\2\\3\end{pmatrix} + \begin{pmatrix}4\\5\\6\end{pmatrix} = \begin{pmatrix}5\\7\\9\end{pmatrix}$ — 矢印をつなげる
- スカラー倍: $2 \cdot \begin{pmatrix}1\\2\\3\end{pmatrix} = \begin{pmatrix}2\\4\\6\end{pmatrix}$ — 矢印を引き伸ばす

### 1.3 長さ (ノルム)

ピタゴラスの定理の 3D 版:

$$
\|\mathbf{v}\| = \sqrt{v_x^2 + v_y^2 + v_z^2}
$$

例: $(1, 2, 2)$ の長さは $\sqrt{1 + 4 + 4} = 3$。

長さ 1 のベクトルを **単位ベクトル** と言います。$\hat{\mathbf{v}} = \mathbf{v} / \|\mathbf{v}\|$。

---

## 2. 内積 — 「どれくらい同じ向きか」

### 2.1 定義

2 つのベクトル $\mathbf{a} = (a_1, a_2, a_3)$, $\mathbf{b} = (b_1, b_2, b_3)$ の **内積** は数字 1 個になります:

$$
\mathbf{a} \cdot \mathbf{b} = a_1 b_1 + a_2 b_2 + a_3 b_3
$$

### 2.2 幾何的意味

$$
\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\| \cdot \|\mathbf{b}\| \cdot \cos\theta
$$

$\theta$ は 2 ベクトルの間の角度。だから:

- 同じ向き ($\theta = 0$): 内積 = 長さの積 (最大)
- 直角 ($\theta = 90°$): 内積 = 0
- 反対向き ($\theta = 180°$): 内積 = 負の最大

**内積 = 「同じ向きさ」を測る道具** と思えばいい。

### 2.3 使われる場所

- ベクトルを「ある方向」に射影する: $\mathbf{a}$ を $\hat{\mathbf{n}}$ 方向に射影した長さ = $\mathbf{a} \cdot \hat{\mathbf{n}}$
- 「点と平面の距離」を計算するのは結局これ (法線方向への射影)

---

## 3. 外積 — 「どれくらい垂直か」と回転軸

### 3.1 定義

3D ベクトル同士の **外積** は **ベクトル** を返します:

$$
\mathbf{a} \times \mathbf{b} = \begin{pmatrix}
a_2 b_3 - a_3 b_2 \\
a_3 b_1 - a_1 b_3 \\
a_1 b_2 - a_2 b_1
\end{pmatrix}
$$

### 3.2 幾何的意味

- 大きさ: $\|\mathbf{a} \times \mathbf{b}\| = \|\mathbf{a}\| \|\mathbf{b}\| \sin\theta$ ($\mathbf{a}$ と $\mathbf{b}$ が作る平行四辺形の面積)
- 向き: $\mathbf{a}$ と $\mathbf{b}$ の両方に垂直 (右手の法則)

### 3.3 回転との関係

角速度ベクトル $\boldsymbol{\omega}$ (向きが回転軸、大きさが角速度) が点 $\mathbf{p}$ を動かす速さは:

$$
\dot{\mathbf{p}} = \boldsymbol{\omega} \times \mathbf{p}
$$

**これが「回転」を扱うときに外積が顔を出す根本的な理由** です。あとで歪対称行列に化けます。

---

## 4. 行列 — ベクトルを変換する箱

### 4.1 定義

行列は数字の長方形の表:

$$
A = \begin{pmatrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
a_{31} & a_{32} & a_{33}
\end{pmatrix}
$$

3×3 行列はコードでは `Eigen::Matrix3d`。

### 4.2 行列とベクトルの掛け算

$$
A \mathbf{v} = \begin{pmatrix}
a_{11} v_1 + a_{12} v_2 + a_{13} v_3 \\
a_{21} v_1 + a_{22} v_2 + a_{23} v_3 \\
a_{31} v_1 + a_{32} v_2 + a_{33} v_3
\end{pmatrix}
$$

各行とベクトルの **内積** を縦に並べたもの。

### 4.3 意味

行列はベクトルを別のベクトルに **変換する道具** です。たとえば:

- 単位行列 $I = \mathrm{diag}(1,1,1)$ は「何もしない」
- $\mathrm{diag}(2,1,1)$ は「x 方向に 2 倍引き伸ばす」
- 後で出てくる「回転行列」は「向きを変える」

---

## 5. 行列の積 — 変換の合成

### 5.1 定義

$C = A B$ のとき、$C$ の $(i, j)$ 成分は「$A$ の $i$ 行目」と「$B$ の $j$ 列目」の内積:

$$
c_{ij} = \sum_k a_{ik} b_{kj}
$$

### 5.2 「順序」が大事

$A B \ne B A$ が普通 (**非可換**)。

直感: $A$ が「90 度横に回す」、$B$ が「30 cm 前に進める」だとして、「回ってから前進」と「前進してから回る」では着地点が違う。これは後の **回転の非可換性** に直結します。

### 5.3 「結合則」は成立

$(AB)C = A(BC)$ は成り立つ。なので「3 つの変換を順にかける」は途中で勝手にまとめてよい。

---

## 6. 逆行列と転置

### 6.1 転置

$A^T$ は行と列を入れ替えたもの。$(A^T)_{ij} = A_{ji}$。

### 6.2 逆行列

「かけ算で 1 (= 単位行列) に戻すもの」。$A^{-1} A = A A^{-1} = I$。

すべての行列に逆行列があるわけではない (**ランク落ち** の行列にはない)。pylot_lio で逆行列を取る場面は:

- 共分散の逆 (Mahalanobis 距離)
- Kalman ゲイン $K = P H^T (H P H^T + R)^{-1}$
- Hessian の逆 (Gauss-Newton)

これらが「逆行列が取れない=不能」になる状況が **縮退** です (§21)。

### 6.3 直交行列

$Q^T Q = I$ を満たす行列 $Q$ を **直交行列** と呼ぶ。つまり $Q^{-1} = Q^T$。

回転行列はすべて直交行列です。これは後で「回転行列の逆 = 転置」という非常に便利な事実につながります。

---

## 7. 固有値・固有ベクトル — 行列の「素性」

### 7.1 定義

行列 $A$ に対して、

$$
A \mathbf{v} = \lambda \mathbf{v}
$$

を満たすゼロでないベクトル $\mathbf{v}$ を **固有ベクトル**、$\lambda$ を **固有値** と言う。

意味: 「$\mathbf{v}$ という方向では、$A$ はただ $\lambda$ 倍に引き伸ばすだけ」。

### 7.2 直感的に何が嬉しいか

行列という複雑な変換も、「**いくつかの方向 (固有ベクトル) × 各方向の倍率 (固有値)**」の組み合わせで書き直せる。

3×3 対称行列なら必ず 3 つの互いに直交する固有ベクトル $\mathbf{u}_1, \mathbf{u}_2, \mathbf{u}_3$ と固有値 $\lambda_1, \lambda_2, \lambda_3$ があり:

$$
A = \lambda_1 \mathbf{u}_1 \mathbf{u}_1^T + \lambda_2 \mathbf{u}_2 \mathbf{u}_2^T + \lambda_3 \mathbf{u}_3 \mathbf{u}_3^T
$$

これを **固有値分解 (スペクトル分解)** と呼びます。

### 7.3 pylot_lio での使い道

- 共分散 $\Sigma$ の固有値分解 → 「ばらつきの主軸と長さ」
- Hessian の固有値分解 → 「拘束が強い方向と弱い方向」 (§21 縮退検出)
- 法線推定 → 共分散の最小固有値方向 = 平面の法線

---

## 8. 対称行列と正定値性

- **対称行列**: $A = A^T$
- **正定値行列**: 任意のゼロでない $\mathbf{v}$ に対し $\mathbf{v}^T A \mathbf{v} > 0$。同値条件: 全ての固有値が正

確率分布の共分散行列、Kalman フィルタの $P$、最小二乗法の Hessian など、推定の世界に出てくる行列はほぼすべて **対称半正定値** です (固有値 ≥ 0)。

「逆行列が取れる」⇔「固有値が全部ゼロでない」⇔「正定値 (推定の文脈では)」。

---

## 9. 共分散 — ばらつきの形

### 9.1 1 次元のおさらい

ばらつきの大きさ = **分散** $\sigma^2 = \mathbb{E}[(x - \mu)^2]$。標準偏差は $\sigma$。

### 9.2 多次元

3D の点群のばらつきは、3 方向にそれぞれの大きさと、方向どうしの「連動」があります。これを表すのが **共分散行列**:

$$
\Sigma = \mathbb{E}\big[(\mathbf{x} - \boldsymbol{\mu})(\mathbf{x} - \boldsymbol{\mu})^T\big] \in \mathbb{R}^{3 \times 3}
$$

成分:

$$
\Sigma_{ij} = \mathbb{E}[(x_i - \mu_i)(x_j - \mu_j)]
$$

対角成分が各軸の分散、非対角成分が「2 軸の連動具合」。

### 9.3 楕円体としての解釈

$\Sigma$ を固有値分解すると、ばらつきの主軸 (固有ベクトル) と各主軸の長さ ($\sqrt{\lambda_i}$) が出てきます:

```
点群が長細く分布:  λ1 ≫ λ2 ≈ λ3 → 棒状
平面状の分布:      λ1 ≈ λ2 ≫ λ3 → 平面 (最小固有値の固有ベクトルが法線)
ボール状:           λ1 ≈ λ2 ≈ λ3 → 等方
```

### 9.4 サンプルからの計算 (Welford)

$N$ 個の点 $\mathbf{x}_1, \dots, \mathbf{x}_N$ から推定:

$$
\boldsymbol{\mu} = \frac{1}{N}\sum \mathbf{x}_i, \qquad
\Sigma = \frac{1}{N}\sum (\mathbf{x}_i - \boldsymbol{\mu})(\mathbf{x}_i - \boldsymbol{\mu})^T
$$

VoxelMap はこれを **Welford のオンライン更新** で 1 点ずつ漸化的に計算します:

$$
\boldsymbol{\mu}_n = \boldsymbol{\mu}_{n-1} + \frac{\mathbf{x}_n - \boldsymbol{\mu}_{n-1}}{n}
$$

$$
M_{2,n} = M_{2,n-1} + (\mathbf{x}_n - \boldsymbol{\mu}_{n-1})(\mathbf{x}_n - \boldsymbol{\mu}_n)^T
$$

$\Sigma_n = M_{2,n} / n$。これで全点をメモリに保持せずに済みます。

---

## 10. 回転行列 SO(3)

### 10.1 何を表すか

3D の「向きを変える」操作を、3×3 の行列 $R$ で書き表したもの:

$$
\mathbf{p}' = R \mathbf{p}
$$

### 10.2 性質

回転行列は次の 2 条件を満たします:

- **直交性**: $R^T R = I$ (長さと角度を保存する)
- **行列式 +1**: $\det(R) = +1$ (鏡像反転を含まない)

このような行列全体を **SO(3) (Special Orthogonal group)** と呼びます。「3 次元の特殊直交群」。

### 10.3 嬉しい事実

- 逆 = 転置: $R^{-1} = R^T$ (計算が超軽い)
- 任意の点を回転しても **長さは変わらない**: $\|R\mathbf{p}\| = \|\mathbf{p}\|$
- 任意の 2 点間の **角度も変わらない**

### 10.4 困った事実

- 9 個の数字を持つが **自由度は 3** だけ (6 個の制約 $R^T R = I$ で縛られる)
- 普通の足し算は使えない: $R_1 + R_2$ は普通の行列だが回転行列ではない
- かけ算は非可換: $R_1 R_2 \ne R_2 R_1$
- 「ちょっとずれた回転」を 9 個の数字で表現するのは厄介 (制約が壊れる)

→ これを解消するのが次の **リー代数** です。

---

## 11. リー代数 so(3) と歪対称行列

### 11.1 歪対称行列

3 次元の歪対称行列とは、$A^T = -A$ を満たす 3×3 行列。具体的には:

$$
A = \begin{pmatrix} 0 & -a_3 & a_2 \\ a_3 & 0 & -a_1 \\ -a_2 & a_1 & 0 \end{pmatrix}
$$

3 個の数字 $(a_1, a_2, a_3)$ で決まる (対角は必ずゼロ)。

### 11.2 ベクトルからの作り方 — skew 作用素

ベクトル $\boldsymbol{\omega} = (\omega_1, \omega_2, \omega_3)$ に対して、次の写像を定義:

$$
[\boldsymbol{\omega}]_\times = \begin{pmatrix} 0 & -\omega_3 & \omega_2 \\ \omega_3 & 0 & -\omega_1 \\ -\omega_2 & \omega_1 & 0 \end{pmatrix}
$$

これは [lie_algebra.hpp](../include/pylot_lio/lie_algebra.hpp) の `skew(v)` です。

### 11.3 「skew は外積を行列で書いただけ」という事実

任意のベクトル $\mathbf{v}$ に対して:

$$
[\boldsymbol{\omega}]_\times \mathbf{v} = \boldsymbol{\omega} \times \mathbf{v}
$$

これは展開すれば確かめられます。**外積 = ベクトル → 行列に化けさせて、普通の行列ベクトル積にしたもの**。

§3.3 で「角速度ベクトルが点を動かす速度は $\boldsymbol{\omega} \times \mathbf{p}$」と言いましたが、これは要するに:

$$
\dot{\mathbf{p}} = [\boldsymbol{\omega}]_\times \mathbf{p}
$$

「点の動きを行列で書ける」=「微小回転は行列の積で扱える」となります。

### 11.4 so(3) という名前

3×3 の歪対称行列全体の集合を **so(3) (small so 3)** と呼びます。これが **SO(3) のリー代数** です。3 次元ベクトル空間と同型 (3 個の独立な数字で書ける)。

「リー代数」と聞くと身構えますが、要は **「SO(3) という曲がった空間を、原点まわりだけ普通のベクトル空間として扱う近似」** です。地球儀の表面が球面でも、自分の足元の数十メートルは平らな地図として扱えるのと同じ。

---

## 12. 指数写像 Exp と Rodrigues 公式

### 12.1 何をしたいか

ベクトル空間 $\mathbb{R}^3$ (so(3) と同一視) と、回転行列の集合 SO(3) の間に橋渡しが欲しい。

「軸 $\hat{\mathbf{u}}$ のまわりに角度 $\theta$ 回転する」を、回転ベクトル $\boldsymbol{\phi} = \theta \hat{\mathbf{u}} \in \mathbb{R}^3$ で表しておいて、これを回転行列 $R$ に変換する写像が **指数写像 (Exp)** です。

### 12.2 行列指数関数

普通の指数関数 $e^x = 1 + x + x^2/2! + x^3/3! + \cdots$ を行列に拡張:

$$
e^A = I + A + \frac{A^2}{2!} + \frac{A^3}{3!} + \cdots
$$

### 12.3 Rodrigues 公式

$A = [\boldsymbol{\phi}]_\times$ (歪対称) のとき、$\theta = \|\boldsymbol{\phi}\|$, $\hat{\mathbf{u}} = \boldsymbol{\phi}/\theta$ とすると、級数が **閉じた形** にまとまります:

$$
\mathrm{Exp}(\boldsymbol{\phi}) = I + \frac{\sin\theta}{\theta}[\boldsymbol{\phi}]_\times + \frac{1 - \cos\theta}{\theta^2}[\boldsymbol{\phi}]_\times^2
$$

これが **Rodrigues 公式**。コードでは [lie_algebra.hpp の expSO3](../include/pylot_lio/lie_algebra.hpp)。

### 12.4 なぜこうなるか (略証)

歪対称行列の累乗には次の恒等式があります (確かめは直接計算):

$$
[\hat{\mathbf{u}}]_\times^2 = \hat{\mathbf{u}} \hat{\mathbf{u}}^T - I, \quad
[\hat{\mathbf{u}}]_\times^3 = -[\hat{\mathbf{u}}]_\times
$$

ということは $A^2, A^4, \dots$ は $A^2$ の倍数、$A^3, A^5, \dots$ は $A$ の倍数。級数を奇数項と偶数項に分けて整理すると、$\sin\theta$ と $\cos\theta$ のテイラー展開がきれいに現れて Rodrigues 式になります。

### 12.5 数値的注意 ($\theta \to 0$)

$\sin\theta/\theta$ と $(1-\cos\theta)/\theta^2$ は $\theta=0$ で形式的に 0/0。実装では小角度のとき Taylor 展開 ($\sin\theta/\theta \approx 1 - \theta^2/6$ など) にフォールバックします。

---

## 13. 対数写像 Log と右ヤコビアン

### 13.1 Log = Exp の逆

回転行列 $R$ から「軸×角度」の回転ベクトル $\boldsymbol{\phi}$ を取り出す:

$$
\theta = \arccos\!\left(\frac{\mathrm{tr}(R) - 1}{2}\right), \quad
[\boldsymbol{\phi}]_\times = \frac{\theta}{2\sin\theta}(R - R^T)
$$

$\theta = 0$ 付近、$\theta = \pi$ 付近で特殊処理が必要 (コードでは数値安定化済み)。

### 13.2 右ヤコビアン $J_r$ — なぜ必要か

「リー代数の足し算」と「群の積」は厳密には一致しません。たとえば:

$$
\mathrm{Exp}(\boldsymbol{\phi} + \delta\boldsymbol{\phi}) \ne \mathrm{Exp}(\boldsymbol{\phi}) \cdot \mathrm{Exp}(\delta\boldsymbol{\phi})
$$

ですが、$\delta\boldsymbol{\phi}$ が小さければ、ある **補正係数行列 $J_r$** を使って次のように書けます:

$$
\mathrm{Exp}(\boldsymbol{\phi} + \delta\boldsymbol{\phi}) \approx \mathrm{Exp}(\boldsymbol{\phi}) \cdot \mathrm{Exp}\big(J_r(\boldsymbol{\phi})\, \delta\boldsymbol{\phi}\big)
$$

この $J_r$ が **右ヤコビアン**。

### 13.3 公式

$$
J_r(\boldsymbol{\phi}) = I - \frac{1 - \cos\theta}{\theta^2}[\boldsymbol{\phi}]_\times + \frac{\theta - \sin\theta}{\theta^3}[\boldsymbol{\phi}]_\times^2
$$

逆 $J_r^{-1}$ も存在し:

$$
J_r^{-1}(\boldsymbol{\phi}) = I + \frac{1}{2}[\boldsymbol{\phi}]_\times + \left(\frac{1}{\theta^2} - \frac{1 + \cos\theta}{2\theta\sin\theta}\right)[\boldsymbol{\phi}]_\times^2
$$

### 13.4 どこで使うか

- ESKF の状態遷移ヤコビアン $F$ の中 (回転誤差の伝播)
- 観測ヤコビアン $H$ の中
- 共分散の boxplus 後の伝播: $P \leftarrow J_r P J_r^T$

「回転についての偏微分」を取りたいときは、必ずこの $J_r$ が顔を出すと思っていてよい。

---

## 14. boxplus / boxminus

### 14.1 動機

ESKF では「状態に小さな誤差を足したい」「2 つの姿勢の差を取りたい」を、回転を含んだまま安全にやりたい。普通の足し算 $R + \delta R$ は SO(3) を逸脱するのでダメ。

### 14.2 定義

回転に対して:

- $R \boxplus \delta\boldsymbol{\phi} := R \cdot \mathrm{Exp}(\delta\boldsymbol{\phi})$
- $R_1 \boxminus R_2 := \mathrm{Log}(R_2^T R_1)$

ここで $\delta\boldsymbol{\phi} \in \mathbb{R}^3$ (リー代数のベクトル表現)。

### 14.3 性質

- $(R \boxplus \delta) \boxminus R = \delta$ (足してから引けば戻る)
- $R \boxplus 0 = R$
- 普通の足し算と違って、**SO(3) を逸脱しない**

### 14.4 SE(3) への拡張

剛体変換 (回転+並進) $T = (R, \mathbf{t})$ に対しても同様に:

$$
T \boxplus \boldsymbol{\xi} := T \cdot \mathrm{Exp}_{SE(3)}(\boldsymbol{\xi}), \quad \boldsymbol{\xi} \in \mathbb{R}^6
$$

ESKF の状態更新は基本的に $\hat{x} \leftarrow \hat{x} \boxplus K \cdot r$ という形になります。

---

## 15. 剛体変換 SE(3)

### 15.1 定義

回転 + 並進を一緒に表すのが **SE(3) (Special Euclidean group)**。3 次元剛体運動の全体。

要素 $T \in SE(3)$ は (回転 $R$, 並進 $\mathbf{t}$) のペア。4×4 同次行列で書く流儀もある:

$$
T = \begin{pmatrix} R & \mathbf{t} \\ \mathbf{0}^T & 1 \end{pmatrix}
$$

### 15.2 点に作用させる

3D 点 $\mathbf{p}$ に対して:

$$
T \mathbf{p} = R \mathbf{p} + \mathbf{t}
$$

### 15.3 合成

2 つの剛体変換の合成 $T_1 \cdot T_2$ は:

$$
T_1 T_2 = \begin{pmatrix} R_1 R_2 & R_1 \mathbf{t}_2 + \mathbf{t}_1 \\ \mathbf{0}^T & 1 \end{pmatrix}
$$

### 15.4 逆

$$
T^{-1} = \begin{pmatrix} R^T & -R^T \mathbf{t} \\ \mathbf{0}^T & 1 \end{pmatrix}
$$

### 15.5 自由度 6 のリー代数 se(3)

接空間は 6 次元 (回転 3 + 並進 3)。Eigen では `Eigen::Isometry3d`、リー代数表現は普通 $\boldsymbol{\xi} = (\boldsymbol{\rho}, \boldsymbol{\phi}) \in \mathbb{R}^6$ で並進部 $\boldsymbol{\rho}$ と回転部 $\boldsymbol{\phi}$ を並べる流儀 (pylot_lio もこの順)。

---

## 16. 最小二乗法と Gauss-Newton

### 16.1 線形最小二乗

「観測残差ベクトル $\mathbf{r}$ の長さの 2 乗を最小化したい」。残差が変数 $\mathbf{x}$ について線形なら ($\mathbf{r} = A\mathbf{x} - \mathbf{b}$):

$$
\min_\mathbf{x} \|A\mathbf{x} - \mathbf{b}\|^2
$$

の解は **正規方程式**:

$$
(A^T A) \mathbf{x} = A^T \mathbf{b}
$$

$A^T A$ が **Hessian** (左辺の行列)、$A^T \mathbf{b}$ が **gradient の符号反転** (右辺)。

### 16.2 非線形 → Gauss-Newton

残差が非線形 $\mathbf{r}(\mathbf{x})$ のときは、現在の推定 $\hat{\mathbf{x}}$ のまわりで線形化:

$$
\mathbf{r}(\hat{\mathbf{x}} + \Delta\mathbf{x}) \approx \mathbf{r}(\hat{\mathbf{x}}) + J \Delta\mathbf{x}, \quad J = \frac{\partial \mathbf{r}}{\partial \mathbf{x}}\bigg|_{\hat{\mathbf{x}}}
$$

これを線形最小二乗で解く:

$$
(J^T J)\, \Delta\mathbf{x} = -J^T \mathbf{r}(\hat{\mathbf{x}})
$$

得られた $\Delta\mathbf{x}$ で $\hat{\mathbf{x}} \leftarrow \hat{\mathbf{x}} \boxplus \Delta\mathbf{x}$ と更新し、収束まで反復。これが **Gauss-Newton 法**。

### 16.3 GICP に現れる Hessian

GICP のコスト (§17) の Hessian は 6×6 (姿勢の自由度)。これが縮退検出 (§21) で固有値分解される対象。

### 16.4 重み付き最小二乗

各残差に「重み」(信頼度の逆) を付けるバージョン:

$$
\min \sum_i \mathbf{r}_i^T W_i \mathbf{r}_i \Rightarrow (J^T W J)\Delta\mathbf{x} = -J^T W \mathbf{r}
$$

$W_i$ が共分散の逆 ($W_i = \Sigma_i^{-1}$) のとき、これは **Mahalanobis 距離の最小化** になります。

---

## 17. Mahalanobis 距離と GICP のコスト

### 17.1 Mahalanobis 距離

ユークリッド距離 $\|\mathbf{d}\|^2 = \mathbf{d}^T \mathbf{d}$ を、共分散で重み付けたもの:

$$
\|\mathbf{d}\|_\Sigma^2 := \mathbf{d}^T \Sigma^{-1} \mathbf{d}
$$

幾何的意味: $\Sigma$ で決まる楕円体を「単位球」に正規化したあとのユークリッド距離。

### 17.2 なぜ要るか

「平面上の点群と新しい点 1 つ」を考えると、平面方向への 1 m のズレと法線方向への 1 m のズレでは **意味が全く違う**。ユークリッドではどちらも同じ「1 m」だが、Mahalanobis では:

- 平面方向: $\Sigma$ の固有値大 → $\Sigma^{-1}$ の固有値小 → 距離が小さく出る (= 罰しない)
- 法線方向: $\Sigma$ の固有値小 → $\Sigma^{-1}$ の固有値大 → 距離が大きく出る (= 厳しく罰する)

### 17.3 GICP のコスト関数

source 点 $\mathbf{p}_i$, target 点 $\mathbf{q}_i$, それぞれの局所共分散 $C_i^s, C_i^t$ について:

$$
E(T) = \sum_i \mathbf{d}_i^T \big(C_i^t + R\, C_i^s R^T\big)^{-1} \mathbf{d}_i, \quad \mathbf{d}_i = \mathbf{q}_i - (R\mathbf{p}_i + \mathbf{t})
$$

ポイント:
- 残差 $\mathbf{d}_i$ は剛体変換 $T = (R, \mathbf{t})$ の関数
- 共分散は target 側に、source 側を回転で持ってきて足したもの → 「両方の不確かさを足し合わせた誤差分布」
- $T$ について非線形 → Gauss-Newton で反復

### 17.4 VGICP の違い

VGICP (`small_gicp_vgicp`) は source も target も「ボクセル単位のガウス分布」として扱う。点と点ではなく **「点群分布と点群分布のマッチ」**。計算量が点数に比べてボクセル数 (一桁少ない) でスケールする。

---

## 18. カルマンフィルタの式

### 18.1 想定する世界

時刻 $k$ で状態 $\mathbf{x}_k$ を推定したい。

- **状態方程式**: $\mathbf{x}_{k+1} = f(\mathbf{x}_k, \mathbf{u}_k) + \mathbf{w}_k$, $\mathbf{w} \sim N(0, Q)$
- **観測方程式**: $\mathbf{z}_k = h(\mathbf{x}_k) + \mathbf{v}_k$, $\mathbf{v} \sim N(0, R)$

$Q$ がプロセスノイズ共分散、$R$ が観測ノイズ共分散。

### 18.2 予測ステップ

ノイズが正規分布なので、推定の不確かさも共分散行列 $P$ で表せます。$f, h$ を現在の推定点で線形化したヤコビアンを $F = \partial f/\partial \mathbf{x}$, $H = \partial h/\partial \mathbf{x}$ とおく:

$$
\hat{\mathbf{x}}^- = f(\hat{\mathbf{x}}, \mathbf{u})
$$

$$
P^- = F P F^T + G Q G^T
$$

($G$ はノイズ入口のヤコビアン、IMU では $Q$ を IMU の世界に持ってくる役)

### 18.3 更新ステップ

観測 $\mathbf{z}$ が来たら:

$$
S = H P^- H^T + R \quad \text{(innovation 共分散)}
$$

$$
K = P^- H^T S^{-1} \quad \text{(Kalman ゲイン)}
$$

$$
\hat{\mathbf{x}}^+ = \hat{\mathbf{x}}^- + K (\mathbf{z} - h(\hat{\mathbf{x}}^-))
$$

$$
P^+ = (I - K H) P^-
$$

### 18.4 Kalman ゲインの意味

$K$ は「観測 vs 予測の信頼度バランス」。

- 観測ノイズ $R$ が小 → 観測を信じる → $K$ 大
- 予測共分散 $P^-$ が大 (予測が信用できない) → 観測を信じる → $K$ 大

「自分の予測の自信」と「センサーの自信」を **共分散で自動的に重み比較** してくれるのが Kalman の本質。

---

## 19. Error-State Kalman Filter (ESKF)

### 19.1 動機

回転を含む状態 (姿勢) を素直に Kalman に入れると、$P$ の更新で「回転の足し算」が必要になり、SO(3) を逸脱します。**変数を誤差にすり替える** ことで解決:

- **名目状態 (nominal)** $\bar{\mathbf{x}}$: 「現在の最善推定値」。回転行列など群そのもの。
- **誤差状態 (error)** $\delta\mathbf{x}$: 名目からのズレを **リー代数 (ベクトル空間)** で表現。Kalman の変数として扱うのはこっち。
- **真の状態**: $\mathbf{x} = \bar{\mathbf{x}} \boxplus \delta\mathbf{x}$

誤差はゼロ近傍 (どんなに大きく見積もっても rad オーダー) なので、線形化が常に有効。$P$ は $\delta\mathbf{x}$ についての共分散なので、ベクトル空間で安全に持てる。

### 19.2 pylot_lio IESKF の状態 (15 次元)

| 添字 | 意味 | 名目変数 | 誤差変数 |
|---|---|---|---|
| 0–2 | 位置 [m] | $\mathbf{p}_w$ | $\delta\mathbf{p}$ |
| 3–5 | 速度 [m/s] | $\mathbf{v}_w$ | $\delta\mathbf{v}$ |
| 6–8 | 姿勢 | $R_{wb}$ (回転行列) | $\delta\boldsymbol{\phi}$ (リー代数) |
| 9–11 | 加速度バイアス | $\mathbf{b}_a$ | $\delta\mathbf{b}_a$ |
| 12–14 | ジャイロバイアス | $\mathbf{b}_g$ | $\delta\mathbf{b}_g$ |

### 19.3 名目状態の伝播 (IMU)

IMU の生値 $\mathbf{a}_m, \boldsymbol{\omega}_m$, ステップ $\Delta t$ で:

$$
\mathbf{p}_w^+ = \mathbf{p}_w + \mathbf{v}_w \Delta t + \tfrac{1}{2}\!\left(R_{wb}(\mathbf{a}_m - \mathbf{b}_a) + \mathbf{g}\right)\Delta t^2
$$

$$
\mathbf{v}_w^+ = \mathbf{v}_w + \big(R_{wb}(\mathbf{a}_m - \mathbf{b}_a) + \mathbf{g}\big)\Delta t
$$

$$
R_{wb}^+ = R_{wb} \cdot \mathrm{Exp}\big((\boldsymbol{\omega}_m - \mathbf{b}_g)\Delta t\big)
$$

バイアスはランダムウォーク: $\mathbf{b}_a^+ = \mathbf{b}_a$, $\mathbf{b}_g^+ = \mathbf{b}_g$ (期待値レベル)。

### 19.4 誤差状態のヤコビアン $F$

誤差 $\delta\mathbf{x}$ が $F$ で伝播 ($\delta\mathbf{x}^+ = F \delta\mathbf{x} + G \mathbf{w}$):

$$
F = \begin{pmatrix}
I & I\Delta t & 0 & 0 & 0 \\
0 & I & -R_{wb}[\mathbf{a}_m - \mathbf{b}_a]_\times \Delta t & -R_{wb}\Delta t & 0 \\
0 & 0 & \mathrm{Exp}\!\big(-(\boldsymbol{\omega}_m - \mathbf{b}_g)\Delta t\big) & 0 & -J_r \Delta t \\
0 & 0 & 0 & I & 0 \\
0 & 0 & 0 & 0 & I
\end{pmatrix}
$$

各ブロックは「Sola のチュートリアル §6」式の導出から直接書き起こしたもの。$J_r$ は §13 の右ヤコビアン (回転誤差伝播に出る)。

### 19.5 観測モデル (LiDAR スキャン)

GICP の結果 $T_{\mathrm{obs}}$ を観測値として使う場合、観測残差は:

$$
\mathbf{r} = \begin{pmatrix} \mathrm{Log}(R_{wb}^T R_{\mathrm{obs}}) \\ R_{wb}^T (\mathbf{t}_{\mathrm{obs}} - \mathbf{p}_w) \end{pmatrix} \in \mathbb{R}^6
$$

観測ヤコビアン $H$ は $\delta\mathbf{x}$ について偏微分したもの (位置と姿勢にだけ非ゼロブロック)。

### 19.6 更新の流れ

1. $S = H P H^T + R$
2. $K = P H^T S^{-1}$
3. $\delta\mathbf{x} = K \mathbf{r}$
4. $\bar{\mathbf{x}} \leftarrow \bar{\mathbf{x}} \boxplus \delta\mathbf{x}$ (誤差を名目に「乗せて」リセット)
5. $P \leftarrow (I - KH) P$ (実装では Joseph form $J_\phi P J_\phi^T$ で対称性も維持)

ステップ 4 の boxplus が、回転を SO(3) に留めたまま更新できる肝。

---

## 20. Iterated ESKF

### 20.1 動機

通常 ESKF は観測ヤコビアン $H$ を **現在の推定 $\hat{\mathbf{x}}$ で 1 回だけ評価** します。残差が大きい (= 線形化点が真値から遠い) と、1 回の更新では収束しない。

### 20.2 反復化

LiDAR 観測のように残差が大きく非線形なものは、Gauss-Newton 風に **同じ観測で複数回線形化** します:

```
loop k = 0..K_max:
  H_k  = ∂h/∂x at x̂_k
  S_k  = H_k P H_k^T + R
  K_k  = P H_k^T S_k^{-1}
  δx_k = K_k · (z - h(x̂_k)) - (I - K_k H_k) · (x̂_k ⊟ x̂_0)
            ↑観測残差              ↑「最初の推定からの蓄積補正」を保つ項
  x̂_{k+1} = x̂_0 ⊞ δx_k
  if ||δx_k|| < threshold: break
P_final = (I - K_K H_K) P
```

### 20.3 第 2 項の意味

普通の Gauss-Newton は線形化点を毎回更新しますが、それだと「Kalman の事前共分散 $P$ が表す事前情報」を使い切ってしまいます。IESKF では「最初の予測 $\hat{\mathbf{x}}_0$ からのズレ」を毎反復で考慮することで、事前情報を正しく保ったまま反復します (Sola §10 / FAST-LIO2 §3.4)。

### 20.4 pylot_lio での実装

- `max_iteration_per_scan` (既定 4): 上の K_max
- 各反復で IPointCloudMap::findNearestNeighbor を呼び直す (対応点も線形化点と一緒に更新する)

---

## 21. Tikhonov 正則化と縮退対応

### 21.1 縮退とは

廊下や対称構造で「点群幾何から決まらない方向」が出ます。この方向では:

- GICP の Hessian $H = J^T W J$ の固有値が極端に小さい
- $H \Delta\mathbf{x} = -\mathbf{g}$ を解くと $\Delta\mathbf{x}$ が爆発
- 1 フレームで姿勢が飛ぶ → マップが渦巻く

### 21.2 普通の Tikhonov

$H$ を $H + \lambda I$ に置き換える (全方向に同じだけリッジ)。これは縮退方向の暴走は止まるが、**well-conditioned 方向の精度も落とす**。

### 21.3 X-ICP / Tuna 2024 流

Hessian を 6×6 を [回転 3×3 | 並進 3×3] のブロック対角と見て、それぞれを固有値分解 $H_b = U_b \Lambda_b U_b^T$。

inlier 数 $n$ で正規化した固有値 $\lambda_i / n < \tau$ なる固有方向 $\mathbf{u}_i$ だけを縮退方向とみなし、

$$
H_{\mathrm{pen}} = \alpha \sum_{i \in \mathrm{deg}} \mathbf{u}_i \mathbf{u}_i^T
$$

これは「縮退方向にだけ単位行列を足す」ような行列。Hessian にこれを足し、右辺には「初期推定からのズレ」を引き戻す項を加える:

$$
(H + H_{\mathrm{pen}})\,\Delta\mathbf{x} = -\mathbf{g} - H_{\mathrm{pen}} \cdot \delta_{\mathrm{twist}}
$$

ここで $\delta_{\mathrm{twist}} = \hat{T}_{\mathrm{current}} \boxminus T_{\mathrm{initial}}$ (現在解と初期推定の差を $\mathbb{R}^6$ で表現)。

### 21.4 嬉しい性質

- 非縮退方向では $H_{\mathrm{pen}}$ がゼロなので **オリジナル GICP と完全一致** (精度を犠牲にしない)
- 縮退方向だけ「初期推定 (IMU や等速度モデル) に引き戻す」力が働く
- パラメータ 3 つ (`rotation_eigenvalue_threshold`, `translation_eigenvalue_threshold`, `regularization_base_factor`)

---

## 22. リザーバサンプリング Algorithm R

### 22.1 解きたい問題

ストリーム (= 全長 $N$ を事前に知らない、メモリに保持できない データ列) から、**一様乱択で $K$ 個** を選んで保持したい。

### 22.2 Vitter (1985) Algorithm R

- 最初の $K$ 個は無条件で容器に入れる
- $n$ 番目 ($n > K$) の要素について:
  - 乱数 $j \in [1, n]$ を引く
  - $j \le K$ なら容器の $j$ 番目を新要素と置き換える、そうでなければ捨てる

### 22.3 正しさ (確率)

最終的に容器の各スロットに $n$ 番目の要素が残る確率 = $K/n$。これがすべての $i = 1, \dots, N$ について同じ ($K/N$) であることを数学的帰納法で示せます (Vitter 原論文 §3)。

### 22.4 pylot_lio での使い道

`voxel_random_map` で各ボクセルに「最大 $K$ 個の生点」を保持。VoxelMap が点をガウス分布 1 個に潰すのに対し、こちらは生点を残すので 1 ボクセルに 2 面が偶然入っても情報が消えない。メモリは $K \times \mathrm{cells}$ で有界。

---

## 参考文献 (このドキュメントで実際に使った導出元)

- J. Sola, "Quaternion kinematics for the error-state Kalman filter" (2017) — リー代数、Rodrigues、右ヤコビアン、ESKF の決定版
- T. D. Barfoot, "State Estimation for Robotics" (Cambridge, 2017) — SO(3)/SE(3) と確率推論の総覧
- W. Xu and F. Zhang, "FAST-LIO2" (T-RO 2022) — IESKF の反復更新式
- A. Segal, D. Haehnel, S. Thrun, "Generalized-ICP" (RSS 2009) — GICP の cost 関数の元論文
- K. Koide et al., "Voxelized GICP" (ICRA 2021) — VGICP
- T. Tuna et al., "X-ICP" (arXiv:2408.11809, 2024) — 縮退方向 Tikhonov 正則化
- J. S. Vitter, "Random Sampling with a Reservoir" (ACM TOMS 1985) — Algorithm R
