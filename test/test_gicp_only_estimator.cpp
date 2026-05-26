// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "pylot_lio/estimator/gicp_only_estimator.hpp"
#include "pylot_lio/map/voxel_map.hpp"
#include "pylot_lio/registration/i_registration.hpp"
#include "pylot_lio/registration/plain_gicp_registration.hpp"

namespace pylot_lio
{

namespace
{

// テスト専用の registration スタブ。 align 呼び出しごとに事前設定した戻り値を返す。
// これにより「収束した結果として X を返した」「未収束だった」を制御してテストできる。
class StubRegistration : public IRegistration
{
public:
  AlignResult align(
    const PointCloud & /*source_cloud_body*/,
    const IPointCloudMap & /*map_world*/,
    const Eigen::Isometry3d & initial_transform_world_body) override
  {
    last_initial_guess_ = initial_transform_world_body;
    all_initial_guesses_.push_back(initial_transform_world_body);
    ++call_count_;
    AlignResult result;
    if (use_initial_guess_aware_cost_) {
      // 候補の良し悪しを cost で表現したいテスト用。 initial_guess の並進ノルムが
      // 「良い候補との距離」と相関するように cost を返す。
      result.transform_world_body = aware_result_pose_;
      result.final_cost = (initial_transform_world_body.translation() -
                           aware_best_initial_guess_.translation()).norm();
      result.converged = true;
      result.iterations = 1;
    } else {
      result.transform_world_body = return_pose_;
      result.iterations = 1;
      result.final_cost = return_cost_;
      result.converged = return_converged_;
    }
    return result;
  }

  std::string describe() const override { return "stub"; }

  Eigen::Isometry3d return_pose_ = Eigen::Isometry3d::Identity();
  bool return_converged_ = true;
  double return_cost_ = 0.0;
  Eigen::Isometry3d last_initial_guess_ = Eigen::Isometry3d::Identity();
  std::vector<Eigen::Isometry3d> all_initial_guesses_;
  int call_count_ = 0;

  // 「複数候補のうち、 ある特定の initial_guess が最良 (cost 最小)」を表現するモード。
  bool use_initial_guess_aware_cost_ = false;
  Eigen::Isometry3d aware_best_initial_guess_ = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d aware_result_pose_ = Eigen::Isometry3d::Identity();
};

// 1 点だけ入った点群を作って map に挿入し、 size()>0 にする (空マップ判定を外す)。
void primeMapWithOnePoint(VoxelMap & map)
{
  PointCloud cloud;
  Point point;
  point.x = 0.0f;
  point.y = 0.0f;
  point.z = 0.0f;
  cloud.push_back(point);
  map.insertScan(cloud, Eigen::Isometry3d::Identity());
}

constexpr int64_t kOneHundredMillisecondsInNanoseconds = 100'000'000;

}  // namespace

TEST(GicpOnlyEstimator, InitializeAndGetState)
{
  GicpOnlyEstimator estimator;
  EXPECT_FALSE(estimator.isInitialized());
  RobotState initial;
  initial.pose_world_body.translation() = Eigen::Vector3d(0.5, 0.0, 0.0);
  estimator.initialize(initial);
  EXPECT_TRUE(estimator.isInitialized());
  EXPECT_NEAR(estimator.getState().pose_world_body.translation().x(), 0.5, 1e-9);
  EXPECT_NEAR(estimator.getState().velocity_world.norm(), 0.0, 1e-9);
}

TEST(GicpOnlyEstimator, PredictWithImuIsNoOp)
{
  GicpOnlyEstimator estimator;
  estimator.initialize(RobotState{});
  ImuSample sample;
  sample.timestamp_ns = 0;
  sample.linear_acceleration_mps2 = Eigen::Vector3d(1.0, 0.0, 0.0);
  sample.angular_velocity_rps = Eigen::Vector3d::Zero();
  estimator.predictWithImu(sample);
  EXPECT_NEAR(estimator.getState().pose_world_body.translation().norm(), 0.0, 1e-9);
}

TEST(GicpOnlyEstimator, EmptyMapKeepsInitialGuessAndSkipsAlign)
{
  // map が空のときは align をスキップし、 initial_guess (= 現在 pose) を維持する。
  GicpOnlyEstimator estimator;
  estimator.initialize(RobotState{});
  VoxelMap map(VoxelMap::Config{});
  PlainGicpRegistration registration(PlainGicpRegistration::Config{});
  PointCloud empty_cloud;
  estimator.updateWithScan(empty_cloud, map, registration, 0);
  EXPECT_NEAR(estimator.getState().pose_world_body.translation().norm(), 0.0, 1e-9);
  EXPECT_FALSE(estimator.getDiagnostics().converged);
}

TEST(GicpOnlyEstimator, VelocityIsUpdatedFromConsecutiveScanTimestamps)
{
  // 2 スキャン分の align を通すと、 (位置差 / dt) で velocity_world が更新される。
  GicpOnlyEstimator estimator;
  estimator.initialize(RobotState{});

  VoxelMap map(VoxelMap::Config{});
  primeMapWithOnePoint(map);

  StubRegistration registration;
  PointCloud empty_cloud;

  // 1 スキャン目: pose を (0,0,0) のままにする。
  registration.return_pose_ = Eigen::Isometry3d::Identity();
  estimator.updateWithScan(
    empty_cloud, map, registration, kOneHundredMillisecondsInNanoseconds);
  EXPECT_NEAR(estimator.getState().velocity_world.norm(), 0.0, 1e-9);

  // 2 スキャン目: pose を (0.1, 0, 0) に動かす。 dt = 0.1s なので速度は 1 m/s。
  Eigen::Isometry3d second_pose = Eigen::Isometry3d::Identity();
  second_pose.translation() = Eigen::Vector3d(0.1, 0.0, 0.0);
  registration.return_pose_ = second_pose;
  estimator.updateWithScan(
    empty_cloud, map, registration, kOneHundredMillisecondsInNanoseconds * 2);

  EXPECT_NEAR(estimator.getState().velocity_world.x(), 1.0, 1e-6);
  EXPECT_NEAR(estimator.getState().velocity_world.y(), 0.0, 1e-9);
  EXPECT_NEAR(estimator.getState().velocity_world.z(), 0.0, 1e-9);
}

TEST(GicpOnlyEstimator, NonConvergedAlignKeepsPreviousPose)
{
  // align が未収束だったら pose は据え置きで diagnostics.converged=false。
  GicpOnlyEstimator estimator;
  RobotState initial;
  initial.pose_world_body.translation() = Eigen::Vector3d(0.5, 0.0, 0.0);
  estimator.initialize(initial);

  VoxelMap map(VoxelMap::Config{});
  primeMapWithOnePoint(map);

  StubRegistration registration;
  PointCloud empty_cloud;
  Eigen::Isometry3d wrong_pose = Eigen::Isometry3d::Identity();
  wrong_pose.translation() = Eigen::Vector3d(99.0, 99.0, 99.0);  // 飛んだ値
  registration.return_pose_ = wrong_pose;
  registration.return_converged_ = false;

  estimator.updateWithScan(
    empty_cloud, map, registration, kOneHundredMillisecondsInNanoseconds);

  // pose は initialize 時の (0.5, 0, 0) のまま (飛んだ wrong_pose は採用されない)。
  EXPECT_NEAR(estimator.getState().pose_world_body.translation().x(), 0.5, 1e-9);
  EXPECT_NEAR(estimator.getState().pose_world_body.translation().y(), 0.0, 1e-9);
  EXPECT_FALSE(estimator.getDiagnostics().converged);
}

TEST(GicpOnlyEstimator, ExtrapolationDeltaIsClamped)
{
  // 1 スキャン目で大きく飛ばす (例: 5 m) と、 2 スキャン目の initial_guess は
  // 「5m + 等速度で更に 5m」となり 10 m まで暴れるところを、 max_translation_m で
  // クランプして 5m + 1m = 6m に抑える。
  GicpOnlyEstimator::Config config;
  config.max_extrapolation_translation_m = 1.0;
  config.max_extrapolation_rotation_rad = 0.5;
  GicpOnlyEstimator estimator(config);
  estimator.initialize(RobotState{});

  VoxelMap map(VoxelMap::Config{});
  primeMapWithOnePoint(map);

  StubRegistration registration;
  PointCloud empty_cloud;

  // 1 スキャン目: pose を (5, 0, 0) にジャンプ (誤対応相当)。
  Eigen::Isometry3d first_pose = Eigen::Isometry3d::Identity();
  first_pose.translation() = Eigen::Vector3d(5.0, 0.0, 0.0);
  registration.return_pose_ = first_pose;
  estimator.updateWithScan(empty_cloud, map, registration, kOneHundredMillisecondsInNanoseconds);

  // 2 スキャン目: registration は何でも良いが、 initial_guess がクランプされているかを確認。
  registration.return_pose_ = first_pose;  // align 結果は問わない
  estimator.updateWithScan(empty_cloud, map, registration, kOneHundredMillisecondsInNanoseconds * 2);

  // クランプ後の initial_guess は 5m + 1m = 6m になっているはず。
  EXPECT_NEAR(registration.last_initial_guess_.translation().x(), 6.0, 1e-6);
}

TEST(GicpOnlyEstimator, SetPoseResetsExtrapolationAndVelocity)
{
  // PGO ジャンプを setPose で表現。 直後の updateWithScan で等速度モデルが
  // 動かない (= initial_guess == current_pose) ことを確認する。
  GicpOnlyEstimator estimator;
  estimator.initialize(RobotState{});

  VoxelMap map(VoxelMap::Config{});
  primeMapWithOnePoint(map);

  StubRegistration registration;
  PointCloud empty_cloud;

  // 2 スキャン進めて has_previous_pose_=true 状態に。
  Eigen::Isometry3d step1 = Eigen::Isometry3d::Identity();
  step1.translation() = Eigen::Vector3d(0.1, 0.0, 0.0);
  registration.return_pose_ = step1;
  estimator.updateWithScan(empty_cloud, map, registration, kOneHundredMillisecondsInNanoseconds);
  Eigen::Isometry3d step2 = Eigen::Isometry3d::Identity();
  step2.translation() = Eigen::Vector3d(0.2, 0.0, 0.0);
  registration.return_pose_ = step2;
  estimator.updateWithScan(empty_cloud, map, registration, kOneHundredMillisecondsInNanoseconds * 2);

  // PGO 修正で pose を強制的にジャンプ。
  Eigen::Isometry3d corrected = Eigen::Isometry3d::Identity();
  corrected.translation() = Eigen::Vector3d(10.0, 0.0, 0.0);
  estimator.setPose(corrected);
  EXPECT_NEAR(estimator.getState().velocity_world.norm(), 0.0, 1e-9);

  // 次スキャン: initial_guess は corrected pose そのものになる (等速度外挿しない)。
  registration.return_pose_ = corrected;
  estimator.updateWithScan(empty_cloud, map, registration, kOneHundredMillisecondsInNanoseconds * 3);
  EXPECT_NEAR(registration.last_initial_guess_.translation().x(), 10.0, 1e-9);
}

TEST(GicpOnlyEstimator, MultiCandidatePicksLowerCostInitialGuess)
{
  // enable_static_candidate=true なら align が 2 回呼ばれ、 cost が小さい候補が採用される。
  // ここでは「静止候補 (= 現状 pose) を best と宣言」して、 cost-aware モードで
  // 静止候補側の align 結果が採用されることを確認する。
  GicpOnlyEstimator::Config config;
  config.enable_static_candidate = true;
  GicpOnlyEstimator estimator(config);
  estimator.initialize(RobotState{});

  VoxelMap map(VoxelMap::Config{});
  primeMapWithOnePoint(map);

  StubRegistration registration;
  PointCloud empty_cloud;

  // 1 スキャン目: pose を (1, 0, 0) に動かす (これにより has_previous_pose_=true)。
  Eigen::Isometry3d step1 = Eigen::Isometry3d::Identity();
  step1.translation() = Eigen::Vector3d(1.0, 0.0, 0.0);
  registration.return_pose_ = step1;
  estimator.updateWithScan(empty_cloud, map, registration, kOneHundredMillisecondsInNanoseconds);

  // 2 スキャン目: cost-aware モードに切り替える。
  // 等速度外挿は (2, 0, 0) を生成し、 静止候補は (1, 0, 0)。
  // 静止候補側を best にすると、 cost-aware モードで cost=0 となり優先される。
  registration.use_initial_guess_aware_cost_ = true;
  registration.aware_best_initial_guess_ = step1;  // 静止候補と一致する位置
  Eigen::Isometry3d aware_result_pose = Eigen::Isometry3d::Identity();
  aware_result_pose.translation() = Eigen::Vector3d(1.05, 0.0, 0.0);
  registration.aware_result_pose_ = aware_result_pose;
  registration.call_count_ = 0;
  registration.all_initial_guesses_.clear();

  estimator.updateWithScan(empty_cloud, map, registration, kOneHundredMillisecondsInNanoseconds * 2);

  // align は 2 回呼ばれる (等速度候補 + 静止候補)。
  EXPECT_EQ(registration.call_count_, 2);
  // 採用結果は aware_result_pose になっている (cost 最小候補で得られた pose)。
  EXPECT_NEAR(estimator.getState().pose_world_body.translation().x(), 1.05, 1e-6);
}

TEST(GicpOnlyEstimator, SanityCheckRejectsHugeCorrection)
{
  // initial_guess と align 結果の差が max_correction_translation_m を超えたら
  // align 結果は棄却され、 initial_guess が採用される。
  GicpOnlyEstimator::Config config;
  config.max_correction_translation_m = 0.1;  // 厳しい閾値
  GicpOnlyEstimator estimator(config);
  estimator.initialize(RobotState{});

  VoxelMap map(VoxelMap::Config{});
  primeMapWithOnePoint(map);

  StubRegistration registration;
  PointCloud empty_cloud;

  // align は (5, 0, 0) を返すが、 initial_guess は (0, 0, 0) なので 5m の補正
  // → 0.1m を大幅に超えて棄却される。
  Eigen::Isometry3d huge_pose = Eigen::Isometry3d::Identity();
  huge_pose.translation() = Eigen::Vector3d(5.0, 0.0, 0.0);
  registration.return_pose_ = huge_pose;

  estimator.updateWithScan(empty_cloud, map, registration, kOneHundredMillisecondsInNanoseconds);

  // pose は initial_guess (=現状 pose の (0,0,0)) のまま、 align 結果は採用されない。
  EXPECT_NEAR(estimator.getState().pose_world_body.translation().norm(), 0.0, 1e-9);
  EXPECT_FALSE(estimator.getDiagnostics().converged);
}

TEST(GicpOnlyEstimator, StationaryDetectionSuppressesPoseUpdateAfterStreak)
{
  // align は毎回走らせる (新設計)。 align 結果 が現状 pose とほぼ同じ場合に
  // streak が積まれ、 streak >= required で pose 更新を抑制し velocity をゼロにする。
  // 旧設計と違い「align スキップ」はしない (ラッチアップを避けるため)。
  GicpOnlyEstimator::Config config;
  config.stationary_translation_threshold_m = 0.01;
  config.stationary_rotation_threshold_rad = 0.01;
  config.stationary_streak_required = 2;
  GicpOnlyEstimator estimator(config);
  estimator.initialize(RobotState{});

  VoxelMap map(VoxelMap::Config{});
  primeMapWithOnePoint(map);

  StubRegistration registration;
  PointCloud empty_cloud;

  // 静止状態: registration が常に (0, 0, 0) を返す = measured_step が常にゼロ。
  registration.return_pose_ = Eigen::Isometry3d::Identity();

  // 3 連続スキャンで align が 3 回呼ばれる (スキップしない、 これが新設計の特徴)。
  estimator.updateWithScan(empty_cloud, map, registration, kOneHundredMillisecondsInNanoseconds);
  estimator.updateWithScan(empty_cloud, map, registration, kOneHundredMillisecondsInNanoseconds * 2);
  estimator.updateWithScan(empty_cloud, map, registration, kOneHundredMillisecondsInNanoseconds * 3);

  EXPECT_EQ(registration.call_count_, 3);  // align は毎回走っている (= ラッチしない前提)
  EXPECT_NEAR(estimator.getState().velocity_world.norm(), 0.0, 1e-9);
  EXPECT_NEAR(estimator.getState().pose_world_body.translation().norm(), 0.0, 1e-9);
}

TEST(GicpOnlyEstimator, StationaryStateExitsWhenMotionResumes)
{
  // ★ regression test for stationary latch-up bug ★
  // 一度 stationary 状態に入っても、 ロボットが動き始めれば自動復帰する。
  // 旧設計では align をスキップしていたため measured_step を取得できず、
  // 「ずっと静止」と判定されてラッチし、 移動を検出できない致命バグがあった。
  GicpOnlyEstimator::Config config;
  config.stationary_translation_threshold_m = 0.01;
  config.stationary_rotation_threshold_rad = 0.01;
  config.stationary_streak_required = 2;
  GicpOnlyEstimator estimator(config);
  estimator.initialize(RobotState{});

  VoxelMap map(VoxelMap::Config{});
  primeMapWithOnePoint(map);

  StubRegistration registration;
  PointCloud empty_cloud;

  // フェーズ 1: 静止状態を 3 連続スキャンで作る → stationary mode 突入。
  registration.return_pose_ = Eigen::Isometry3d::Identity();
  for (int scan_index = 1; scan_index <= 3; ++scan_index) {
    estimator.updateWithScan(
      empty_cloud, map, registration,
      kOneHundredMillisecondsInNanoseconds * scan_index);
  }
  EXPECT_NEAR(estimator.getState().pose_world_body.translation().norm(), 0.0, 1e-9);

  // フェーズ 2: ロボットが急に 0.5 m 動いた状況。 registration は新位置を返す。
  Eigen::Isometry3d moved_pose = Eigen::Isometry3d::Identity();
  moved_pose.translation() = Eigen::Vector3d(0.5, 0.0, 0.0);
  registration.return_pose_ = moved_pose;
  estimator.updateWithScan(
    empty_cloud, map, registration, kOneHundredMillisecondsInNanoseconds * 4);

  // 旧バグ: align がスキップされていたので pose は (0,0,0) のまま (静止ラッチ)。
  // 新設計: align は走るので measured_step=0.5 > 閾値、 streak リセット、 pose 更新復活。
  EXPECT_NEAR(estimator.getState().pose_world_body.translation().x(), 0.5, 1e-6)
    << "stationary mode failed to exit when sensor detected motion (latch-up regression)";
  // 続く 1 スキャンで velocity も復活する (現状 (0.5) → 次 align (0.5) なら dt=0.1 で
  // 0 だが、 さらに動かして velocity を確認するのは別テストの責務とする)。
}

TEST(GicpOnlyEstimator, EmaSmoothsPoseTowardAlignResult)
{
  // ema_alpha_translation=0.5 で、 「現状 pose=(0)」 と 「align 結果=(0.2)」 の
  // 半分の (0.1) に補正される。
  GicpOnlyEstimator::Config config;
  config.ema_alpha_translation = 0.5;
  config.ema_alpha_rotation = 0.5;
  GicpOnlyEstimator estimator(config);
  estimator.initialize(RobotState{});

  VoxelMap map(VoxelMap::Config{});
  primeMapWithOnePoint(map);

  StubRegistration registration;
  PointCloud empty_cloud;

  // 1 スキャン目: has_previous_pose_=false なので EMA は適用されない (素通し)。
  Eigen::Isometry3d step1 = Eigen::Isometry3d::Identity();
  step1.translation() = Eigen::Vector3d(0.2, 0.0, 0.0);
  registration.return_pose_ = step1;
  estimator.updateWithScan(empty_cloud, map, registration, kOneHundredMillisecondsInNanoseconds);
  EXPECT_NEAR(estimator.getState().pose_world_body.translation().x(), 0.2, 1e-9);

  // 2 スキャン目: align 結果を (0.4) としつつ、 EMA で前 pose (0.2) との半分 (0.3) になる。
  Eigen::Isometry3d step2 = Eigen::Isometry3d::Identity();
  step2.translation() = Eigen::Vector3d(0.4, 0.0, 0.0);
  registration.return_pose_ = step2;
  estimator.updateWithScan(empty_cloud, map, registration, kOneHundredMillisecondsInNanoseconds * 2);

  EXPECT_NEAR(estimator.getState().pose_world_body.translation().x(), 0.3, 1e-6);
}

TEST(GicpOnlyEstimator, AccelerationCandidateExtrapolatesBeyondVelocity)
{
  // 等加速度候補は「等速度外挿に jerk_delta を加算」した位置を提案するので、
  // 等速度候補とは別の initial_guess が registration に渡されることを確認する。
  GicpOnlyEstimator::Config config;
  config.enable_acceleration_candidate = true;
  // 検証しやすいよう、 等速度モデルが clamp で歪まないように上限を緩める。
  config.max_extrapolation_translation_m = 10.0;
  GicpOnlyEstimator estimator(config);
  estimator.initialize(RobotState{});

  VoxelMap map(VoxelMap::Config{});
  primeMapWithOnePoint(map);

  StubRegistration registration;
  PointCloud empty_cloud;

  // 1 スキャン目: pose を (1, 0, 0) に。
  Eigen::Isometry3d step1 = Eigen::Isometry3d::Identity();
  step1.translation() = Eigen::Vector3d(1.0, 0.0, 0.0);
  registration.return_pose_ = step1;
  estimator.updateWithScan(empty_cloud, map, registration, kOneHundredMillisecondsInNanoseconds);

  // 2 スキャン目: pose を (3, 0, 0) に → delta は (2, 0, 0) で「加速」している。
  Eigen::Isometry3d step2 = Eigen::Isometry3d::Identity();
  step2.translation() = Eigen::Vector3d(3.0, 0.0, 0.0);
  registration.return_pose_ = step2;
  estimator.updateWithScan(empty_cloud, map, registration, kOneHundredMillisecondsInNanoseconds * 2);

  // 3 スキャン目: has_previous_delta_=true なので等加速度候補が有効化。
  // 等速度候補は (3 + 2 = 5, 0, 0)、 等加速度候補は (5 + jerk_delta) になる。
  // jerk_delta は max_extrapolation_translation_m=1.0 (デフォルト) で clamp されるため、
  // 加速度候補の initial_guess は約 (6, 0, 0) になる。
  registration.return_pose_ = step2;  // align 結果は据え置きでよい
  registration.all_initial_guesses_.clear();
  registration.call_count_ = 0;
  estimator.updateWithScan(empty_cloud, map, registration, kOneHundredMillisecondsInNanoseconds * 3);

  // align が 2 回 (等速度候補 + 等加速度候補) 呼ばれる。
  EXPECT_EQ(registration.call_count_, 2);
  // 2 つの initial_guess に「(5,0,0) (= 等速度)」と「(6,0,0) 付近 (= 等加速度 clamp 後)」が
  // 含まれている。
  bool has_velocity_guess = false;
  bool has_acceleration_guess = false;
  for (const auto & guess : registration.all_initial_guesses_) {
    if (std::abs(guess.translation().x() - 5.0) < 1e-6) {
      has_velocity_guess = true;
    }
    if (guess.translation().x() > 5.5 && guess.translation().x() < 6.5) {
      has_acceleration_guess = true;
    }
  }
  EXPECT_TRUE(has_velocity_guess);
  EXPECT_TRUE(has_acceleration_guess);
}

TEST(GicpOnlyEstimator, JerkSanityCheckRejectsAbruptDeltaChange)
{
  // max_jerk_translation_m を厳しく設定し、 前回 delta=(0.5,0,0) から
  // 突然 delta=(5,0,0) になるような align 結果は拒否する。
  GicpOnlyEstimator::Config config;
  config.max_jerk_translation_m = 0.2;  // 0.5 → 5 の差 (4.5m) より十分小さい閾値
  GicpOnlyEstimator estimator(config);
  estimator.initialize(RobotState{});

  VoxelMap map(VoxelMap::Config{});
  primeMapWithOnePoint(map);

  StubRegistration registration;
  PointCloud empty_cloud;

  // 1 スキャン目: pose を (0.5, 0, 0) に。
  Eigen::Isometry3d step1 = Eigen::Isometry3d::Identity();
  step1.translation() = Eigen::Vector3d(0.5, 0.0, 0.0);
  registration.return_pose_ = step1;
  estimator.updateWithScan(empty_cloud, map, registration, kOneHundredMillisecondsInNanoseconds);

  // 2 スキャン目: 一旦素直に進む → has_previous_delta_=true にする。
  Eigen::Isometry3d step2 = Eigen::Isometry3d::Identity();
  step2.translation() = Eigen::Vector3d(1.0, 0.0, 0.0);
  registration.return_pose_ = step2;
  estimator.updateWithScan(empty_cloud, map, registration, kOneHundredMillisecondsInNanoseconds * 2);

  // 3 スキャン目: align が (6, 0, 0) を返す。 等速度モデルなら (1.5, 0, 0) になるべき。
  // delta は (6 - 1) = 5、 前回 delta=(0.5)、 jerk = 4.5m → 閾値 0.2m を大きく超えて拒否。
  Eigen::Isometry3d abrupt_pose = Eigen::Isometry3d::Identity();
  abrupt_pose.translation() = Eigen::Vector3d(6.0, 0.0, 0.0);
  registration.return_pose_ = abrupt_pose;
  estimator.updateWithScan(empty_cloud, map, registration, kOneHundredMillisecondsInNanoseconds * 3);

  // jerk チェックで拒否され、 velocity_initial_guess (=1.5) が採用される。
  EXPECT_NEAR(estimator.getState().pose_world_body.translation().x(), 1.5, 1e-6);
  EXPECT_FALSE(estimator.getDiagnostics().converged);
}

}  // namespace pylot_lio

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
