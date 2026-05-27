// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
//
// オフライン rosbag リーダ (GLIM の glim_rosbag 相当)。
// rosbag2 を直接読み、 点群 / IMU メッセージをタイムスタンプ (= 記録) 順に
// LioNode へ *直接* 投入する。 DDS / ros2 bag play を経由しないため:
//   - 再生レートに依存せず、 ハードが許す限り高速に処理が走る。
//   - キュー溢れによるフレーム取りこぼしが原理的に起きない (全フレーム処理)。
// 高速で bag を流しても破綻しない「オフライン処理モード」を提供する。
//
// 使い方:
//   ros2 run pylot_lio lio_rosbag <bag_path> \
//       --ros-args --params-file <mid360.yaml> --params-file <presets/metal_vgicp.yaml> \
//       -p input_cloud_topic:=/livox/lidar -p input_imu_topic:=/livox/imu \
//       -p input_cloud_format:=livox_custom
//
// パラメータ (トピック名・形式・preset) は通常の LioNode と同じものを使う。
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/serialization.hpp>
#include <rosbag2_cpp/reader.hpp>

#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "pylot_lio/lio_node.hpp"

#ifdef PYLOT_LIO_HAS_LIVOX_DRIVER
#include <livox_ros_driver2/msg/custom_msg.hpp>
#endif

namespace
{

// --ros-args より前にある最初の位置引数を bag パスとして取り出す。
std::string extractBagPath(int argc, char ** argv)
{
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--ros-args") {
      break;  // 以降は ROS 引数。
    }
    if (!arg.empty() && arg[0] != '-') {
      return arg;
    }
  }
  return std::string();
}

}  // namespace

int main(int argc, char ** argv)
{
  const std::string bag_path = extractBagPath(argc, argv);
  if (bag_path.empty()) {
    std::fprintf(
      stderr,
      "usage: lio_rosbag <bag_path> --ros-args --params-file <yaml> [-p input_cloud_topic:=...]\n");
    return 1;
  }

  rclcpp::init(argc, argv);
  auto node = std::make_shared<pylot_lio::LioNode>();

  // トピック名・点群形式は LioNode が宣言済みのパラメータから取得する
  // (購読経路と同一の設定を使うため二重管理しない)。
  const std::string cloud_topic = node->get_parameter("input_cloud_topic").as_string();
  const std::string imu_topic = node->get_parameter("input_imu_topic").as_string();
  const std::string cloud_format = node->get_parameter("input_cloud_format").as_string();

  RCLCPP_INFO(
    node->get_logger(),
    "[lio_rosbag] offline mode: bag='%s' cloud_topic='%s' (%s) imu_topic='%s'",
    bag_path.c_str(), cloud_topic.c_str(), cloud_format.c_str(), imu_topic.c_str());

  rosbag2_cpp::Reader reader;
  try {
    reader.open(bag_path);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(node->get_logger(), "[lio_rosbag] failed to open bag: %s", e.what());
    rclcpp::shutdown();
    return 1;
  }

  rclcpp::Serialization<sensor_msgs::msg::Imu> imu_serialization;
  rclcpp::Serialization<sensor_msgs::msg::PointCloud2> cloud2_serialization;
#ifdef PYLOT_LIO_HAS_LIVOX_DRIVER
  rclcpp::Serialization<livox_ros_driver2::msg::CustomMsg> livox_serialization;
#endif
  const bool cloud_is_livox = (cloud_format == "livox_custom");

  std::uint64_t imu_count = 0;
  std::uint64_t cloud_count = 0;
  std::uint64_t message_count = 0;

  // bag 内の最初/最後のメッセージ時刻 (ナノ秒) から bag の実時間長を推定し、
  // 処理 wall time と比較して「実時間の何倍速で処理できたか」を出す。
  std::int64_t first_bag_time_ns = -1;
  std::int64_t last_bag_time_ns = 0;

  // spin_some を毎メッセージ呼ぶと数十万回のポーリングが支配的になるため間引く。
  // odom / cloud_world は callback 内 publish() で spin 無しでも送出される。 spin_some は
  // full-map timer や watchdog を時々回すため (= RViz の全体地図更新) だけに使う。
  constexpr int kSpinEveryNMessages = 512;
  // 進捗ログの間隔 (cloud 数)。
  constexpr int kLogEveryNClouds = 200;

  const auto wall_start = std::chrono::steady_clock::now();

  // bag を記録順 (= おおむねタイムスタンプ順、 IMU が先行スキャンより前に来る因果順) に
  // 読み、 該当トピックを LioNode へ直接投入する。 再生レート制御は一切せず最大速度で回す。
  while (rclcpp::ok() && reader.has_next()) {
    rosbag2_storage::SerializedBagMessageSharedPtr bag_message = reader.read_next();
    const std::string & topic = bag_message->topic_name;
    const std::int64_t bag_time_ns = bag_message->recv_timestamp;
    if (first_bag_time_ns < 0) {
      first_bag_time_ns = bag_time_ns;
    }
    last_bag_time_ns = bag_time_ns;
    rclcpp::SerializedMessage serialized(*bag_message->serialized_data);

    if (topic == imu_topic) {
      auto imu_msg = std::make_shared<sensor_msgs::msg::Imu>();
      imu_serialization.deserialize_message(&serialized, imu_msg.get());
      node->injectImu(imu_msg);
      ++imu_count;
    } else if (topic == cloud_topic) {
      if (cloud_is_livox) {
#ifdef PYLOT_LIO_HAS_LIVOX_DRIVER
        auto cloud_msg = std::make_shared<livox_ros_driver2::msg::CustomMsg>();
        livox_serialization.deserialize_message(&serialized, cloud_msg.get());
        node->injectLivoxCustomCloud(cloud_msg);
        ++cloud_count;
#else
        RCLCPP_ERROR_ONCE(
          node->get_logger(),
          "[lio_rosbag] input_cloud_format='livox_custom' but built without "
          "livox_ros_driver2; cannot deserialize.");
#endif
      } else {
        auto cloud_msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
        cloud2_serialization.deserialize_message(&serialized, cloud_msg.get());
        node->injectLidarCloud(cloud_msg);
        ++cloud_count;
      }
      if (cloud_count % kLogEveryNClouds == 0) {
        const double elapsed_s =
          std::chrono::duration<double>(
            std::chrono::steady_clock::now() - wall_start).count();
        RCLCPP_INFO(
          node->get_logger(),
          "[lio_rosbag] %llu clouds processed (%.1f clouds/s)",
          static_cast<unsigned long long>(cloud_count),
          elapsed_s > 0.0 ? cloud_count / elapsed_s : 0.0);
      }
    }

    if (++message_count % kSpinEveryNMessages == 0) {
      rclcpp::spin_some(node);
    }
  }

  const double wall_s =
    std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();
  const double bag_s = (first_bag_time_ns >= 0)
    ? static_cast<double>(last_bag_time_ns - first_bag_time_ns) * 1e-9 : 0.0;
  const double realtime_factor = (wall_s > 0.0) ? bag_s / wall_s : 0.0;

  RCLCPP_INFO(
    node->get_logger(),
    "[lio_rosbag] done: %llu clouds, %llu imu in %.2f s "
    "(bag=%.2f s, %.1f clouds/s, %.1fx realtime).",
    static_cast<unsigned long long>(cloud_count),
    static_cast<unsigned long long>(imu_count),
    wall_s, bag_s, wall_s > 0.0 ? cloud_count / wall_s : 0.0, realtime_factor);

  rclcpp::shutdown();
  return 0;
}
