// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
// pylot_lio LIO ノードの実行可能エントリ (ライブ ROS 購読モード)。
// ノード本体は include/pylot_lio/lio_node.hpp に定義され、 オフライン rosbag リーダ
// (lio_rosbag) からも再利用される。
#include "pylot_lio/lio_node.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<pylot_lio::LioNode>());
  rclcpp::shutdown();
  return 0;
}
