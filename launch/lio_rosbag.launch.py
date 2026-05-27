"""pylot_lio オフライン rosbag 処理ランチ (GLIM の glim_rosbag 相当)。

lio_rosbag 実行ファイルを起動する。 rosbag2 を直接読み、 点群 / IMU を記録順に
LIO pipeline へ直接投入する (DDS / ros2 bag play を経由しない)。 再生レートに依存せず
ハードが許す限り高速に、 フレーム取りこぼしなく全フレームを処理する。

使用例:

    # Livox Mid-360 の bag を GPU-only metal_vgicp でオフライン処理
    ros2 launch pylot_lio lio_rosbag.launch.py \\
        bag:=/path/to/livox_bag \\
        preset:=metal_vgicp \\
        input_cloud_topic:=/livox/lidar \\
        input_imu_topic:=/livox/imu \\
        input_cloud_format:=livox_custom

    # Velodyne + xsens の bag (PointCloud2)
    ros2 launch pylot_lio lio_rosbag.launch.py \\
        bag:=/path/to/velodyne_bag \\
        preset:=velodyne_xsens \\
        input_cloud_topic:=/velodyne_points \\
        input_imu_topic:=/imu/data \\
        input_cloud_format:=pointcloud2

注意:
  - bag は必須。 指定しないと lio_rosbag が usage を出して終了する。
  - オフラインでは TF lookup が回りにくいので extrinsic_source は "config" を推奨
    (preset / パラメータで指定)。
  - 引数は config/mid360.yaml と preset YAML より優先される (起動時 1 回読み取り)。
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    package_share_directory = get_package_share_directory('pylot_lio')
    default_config = os.path.join(package_share_directory, 'config', 'mid360.yaml')

    launch_arguments = [
        DeclareLaunchArgument(
            'bag', default_value='',
            description='rosbag2 ディレクトリのパス (必須)'),
        DeclareLaunchArgument(
            'preset', default_value='',
            description='プリセット名 (config/presets/<name>.yaml). 空ならデフォルトのみ'),
        DeclareLaunchArgument(
            'log_level', default_value='info',
            description='ノードログレベル'),

        DeclareLaunchArgument(
            'input_cloud_topic', default_value='/livox/lidar',
            description='点群入力トピック (例: /livox/lidar, /velodyne_points)'),
        DeclareLaunchArgument(
            'input_imu_topic', default_value='/livox/imu',
            description='IMU 入力トピック (例: /livox/imu, /imu/data)'),
        DeclareLaunchArgument(
            'input_cloud_format', default_value='pointcloud2',
            description='点群形式: "livox_custom" (Livox CustomMsg) | "pointcloud2"'),
        DeclareLaunchArgument(
            'output_odom_topic', default_value='/lio/odom',
            description='odometry 出力トピック'),
        DeclareLaunchArgument(
            'output_cloud_topic', default_value='/lio/cloud_world',
            description='world 点群出力トピック'),
        DeclareLaunchArgument(
            'output_diag_topic', default_value='/lio/diag',
            description='診断 JSON 出力トピック'),

        DeclareLaunchArgument(
            'world_frame_id', default_value='world',
            description='world フレーム名'),
        DeclareLaunchArgument(
            'body_frame_id', default_value='base_link',
            description='body フレーム名'),

        DeclareLaunchArgument(
            'auto_estimate_gravity', default_value='true',
            description='起動直後の IMU 平均から重力ベクトルを自動推定 (true/false)'),
        DeclareLaunchArgument(
            'gravity_world_x', default_value='0.0',
            description='重力ベクトル X 成分 (world frame, m/s^2)'),
        DeclareLaunchArgument(
            'gravity_world_y', default_value='0.0',
            description='重力ベクトル Y 成分'),
        DeclareLaunchArgument(
            'gravity_world_z', default_value='0.0',
            description='重力ベクトル Z 成分 (REP-103 z-up なら -9.81)'),
    ]

    # preset YAML をパラメータファイル列の 2 番目に挿入 (後勝ちでデフォルトを上書き)。
    # 空文字列のときはデフォルト YAML を再指定 (二重ロードだが副作用なし)。
    preset_file_substitution = PythonExpression([
        '"',
        os.path.join(package_share_directory, 'config', 'presets'),
        '/" + "',
        LaunchConfiguration('preset'),
        '" + ".yaml" if "',
        LaunchConfiguration('preset'),
        '" else "',
        default_config,
        '"',
    ])

    override_parameters = {
        'input_cloud_topic': LaunchConfiguration('input_cloud_topic'),
        'input_imu_topic': LaunchConfiguration('input_imu_topic'),
        'input_cloud_format': LaunchConfiguration('input_cloud_format'),
        'output_odom_topic': LaunchConfiguration('output_odom_topic'),
        'output_cloud_topic': LaunchConfiguration('output_cloud_topic'),
        'output_diag_topic': LaunchConfiguration('output_diag_topic'),
        'world_frame_id': LaunchConfiguration('world_frame_id'),
        'body_frame_id': LaunchConfiguration('body_frame_id'),
        'auto_estimate_gravity':
            ParameterValue(LaunchConfiguration('auto_estimate_gravity'), value_type=bool),
        'gravity_world_x':
            ParameterValue(LaunchConfiguration('gravity_world_x'), value_type=float),
        'gravity_world_y':
            ParameterValue(LaunchConfiguration('gravity_world_y'), value_type=float),
        'gravity_world_z':
            ParameterValue(LaunchConfiguration('gravity_world_z'), value_type=float),
    }

    # bag パスは lio_rosbag の位置引数として --ros-args より前に渡す
    # (lio_rosbag は --ros-args 以前の最初の非フラグ引数を bag パスとみなす)。
    lio_rosbag_node = Node(
        package='pylot_lio',
        executable='lio_rosbag',
        name='pylot_lio',
        output='screen',
        arguments=[
            LaunchConfiguration('bag'),
            '--ros-args', '--log-level', LaunchConfiguration('log_level'),
        ],
        parameters=[default_config, preset_file_substitution, override_parameters],
    )

    return LaunchDescription(launch_arguments + [lio_rosbag_node])
