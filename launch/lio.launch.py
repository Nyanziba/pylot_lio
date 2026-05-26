"""pylot_lio 起動ランチ。

使用例:

    # Livox Mid-360 (デフォルト: /livox/lidar + /livox/imu)
    ros2 launch pylot_lio lio.launch.py preset:=ieskf_smallgicp_voxel

    # Velodyne + xsens IMU の rosbag を流す場合 (トピック名のみ差し替え)
    ros2 launch pylot_lio lio.launch.py \\
        preset:=ieskf_smallgicp_voxel \\
        input_cloud_topic:=/velodyne_points \\
        input_imu_topic:=/imu/data

    # 出力トピックを変える / TF 出力を抑制する
    ros2 launch pylot_lio lio.launch.py \\
        output_odom_topic:=/my/odom \\
        publish_tf:=false

引数は config/mid360.yaml と preset YAML より優先されます (起動時 1 回読み取り)。
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

    # ===== 引数定義 =====
    # 4 つのバックエンドはプリセット YAML から読む。トピック / フレーム ID だけは
    # rosbag が変わるたびに頻繁に変えるので launch 引数として露出する。
    launch_arguments = [
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
            description='起動直後の IMU 平均から重力ベクトルを自動推定 (true/false). '
                        'bag 冒頭で車両が動いている場合は false にして手動指定を推奨'),
        DeclareLaunchArgument(
            'gravity_world_x', default_value='0.0',
            description='重力ベクトル X 成分 (world frame, m/s^2). auto_estimate_gravity=false 時のみ使用'),
        DeclareLaunchArgument(
            'gravity_world_y', default_value='0.0',
            description='重力ベクトル Y 成分'),
        DeclareLaunchArgument(
            'gravity_world_z', default_value='0.0',
            description='重力ベクトル Z 成分 (REP-103 z-up なら -9.81, NED なら +9.81)'),
    ]

    # preset YAML をパラメータファイル列の 2 番目に挿入する。
    # parameters=[file1, file2, dict] は後勝ちなので preset がデフォルトを上書きする。
    # 空文字列を渡すと launch_ros が "." をパスとみなして警告を出すため、空のときは
    # デフォルト YAML を再度指定する (= 二重ロードだが副作用なし) でフォールバックする。
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

    # ROS パラメータ dict は dict キーごとに override を効かせるため、空文字列のときだけ
    # スキップするロジックは Python 側で組み立てると複雑になる。
    # 代わりに launch_ros の OpaqueFunction を使うほうがきれいだが、依存を増やしたくないので
    # node 内側で「空文字列なら何もしない」判定をする戦略を採る。
    # → lio_node.cpp の declareAndReadParameters で空文字列を「指定なし」とみなすよう実装済み。
    override_parameters = {
        'input_cloud_topic': LaunchConfiguration('input_cloud_topic'),
        'input_imu_topic': LaunchConfiguration('input_imu_topic'),
        'output_odom_topic': LaunchConfiguration('output_odom_topic'),
        'output_cloud_topic': LaunchConfiguration('output_cloud_topic'),
        'output_diag_topic': LaunchConfiguration('output_diag_topic'),
        'world_frame_id': LaunchConfiguration('world_frame_id'),
        'body_frame_id': LaunchConfiguration('body_frame_id'),
        # bool / double は文字列のままだと declare_parameter<T> の型と合わずに reject されるので
        # ParameterValue で明示的に型コエースする。
        'auto_estimate_gravity':
            ParameterValue(LaunchConfiguration('auto_estimate_gravity'), value_type=bool),
        'gravity_world_x':
            ParameterValue(LaunchConfiguration('gravity_world_x'), value_type=float),
        'gravity_world_y':
            ParameterValue(LaunchConfiguration('gravity_world_y'), value_type=float),
        'gravity_world_z':
            ParameterValue(LaunchConfiguration('gravity_world_z'), value_type=float),
    }

    lio_node = Node(
        package='pylot_lio',
        executable='lio_node',
        name='pylot_lio',
        output='screen',
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
        parameters=[default_config, preset_file_substitution, override_parameters],
    )

    return LaunchDescription(launch_arguments + [lio_node])
