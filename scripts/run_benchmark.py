#!/usr/bin/env python3
# Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
"""rosbag2 を入力に、複数プリセットで pylot_lio を順番に走らせ、

各プリセットの推定軌跡を TUM 形式 (timestamp tx ty tz qx qy qz qw) で書き出す。

Mid-360 実機には Ground Truth が無い想定なので、ここでは「同じ rosbag を入力に、
各プリセットがどんな軌跡を吐いたか」を集める段階。相互比較は compute_relative_metrics.py に任せる。

使用例:
    python3 run_benchmark.py \\
        --bag /path/to/mid360_loop.bag \\
        --output_dir /tmp/pylot_lio_bench \\
        --presets ieskf_smallgicp_voxel hgo_plaingicp_normal gicp_only_kdtree

依存: ros2 (launch / bag), Python rclpy。
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


@dataclasses.dataclass
class BenchmarkRun:
    preset_name: str
    bag_path: Path
    output_dir: Path
    timeout_s: int

    @property
    def trajectory_path(self) -> Path:
        return self.output_dir / f"{self.preset_name}.tum"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", required=True, type=Path, help="入力 rosbag2 ディレクトリ")
    parser.add_argument(
        "--output_dir", required=True, type=Path,
        help="プリセットごとの TUM 軌跡を出力するディレクトリ")
    parser.add_argument(
        "--presets", nargs="+", required=True,
        help="評価するプリセット名 (config/presets/<name>.yaml に対応)")
    parser.add_argument(
        "--bag_rate", default=1.0, type=float,
        help="ros2 bag play --rate 引数")
    parser.add_argument(
        "--timeout_s", default=600, type=int,
        help="1 プリセットあたりの上限秒数")
    return parser.parse_args()


def launch_lio_node(preset_name: str, log_path: Path) -> subprocess.Popen:
    log_file = open(log_path, "w", encoding="utf-8")
    return subprocess.Popen(
        [
            "ros2", "launch", "pylot_lio", "lio.launch.py",
            f"preset:={preset_name}", "log_level:=warn",
        ],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )


def record_odom_to_tum(trajectory_path: Path) -> subprocess.Popen:
    """ros2 topic echo を使って /lio/odom を TUM 形式に整形して保存する補助プロセス。

    実装は別ファイル (odom_to_tum.py) に分けると見通しが良いが、シンプルさのためここに含める。
    本番では rclpy ノードに置き換えてもよい。"""
    # ros2 topic echo は CSV-like 出力を持たないため Python ワンライナーで整形する。
    inline_python = f"""
import rclpy, sys
from rclpy.node import Node
from nav_msgs.msg import Odometry

trajectory_handle = open({str(trajectory_path)!r}, 'w', encoding='utf-8')

class TumWriter(Node):
    def __init__(self):
        super().__init__('pylot_lio_tum_writer')
        self.create_subscription(Odometry, '/lio/odom', self.on_odom, 50)

    def on_odom(self, msg):
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        line = f'{{timestamp:.9f}} {{position.x}} {{position.y}} {{position.z}} ' \\
               f'{{orientation.x}} {{orientation.y}} {{orientation.z}} {{orientation.w}}\\n'
        trajectory_handle.write(line)
        trajectory_handle.flush()

rclpy.init()
node = TumWriter()
try:
    rclpy.spin(node)
finally:
    node.destroy_node()
    rclpy.shutdown()
    trajectory_handle.close()
"""
    return subprocess.Popen(
        ["python3", "-c", inline_python],
        preexec_fn=os.setsid,
    )


def play_bag(bag_path: Path, rate: float) -> subprocess.Popen:
    return subprocess.Popen(
        ["ros2", "bag", "play", str(bag_path), "--rate", f"{rate}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )


def kill_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGINT)
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_one(benchmark_run: BenchmarkRun, bag_rate: float) -> bool:
    benchmark_run.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = benchmark_run.output_dir / f"{benchmark_run.preset_name}.log"

    print(f"[pylot_lio bench] start preset={benchmark_run.preset_name}")
    lio_process = launch_lio_node(benchmark_run.preset_name, log_path)
    time.sleep(3.0)  # ノード起動待ち

    tum_writer = record_odom_to_tum(benchmark_run.trajectory_path)
    bag_process = play_bag(benchmark_run.bag_path, bag_rate)

    start_time = time.time()
    success = True
    try:
        while True:
            if bag_process.poll() is not None:
                break
            if time.time() - start_time > benchmark_run.timeout_s:
                print(f"[pylot_lio bench] timeout preset={benchmark_run.preset_name}")
                success = False
                break
            time.sleep(0.5)
    finally:
        kill_process_group(bag_process)
        kill_process_group(tum_writer)
        kill_process_group(lio_process)

    print(f"[pylot_lio bench] done preset={benchmark_run.preset_name} "
          f"trajectory={benchmark_run.trajectory_path}")
    return success


def main() -> int:
    args = parse_arguments()
    if not args.bag.exists():
        print(f"bag not found: {args.bag}", file=sys.stderr)
        return 2

    results = []
    for preset_name in args.presets:
        benchmark_run = BenchmarkRun(
            preset_name=preset_name,
            bag_path=args.bag,
            output_dir=args.output_dir,
            timeout_s=args.timeout_s,
        )
        ok = run_one(benchmark_run, args.bag_rate)
        results.append((preset_name, ok))

    print("[pylot_lio bench] summary:")
    for preset_name, ok in results:
        print(f"  {preset_name}: {'OK' if ok else 'FAIL'}")
    return 0 if all(ok for _, ok in results) else 1


if __name__ == "__main__":
    sys.exit(main())
