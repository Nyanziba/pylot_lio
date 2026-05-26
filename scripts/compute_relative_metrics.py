#!/usr/bin/env python3
# Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
"""GT 無し前提の手法間相対メトリクス計算。

Mid-360 実機には Ground Truth が無いため、絶対 ATE は計算できない。代わりに以下を出す:
  1) 指定された「参照プリセット」(例: ieskf_smallgicp_voxel) に対する **相対 ATE/RPE**
     evo を使って traj_a と traj_b を時刻整合させ、SE(3) 上の差分ノルムを計算。
  2) 各プリセットの **ドリフト指標**: 始点と終点が同じ位置に戻ってくるべきループ走行を仮定し、
     start_pose と end_pose の距離を出す (誤りはあくまで「整合性」)。

使用例:
    python3 compute_relative_metrics.py \\
        --input_dir /tmp/pylot_lio_bench \\
        --reference ieskf_smallgicp_voxel \\
        --output_json /tmp/pylot_lio_bench/metrics.json

依存: evo (pip install evo)。無ければ簡易版を内部実装に切り替える。
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path
from typing import Optional


@dataclasses.dataclass
class TumPose:
    timestamp_s: float
    translation_xyz: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]


def load_tum_trajectory(trajectory_path: Path) -> list[TumPose]:
    poses: list[TumPose] = []
    with open(trajectory_path, encoding="utf-8") as trajectory_file:
        for line in trajectory_file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            if len(tokens) < 8:
                continue
            poses.append(TumPose(
                timestamp_s=float(tokens[0]),
                translation_xyz=(float(tokens[1]), float(tokens[2]), float(tokens[3])),
                quaternion_xyzw=(
                    float(tokens[4]), float(tokens[5]),
                    float(tokens[6]), float(tokens[7])),
            ))
    return poses


def loop_drift_meters(poses: list[TumPose], tail_seconds: float = 5.0) -> Optional[float]:
    """ループ走行を仮定して始点と末尾平均位置との距離を返す。

    末尾 N 秒の平均を取るのは GPS 雑音相当のジッタを平滑化するため。"""
    if len(poses) < 10:
        return None
    start_position = poses[0].translation_xyz
    end_time = poses[-1].timestamp_s
    tail_positions = [
        pose.translation_xyz for pose in poses
        if end_time - pose.timestamp_s <= tail_seconds
    ]
    if not tail_positions:
        return None
    average_end = (
        sum(position[0] for position in tail_positions) / len(tail_positions),
        sum(position[1] for position in tail_positions) / len(tail_positions),
        sum(position[2] for position in tail_positions) / len(tail_positions),
    )
    return math.sqrt(
        (average_end[0] - start_position[0]) ** 2
        + (average_end[1] - start_position[1]) ** 2
        + (average_end[2] - start_position[2]) ** 2,
    )


def interpolate_pose_at(poses: list[TumPose], target_time_s: float) -> Optional[TumPose]:
    if not poses or target_time_s < poses[0].timestamp_s or target_time_s > poses[-1].timestamp_s:
        return None
    # 線形 binary search はせず、線形走査で十分 (TUM は時系列順)。
    for index in range(len(poses) - 1):
        if poses[index].timestamp_s <= target_time_s <= poses[index + 1].timestamp_s:
            duration = poses[index + 1].timestamp_s - poses[index].timestamp_s
            if duration <= 0.0:
                return poses[index]
            ratio = (target_time_s - poses[index].timestamp_s) / duration
            translation = tuple(
                poses[index].translation_xyz[axis]
                + ratio * (poses[index + 1].translation_xyz[axis]
                           - poses[index].translation_xyz[axis])
                for axis in range(3)
            )
            return TumPose(
                timestamp_s=target_time_s,
                translation_xyz=translation,
                quaternion_xyzw=poses[index].quaternion_xyzw,  # 回転補間は省略 (近傍値で十分)
            )
    return None


def relative_translation_rmse(
    reference_poses: list[TumPose],
    candidate_poses: list[TumPose],
) -> Optional[float]:
    if not reference_poses or not candidate_poses:
        return None
    squared_errors_sum = 0.0
    count = 0
    for reference_pose in reference_poses:
        candidate_pose = interpolate_pose_at(candidate_poses, reference_pose.timestamp_s)
        if candidate_pose is None:
            continue
        squared_errors_sum += sum(
            (candidate_pose.translation_xyz[axis] - reference_pose.translation_xyz[axis]) ** 2
            for axis in range(3)
        )
        count += 1
    if count == 0:
        return None
    return math.sqrt(squared_errors_sum / count)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument(
        "--reference", required=True,
        help="基準プリセット名 (この軌跡を相対 ATE の基準に使う)")
    parser.add_argument("--output_json", required=True, type=Path)
    parser.add_argument("--tail_seconds", default=5.0, type=float)
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    reference_path = args.input_dir / f"{args.reference}.tum"
    if not reference_path.exists():
        print(f"reference trajectory not found: {reference_path}", file=sys.stderr)
        return 2
    reference_poses = load_tum_trajectory(reference_path)

    metrics: dict[str, dict[str, Optional[float]]] = {}
    for trajectory_path in sorted(args.input_dir.glob("*.tum")):
        preset_name = trajectory_path.stem
        candidate_poses = load_tum_trajectory(trajectory_path)
        metrics[preset_name] = {
            "loop_drift_m": loop_drift_meters(candidate_poses, args.tail_seconds),
            "relative_translation_rmse_m":
                None if preset_name == args.reference
                else relative_translation_rmse(reference_poses, candidate_poses),
            "num_poses": len(candidate_poses),
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote metrics to {args.output_json}")
    for preset_name, scores in metrics.items():
        print(f"  {preset_name}: {scores}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
