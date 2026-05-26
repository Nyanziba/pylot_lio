#!/usr/bin/env python3
# Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
"""run_benchmark.py + compute_relative_metrics.py の出力から、

ブラウザで開ける単一 HTML ダッシュボードを生成する。Plotly が入っていれば対話 3D グラフ、
無ければプレーン HTML テーブルにフォールバックする。

使用例:
    python3 make_dashboard.py \\
        --input_dir /tmp/pylot_lio_bench \\
        --metrics /tmp/pylot_lio_bench/metrics.json \\
        --output_html /tmp/pylot_lio_bench/dashboard.html

依存: plotly (オプション)。
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Iterable


def load_tum_xyz(trajectory_path: Path) -> tuple[list[float], list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    with open(trajectory_path, encoding="utf-8") as trajectory_file:
        for line in trajectory_file:
            tokens = line.strip().split()
            if len(tokens) < 8:
                continue
            xs.append(float(tokens[1]))
            ys.append(float(tokens[2]))
            zs.append(float(tokens[3]))
    return xs, ys, zs


def build_plotly_html(
    trajectory_files: Iterable[Path],
    metrics: dict[str, dict],
) -> str:
    import plotly.graph_objects as plotly_graph  # type: ignore

    figure = plotly_graph.Figure()
    for trajectory_path in sorted(trajectory_files):
        preset_name = trajectory_path.stem
        xs, ys, zs = load_tum_xyz(trajectory_path)
        figure.add_trace(plotly_graph.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines",
            name=preset_name,
        ))
    figure.update_layout(
        title="pylot_lio: estimated trajectories",
        scene={
            "xaxis_title": "x [m]",
            "yaxis_title": "y [m]",
            "zaxis_title": "z [m]",
            "aspectmode": "data",
        },
        legend={"orientation": "h"},
    )
    trajectory_html = figure.to_html(full_html=False, include_plotlyjs="cdn")

    rows = []
    for preset_name in sorted(metrics.keys()):
        scores = metrics[preset_name]
        rows.append(
            "<tr>"
            f"<td>{html.escape(preset_name)}</td>"
            f"<td>{scores.get('loop_drift_m')}</td>"
            f"<td>{scores.get('relative_translation_rmse_m')}</td>"
            f"<td>{scores.get('num_poses')}</td>"
            "</tr>")
    table_html = (
        "<table border='1' cellpadding='4'>"
        "<thead><tr>"
        "<th>preset</th><th>loop_drift_m</th>"
        "<th>relative_translation_rmse_m</th><th>num_poses</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>")

    return f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<title>pylot_lio dashboard</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px;}}</style>
</head><body>
<h1>pylot_lio benchmark dashboard</h1>
<h2>Estimated trajectories</h2>
{trajectory_html}
<h2>Relative metrics</h2>
{table_html}
</body></html>
"""


def build_fallback_html(
    trajectory_files: Iterable[Path],
    metrics: dict[str, dict],
) -> str:
    rows = []
    for preset_name in sorted(metrics.keys()):
        scores = metrics[preset_name]
        rows.append(
            "<tr>"
            f"<td>{html.escape(preset_name)}</td>"
            f"<td>{scores.get('loop_drift_m')}</td>"
            f"<td>{scores.get('relative_translation_rmse_m')}</td>"
            f"<td>{scores.get('num_poses')}</td>"
            "</tr>")
    table_html = (
        "<table border='1' cellpadding='4'>"
        "<thead><tr>"
        "<th>preset</th><th>loop_drift_m</th>"
        "<th>relative_translation_rmse_m</th><th>num_poses</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>")
    trajectory_files_listed = "".join(
        f"<li>{html.escape(str(path))}</li>" for path in sorted(trajectory_files))
    return f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<title>pylot_lio dashboard (no plotly)</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px;}}</style>
</head><body>
<h1>pylot_lio benchmark dashboard (no plotly)</h1>
<p>plotly が見つからないので軌跡可視化はスキップ。pip install plotly で有効化される。</p>
<h2>Trajectory files</h2>
<ul>{trajectory_files_listed}</ul>
<h2>Relative metrics</h2>
{table_html}
</body></html>
"""


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--output_html", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    if not args.metrics.exists():
        print(f"metrics JSON not found: {args.metrics}", file=sys.stderr)
        return 2
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    trajectory_files = list(args.input_dir.glob("*.tum"))

    try:
        import plotly.graph_objects  # noqa: F401  type: ignore
        html_text = build_plotly_html(trajectory_files, metrics)
    except ImportError:
        html_text = build_fallback_html(trajectory_files, metrics)

    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(html_text, encoding="utf-8")
    print(f"wrote dashboard to {args.output_html}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
