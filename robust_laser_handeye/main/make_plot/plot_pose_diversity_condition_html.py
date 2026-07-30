#!/usr/bin/env python3
"""Generate Single/Three interactive HTML for every pose-diversity level."""

from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path
from typing import Sequence

from main.make_plot import plot_random_single_three_pose_html as pose_html


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--levels",
        nargs="+",
        default=("restricted", "moderate", "wide"),
    )
    parser.add_argument("--total-scans", type=int, default=108)
    parser.add_argument("--trial-index", type=int, default=0)
    parser.add_argument("--plane-half-size-mm", type=float, default=72.0)
    parser.add_argument("--three-plane-spacing-mm", type=float, default=320.0)
    parser.add_argument("--target-spread-radius-mm", type=float, default=38.0)
    parser.add_argument("--max-orientation-axes", type=int, default=18)
    parser.add_argument(
        "--full-azimuth-presentation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Default false keeps each run_pose_diversity_comparison pose "
            "condition faithful."
        ),
    )
    parser.add_argument(
        "--plotly-js",
        choices=("inline", "cdn"),
        default="inline",
    )
    return parser.parse_args(argv)


def _condition_config(path: Path) -> dict:
    manifest_path = path / "comparison_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"incomplete comparison manifest: {manifest_path}")
    return manifest["config"]


def _summary(config: dict) -> str:
    tilt = config["view_tilt_range_deg"]
    azimuth = config["view_azimuth_range_deg"]
    roll = config["sensor_roll_range_deg"]
    return (
        f"tilt {tilt[0]:g}°–{tilt[1]:g}° · "
        f"azimuth {azimuth[0]:g}°–{azimuth[1]:g}° · "
        f"roll {roll[0]:g}°–{roll[1]:g}°"
    )


def _write_suite_index(
    output: Path,
    *,
    entries: list[tuple[str, str]],
    total_scans: int,
    trial_index: int,
) -> None:
    cards = "\n".join(
        f"""
        <a class="card" href="{escape(level)}/index.html">
          <strong>{escape(level.title())}</strong>
          <span>{escape(summary)}</span>
          <small>Single + Three · N={total_scans}</small>
        </a>
        """
        for level, summary in entries
    )
    output.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Pose-diversity 3D conditions</title>
  <style>
    body {{ margin: 0; background: #f2f4f7; color: #182230;
      font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
    main {{ max-width: 1080px; margin: 0 auto; padding: 52px 24px; }}
    h1 {{ margin: 0 0 9px; font-size: 30px; }}
    p {{ margin: 0; color: #667085; line-height: 1.55; }}
    .grid {{ display: grid; grid-template-columns: repeat(3, 1fr);
      gap: 18px; margin-top: 30px; }}
    .card {{ display: flex; min-height: 176px; flex-direction: column;
      padding: 25px; background: white; color: #182230;
      text-decoration: none; border: 1px solid #dfe3e8; border-radius: 15px;
      box-shadow: 0 8px 24px rgba(16,24,40,.07); }}
    .card:hover {{ border-color: #4c78a8; transform: translateY(-2px); }}
    strong {{ font-size: 21px; margin-bottom: 13px; }}
    span {{ color: #475467; font-size: 14px; line-height: 1.55; }}
    small {{ margin-top: auto; padding-top: 18px; color: #98a2b3; }}
    .note {{ margin-top: 24px; padding: 14px 16px; border-radius: 10px;
      background: #eaf2fb; color: #344054; font-size: 13px; }}
    @media (max-width: 760px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body><main>
  <h1>Pose-diversity condition visualizations</h1>
  <p>Conditions from run_pose_diversity_comparison.sh ·
    trial {trial_index:06d}</p>
  <div class="grid">{cards}</div>
  <div class="note">
    Orientation ranges are retained for Restricted, Moderate, and Wide.
    Target centers are spread over each surface for readability, and the
    Three-plane view uses a display-only table/left-wall/back-wall layout.
    Every 3D page includes Experiment condition and Diverse display modes.
  </div>
</main></body>
</html>
""",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.total_scans <= 0:
        raise SystemExit("--total-scans must be positive")
    if args.trial_index < 0:
        raise SystemExit("--trial-index must be non-negative")
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    entries: list[tuple[str, str]] = []
    for level in args.levels:
        condition_root = dataset_root / f"{level}_N{args.total_scans}"
        config = _condition_config(condition_root)
        if int(config["total_scans"]) != args.total_scans:
            raise ValueError(
                f"{condition_root} scan count does not match "
                f"{args.total_scans}"
            )
        viewer_args = [
            "--dataset-root",
            str(condition_root),
            "--output-dir",
            str(output_dir / level),
            "--trial-index",
            str(args.trial_index),
            "--plane-half-size-mm",
            str(args.plane_half_size_mm),
            "--three-plane-spacing-mm",
            str(args.three_plane_spacing_mm),
            "--target-spread-radius-mm",
            str(args.target_spread_radius_mm),
            "--max-orientation-axes",
            str(args.max_orientation_axes),
            "--plotly-js",
            args.plotly_js,
        ]
        viewer_args.append(
            "--full-azimuth-presentation"
            if args.full_azimuth_presentation
            else "--no-full-azimuth-presentation"
        )
        pose_html.main(viewer_args)
        entries.append((level, _summary(config)))

    _write_suite_index(
        output_dir / "index.html",
        entries=entries,
        total_scans=args.total_scans,
        trial_index=args.trial_index,
    )
    print(f"Saved pose-diversity HTML suite: {output_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
