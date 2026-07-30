#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
import time

import numpy as np

from laser_handeye.tan2025 import (
    AlternatingPlaneRefiner,
    CalibrationPipeline,
    NonlinearPlaneRefiner,
    PaperSimulationConfig,
    SensorNoiseConfig,
    Tan2025ClosedFormEstimator,
    ThreeWayComparisonConfig,
    analytic_model_dict,
    flatten_three_way_rows,
    run_three_way_monte_carlo,
    summarize_paired_differences,
    summarize_three_way_methods,
    three_way_trials_detail_dict,
)
from laser_handeye.tan2025.experiments import (
    run_gaussian_noise_sweep,
    run_paper_trial,
    run_pose_count_sweep,
    simulation_config_dict,
    summarize_trials,
    trial_detail_dict,
    write_csv,
    write_json,
)


def _add_geometry_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--translation-poses", type=int, default=36)
    parser.add_argument("--composite-poses", type=int, default=30)
    parser.add_argument("--profile-points", type=int, default=640)
    parser.add_argument("--tilt-span-deg", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=7)


def _add_noise_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--noise-mode",
        choices=["none", "gaussian_xz", "uniform_xz", "constant_range_bias"],
        default="none",
    )
    parser.add_argument("--noise-mm", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analytic reproduction of Tan et al., IEEE TIM 2025, "
            "doi:10.1109/TIM.2025.3551450"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser("single", help="run one seeded calibration")
    _add_geometry_options(single)
    _add_noise_options(single)
    single.add_argument(
        "--refiner",
        choices=["none", "alternating", "nonlinear"],
        default="none",
        help="optional method appended after the Tan closed-form estimate",
    )
    single.add_argument("--max-iter", type=int, default=30)
    single.add_argument("--max-nfev", type=int, default=100)
    single.add_argument(
        "--output-dir", type=Path, default=Path("runs/tan2025_numpy/single")
    )

    noise = subparsers.add_parser(
        "noise-sweep", help="reproduce the Table III-style Gaussian sweep"
    )
    _add_geometry_options(noise)
    noise.set_defaults(translation_poses=48, composite_poses=48)
    noise.add_argument(
        "--sigma-mm",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    )

    absolute = subparsers.add_parser(
        "bias-sweep",
        help=(
            "Table I/II-style sweep using the explicit constant range-bias "
            "interpretation of the paper's ambiguous absolute noise"
        ),
    )
    _add_geometry_options(absolute)
    absolute.set_defaults(translation_poses=36, composite_poses=48)
    absolute.add_argument(
        "--bias-mm", type=float, nargs="+", default=[0.05, 0.10, 0.15, 0.20]
    )
    absolute.add_argument(
        "--composite-counts", type=int, nargs="+", default=[12, 24, 36, 48]
    )
    absolute.add_argument("--trials-per-cell", type=int, default=1)
    absolute.add_argument(
        "--output-dir", type=Path, default=Path("runs/tan2025_numpy/bias_sweep")
    )
    noise.add_argument("--trials", type=int, default=40)
    noise.add_argument(
        "--output-dir", type=Path, default=Path("runs/tan2025_numpy/noise_sweep")
    )

    counts = subparsers.add_parser(
        "count-sweep", help="reproduce the Fig. 6/7-style pose-count sweep"
    )
    _add_geometry_options(counts)
    _add_noise_options(counts)
    counts.set_defaults(noise_mode="gaussian_xz", noise_mm=0.03)
    counts.add_argument(
        "--translation-counts",
        type=int,
        nargs="+",
        default=list(range(12, 49, 6)),
    )
    counts.add_argument(
        "--composite-counts",
        type=int,
        nargs="+",
        default=list(range(9, 49, 3)),
    )
    counts.add_argument("--trials-per-cell", type=int, default=1)
    counts.add_argument(
        "--output-dir", type=Path, default=Path("runs/tan2025_numpy/count_sweep")
    )

    comparison = subparsers.add_parser(
        "compare-monte-carlo",
        help=(
            "compare Tan, Tan-initialized alternating, and alternating-only "
            "on one shared Tan dataset per trial"
        ),
    )
    _add_geometry_options(comparison)
    _add_noise_options(comparison)
    comparison.set_defaults(noise_mode="gaussian_xz", noise_mm=0.1)
    comparison.add_argument("--trials", type=int, default=100)
    comparison.add_argument(
        "--iterative-only-init",
        choices=["relative_gt", "carlson_gt", "identity"],
        default="identity",
        help=(
            "identity is the no-GT comparison default; relative_gt matches "
            "the old benchmark's GT-assisted +/-10%% initializer"
        ),
    )
    comparison.add_argument("--relative-offset", type=float, default=0.1)
    comparison.add_argument(
        "--carlson-translation-range-mm", type=float, default=200.0
    )
    comparison.add_argument("--carlson-angle-range-deg", type=float, default=30.0)
    comparison.add_argument("--max-iter", type=int, default=100)
    comparison.add_argument("--tol", type=float, default=1e-9)
    comparison.add_argument(
        "--plane-offset-mode",
        choices=["joint", "difference", "fitted"],
        default="joint",
    )
    comparison.add_argument(
        "--max-translation-offset-condition", type=float, default=1e6
    )
    comparison.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/tan2025_numpy/three_way_comparison"),
    )
    return parser.parse_args()


def _make_config(args: argparse.Namespace) -> PaperSimulationConfig:
    noise_mode = getattr(args, "noise_mode", "none")
    noise_mm = float(getattr(args, "noise_mm", 0.0))
    dropout = float(getattr(args, "dropout", 0.0))
    return PaperSimulationConfig(
        num_translation_poses=int(args.translation_poses),
        num_composite_poses=int(args.composite_poses),
        num_profile_points=int(args.profile_points),
        composite_tilt_span_deg=float(args.tilt_span_deg),
        sensor_noise=SensorNoiseConfig(
            mode=noise_mode,
            magnitude_mm=noise_mm,
            dropout_probability=dropout,
        ),
    )


def _plot_noise_summary(summary: list[dict[str, object]], path: Path) -> None:
    import matplotlib.pyplot as plt

    sigma = np.asarray([row["noise_magnitude_mm"] for row in summary], dtype=float)
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    panels = (
        ("rotation_euler_l1_error_deg", "Euler L1 error [deg]"),
        ("translation_error_mm", "Translation error [mm]"),
        ("reconstruction_mean_error_mm", "Reconstruction error [mm]"),
    )
    for axis, (field, label) in zip(axes, panels):
        mean = np.asarray([row[f"{field}_mean"] for row in summary], dtype=float)
        std = np.asarray([row[f"{field}_std"] for row in summary], dtype=float)
        axis.errorbar(sigma, mean, yerr=std, marker="o", capsize=3)
        axis.set_xlabel("Gaussian sigma [mm]")
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.3)
    figure.suptitle("Tan 2025 analytic Gaussian-noise sweep")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_count_summary(summary: list[dict[str, object]], path: Path) -> None:
    import matplotlib.pyplot as plt

    translation = sorted({int(row["num_translation_poses"]) for row in summary})
    composite = sorted({int(row["num_composite_poses"]) for row in summary})
    lookup = {
        (int(row["num_translation_poses"]), int(row["num_composite_poses"])): row
        for row in summary
    }
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.5))
    for axis, field, title in (
        (axes[0], "rotation_euler_l1_error_deg_mean", "Euler L1 error [deg]"),
        (axes[1], "translation_error_mm_mean", "Translation error [mm]"),
    ):
        values = np.asarray(
            [[lookup[(t, c)][field] for c in composite] for t in translation],
            dtype=float,
        )
        image = axis.imshow(values, origin="lower", aspect="auto")
        axis.set_xticks(range(len(composite)), composite, rotation=45)
        axis.set_yticks(range(len(translation)), translation)
        axis.set_xlabel("Composite poses")
        axis.set_ylabel("Translation poses")
        axis.set_title(title)
        figure.colorbar(image, ax=axis)
    figure.suptitle("Tan 2025 analytic pose-count sweep")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_bias_summary(summary: list[dict[str, object]], path: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    composite_counts = sorted(
        {int(row["num_composite_poses"]) for row in summary}
    )
    for composite_count in composite_counts:
        rows = sorted(
            (
                row
                for row in summary
                if int(row["num_composite_poses"]) == composite_count
            ),
            key=lambda row: float(row["noise_magnitude_mm"]),
        )
        bias = [float(row["noise_magnitude_mm"]) for row in rows]
        axes[0].plot(
            bias,
            [float(row["rotation_euler_l1_error_deg_mean"]) for row in rows],
            marker="o",
            label=f"{composite_count} poses",
        )
        axes[1].plot(
            bias,
            [float(row["translation_error_mm_mean"]) for row in rows],
            marker="o",
            label=f"{composite_count} poses",
        )
    axes[0].set_ylabel("Euler L1 error [deg]")
    axes[1].set_ylabel("Translation error [mm]")
    for axis in axes:
        axis.set_xlabel("Constant range bias [mm]")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    figure.suptitle("Tan 2025 explicit constant-range-bias interpretation")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_three_way_comparison(
    rows: list[dict[str, object]],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    methods = (
        "tan_closed_form",
        "tan_then_alternating",
        "alternating_only",
    )
    labels = ("Tan", "Tan -> alternating", "Alternating only")
    panels = (
        ("rotation_geodesic_error_deg", "Rotation error [deg]"),
        ("translation_error_mm", "Translation error [mm]"),
        ("reconstruction_mean_error_mm", "Reconstruction error [mm]"),
        ("iterations", "Alternating iterations"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    for axis, (field, title) in zip(axes.ravel(), panels):
        values = []
        for method in methods:
            group = [
                float(row[field])
                for row in rows
                if row["method"] == method
                and bool(row["completed"])
                and row[field] is not None
                and np.isfinite(float(row[field]))
            ]
            values.append(group)
        axis.boxplot(values, labels=labels, showmeans=True)
        axis.set_ylabel(title)
        axis.tick_params(axis="x", rotation=12)
        axis.grid(True, axis="y", alpha=0.3)
    figure.suptitle("Shared-data Tan 2025 three-way Monte Carlo")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run_single(args: argparse.Namespace) -> None:
    config = _make_config(args)
    estimator = Tan2025ClosedFormEstimator()
    pipeline = None
    if args.refiner == "alternating":
        pipeline = CalibrationPipeline(
            estimator,
            [AlternatingPlaneRefiner(max_iter=args.max_iter)],
        )
    elif args.refiner == "nonlinear":
        pipeline = CalibrationPipeline(
            estimator,
            [NonlinearPlaneRefiner(max_nfev=args.max_nfev)],
        )

    trial, estimate, pipeline_result = run_paper_trial(
        config,
        seed=args.seed,
        estimator=None if pipeline is not None else estimator,
        pipeline=pipeline,
    )
    detail = trial_detail_dict(trial, estimate, pipeline_result)
    detail["simulation_config"] = simulation_config_dict(config)
    detail["parameter_provenance"] = {
        "paper_disclosed": {
            "T_ef_s_true": config.T_ef_s_true.tolist(),
            "plane_normal_base": config.plane_normal_base.tolist(),
            "profile_points": config.num_profile_points,
            "scan_angle_deg": config.scan_angle_deg,
            "sensor_z_range_mm": [config.sensor_z_min_mm, config.sensor_z_max_mm],
        },
        "undisclosed_pose_distribution_choice": {
            "plane_center_base_mm": list(config.plane_center_base_mm),
            "composite_tilt_span_deg": config.composite_tilt_span_deg,
            "composite_roll_span_deg": config.composite_roll_span_deg,
        },
        "noise_definition": {
            "mode": config.sensor_noise.mode,
            "magnitude_mm": config.sensor_noise.magnitude_mm,
        },
        "analytic_model": analytic_model_dict(config),
    }
    write_json(args.output_dir / "result.json", detail)

    row = trial.to_row()
    print(f"stage: {row['stage_name']}")
    print(
        "errors: "
        f"normal={row['plane_normal_error_deg']:.6g} deg, "
        f"rotation={row['rotation_geodesic_error_deg']:.6g} deg "
        f"(Euler L1={row['rotation_euler_l1_error_deg']:.6g} deg), "
        f"translation={row['translation_error_mm']:.6g} mm"
    )
    print(
        "reconstruction: "
        f"mean={row['reconstruction_mean_error_mm']:.6g} mm, "
        f"MPDE={row['self_fitted_mpde_mm']:.6g} mm"
    )
    print(
        "observability: "
        f"normal={row['normal_rank']}/3, "
        f"rotation={row['rotation_rank']}/6, "
        f"translation={row['translation_rank']}/3"
    )
    print(f"saved {args.output_dir / 'result.json'}")


def run_noise(args: argparse.Namespace) -> None:
    config = _make_config(args)
    started = time.perf_counter()
    trials, summary = run_gaussian_noise_sweep(
        config,
        sigma_values_mm=args.sigma_mm,
        trials_per_sigma=args.trials,
        seed=args.seed,
    )
    write_csv(args.output_dir / "trials.csv", [item.to_row() for item in trials])
    write_csv(args.output_dir / "summary.csv", summary)
    write_json(
        args.output_dir / "metadata.json",
        {
            "experiment": "paper_table_iii_style_gaussian_noise_sweep",
            "gaussian_definition": "independent N(0, sigma^2) on sensor X and Z",
            "paper_pose_distribution_reproducible": False,
            "reason": "pose ranges, distribution, scene, and seed are not published",
            "base_seed": args.seed,
            "trials_per_sigma": args.trials,
            "simulation_config": simulation_config_dict(config),
            "analytic_model": analytic_model_dict(config),
            "sigma_values_mm": [float(value) for value in args.sigma_mm],
            "elapsed_s": time.perf_counter() - started,
        },
    )
    _plot_noise_summary(summary, args.output_dir / "noise_sweep.png")
    print(
        f"completed {len(trials)} trials in {time.perf_counter() - started:.2f} s; "
        f"saved {args.output_dir}"
    )


def run_counts(args: argparse.Namespace) -> None:
    config = _make_config(args)
    generated_config = replace(
        config,
        num_translation_poses=max(args.translation_counts),
        num_composite_poses=max(args.composite_counts),
    )
    started = time.perf_counter()
    trials, summary = run_pose_count_sweep(
        config,
        translation_counts=args.translation_counts,
        composite_counts=args.composite_counts,
        seed=args.seed,
        trials_per_cell=args.trials_per_cell,
    )
    write_csv(args.output_dir / "trials.csv", [item.to_row() for item in trials])
    write_csv(args.output_dir / "summary.csv", summary)
    write_json(
        args.output_dir / "metadata.json",
        {
            "experiment": "paper_fig_6_7_style_pose_count_sweep",
            "base_seed": args.seed,
            "trials_per_cell": args.trials_per_cell,
            "noise_definition": {
                "mode": config.sensor_noise.mode,
                "magnitude_mm": config.sensor_noise.magnitude_mm,
            },
            "paper_pose_distribution_reproducible": False,
            "reason": "pose ranges, distribution, scene, and seed are not published",
            "simulation_config": simulation_config_dict(generated_config),
            "analytic_model": analytic_model_dict(generated_config),
            "translation_counts": [int(value) for value in args.translation_counts],
            "composite_counts": [int(value) for value in args.composite_counts],
            "elapsed_s": time.perf_counter() - started,
        },
    )
    _plot_count_summary(summary, args.output_dir / "count_sweep.png")
    print(
        f"completed {len(trials)} trials in {time.perf_counter() - started:.2f} s; "
        f"saved {args.output_dir}"
    )


def run_bias(args: argparse.Namespace) -> None:
    base_config = _make_config(args)
    started = time.perf_counter()
    all_trials = []
    for bias in args.bias_mm:
        config = replace(
            base_config,
            sensor_noise=SensorNoiseConfig(
                mode="constant_range_bias",
                magnitude_mm=float(bias),
            ),
        )
        trials, _ = run_pose_count_sweep(
            config,
            translation_counts=[args.translation_poses],
            composite_counts=args.composite_counts,
            seed=args.seed,
            trials_per_cell=args.trials_per_cell,
        )
        all_trials.extend(trials)
    summary = summarize_trials(
        all_trials,
        group_fields=(
            "noise_magnitude_mm",
            "num_translation_poses",
            "num_composite_poses",
        ),
    )
    write_csv(
        args.output_dir / "trials.csv", [item.to_row() for item in all_trials]
    )
    write_csv(args.output_dir / "summary.csv", summary)
    write_json(
        args.output_dir / "metadata.json",
        {
            "experiment": "paper_table_i_ii_style_absolute_noise_sweep",
            "absolute_noise_interpretation": (
                "one positive constant bias is added to every measured ray range"
            ),
            "paper_definition_available": False,
            "base_seed": args.seed,
            "trials_per_cell": args.trials_per_cell,
            "simulation_config": simulation_config_dict(base_config),
            "analytic_model": analytic_model_dict(base_config),
            "bias_values_mm": [float(value) for value in args.bias_mm],
            "composite_counts": [int(value) for value in args.composite_counts],
            "elapsed_s": time.perf_counter() - started,
        },
    )
    _plot_bias_summary(summary, args.output_dir / "bias_sweep.png")
    print(
        f"completed {len(all_trials)} trials in "
        f"{time.perf_counter() - started:.2f} s; saved {args.output_dir}"
    )


def run_three_way_comparison(args: argparse.Namespace) -> None:
    simulation_config = _make_config(args)
    comparison_config = ThreeWayComparisonConfig(
        trials=args.trials,
        seed=args.seed,
        iterative_only_init=args.iterative_only_init,
        relative_offset=args.relative_offset,
        carlson_translation_range_mm=args.carlson_translation_range_mm,
        carlson_angle_range_deg=args.carlson_angle_range_deg,
        max_iter=args.max_iter,
        tol=args.tol,
        plane_offset_mode=args.plane_offset_mode,
        max_translation_offset_condition=args.max_translation_offset_condition,
    )
    started = time.perf_counter()
    progress_step = max(1, comparison_config.trials // 10)

    def report_progress(completed: int, total: int) -> None:
        if completed == total or completed % progress_step == 0:
            print(f"completed trials: {completed}/{total}", flush=True)

    trials = run_three_way_monte_carlo(
        simulation_config,
        comparison_config,
        progress=report_progress,
    )
    rows = flatten_three_way_rows(trials)
    method_summary = summarize_three_way_methods(trials)
    paired_summary = summarize_paired_differences(
        trials,
        bootstrap_seed=comparison_config.seed,
    )

    write_csv(args.output_dir / "trials.csv", rows)
    write_csv(args.output_dir / "method_summary.csv", method_summary)
    write_csv(args.output_dir / "paired_differences.csv", paired_summary)
    write_json(
        args.output_dir / "trials.json",
        three_way_trials_detail_dict(trials),
    )
    elapsed = time.perf_counter() - started
    write_json(
        args.output_dir / "metadata.json",
        {
            "experiment": "shared_tan2025_three_way_monte_carlo",
            "simulation_config": simulation_config_dict(simulation_config),
            "analytic_model": analytic_model_dict(simulation_config),
            "comparison_config": asdict(comparison_config),
            "iterative_only_initialization": {
                "source": comparison_config.iterative_only_init,
                "uses_ground_truth": (
                    comparison_config.iterative_init_uses_ground_truth
                ),
                "note": (
                    "relative_gt/carlson_gt are synthetic oracle initializers; "
                    "identity does not use the simulated ground truth"
                ),
            },
            "shared_data_contract": (
                "each trial generates one simulation; all three methods consume "
                "the same ordered LaserScan objects and dataset SHA-256"
            ),
            "method_definitions": {
                "tan_closed_form": "Tan equations (17)-(54), no iteration",
                "tan_then_alternating": (
                    "the same Tan result followed by the joint single-plane "
                    "alternating solver"
                ),
                "alternating_only": (
                    "the same alternating solver and data with only the selected "
                    "generic initialization"
                ),
            },
            "timing_definition": (
                "tan_then_alternating total_elapsed_s includes the shared Tan "
                "initializer time plus refinement_elapsed_s; iterative branch "
                "execution order alternates by trial to reduce order bias"
            ),
            "paired_statistics_population": (
                "completed method pairs, including nonconverged results; "
                "both_converged is reported separately"
            ),
            "elapsed_s": elapsed,
        },
    )
    _plot_three_way_comparison(rows, args.output_dir / "comparison.png")

    for summary in method_summary:
        def format_mean(field: str) -> str:
            value = summary[field]
            return "n/a" if value is None else f"{float(value):.6g}"

        print(
            f"{summary['method']}: "
            f"completed={summary['completed']}/{summary['trials_requested']}, "
            f"converged={summary['converged']}/{summary['trials_requested']}, "
            "rotation_mean="
            f"{format_mean('rotation_geodesic_error_deg_mean')} deg, "
            "translation_mean="
            f"{format_mean('translation_error_mm_mean')} mm, "
            f"iterations_mean={format_mean('iterations_mean')}"
        )
    print(f"completed comparison in {elapsed:.2f} s; saved {args.output_dir}")


def main() -> None:
    args = parse_args()
    if args.command == "single":
        run_single(args)
    elif args.command == "noise-sweep":
        run_noise(args)
    elif args.command == "count-sweep":
        run_counts(args)
    elif args.command == "bias-sweep":
        run_bias(args)
    elif args.command == "compare-monte-carlo":
        run_three_way_comparison(args)
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
