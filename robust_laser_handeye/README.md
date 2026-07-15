# Robust 2D Laser Sensor Hand-Eye Calibration

This repository reconstructs the simulation in *Robust hand-eye calibration
of 2D laser sensors using a single-plane* and contains matched single-plane and
three-plane benchmarks.

Main modules:

- `simulation.py`: ground-truth systems, circular-pattern poses and profiles
- `calibration.py`: known/unknown-plane linear calibration
- `initialization.py`: relative and Carlson-style initial estimates
- `run_single_plane_optimal_benchmark.py`: main circular-pattern experiment
- `run_three_plane_benchmark.py`: matched three-plane comparison

## Important identifiability result

The strict reduced circular grid contains 9 target lines and
`3 heights x 1 theta x 3 beta = 9` poses per line, for 81 scans. Under the
paper-incidence geometry, every `theta=30 deg` pose has the same plane-normal
component along sensor Z. When the plane offset is unknown, translation along
the sensor optical axis and the plane offset therefore form an exact gauge.
Changing only `d` or `beta`, adding profile points, increasing iterations, or
running nonlinear refinement cannot recover that missing absolute translation.

This explains the large clean-data error from the literal legacy iteration:
with seed 7, 10 Carlson-initialized systems and zero profile noise, the strict
81-scan configuration gives a 53.7973 mm median translation error even though
the median rotation error is numerically zero.

The main benchmark consequently distinguishes two configurations:

| Configuration | Scans | Meaning |
|---|---:|---|
| Strict reduced grid | 81 | Paper-incidence `theta=30`; rank deficient for unknown plane offset |
| Observable extension | 81 + 24 | Adds explicitly counted `theta=60` reference scans |
| Three-plane comparison | 3 x 35 = 105 | Same total number of scans |

The 24 reference scans are an engineering observability extension. They are
not presented as part of the paper's reduced nine-combination grid.

## Iteration and convergence tolerance

For both `--plane-offset-mode joint` and `--plane-offset-mode fitted`, in both
the single-plane and three-plane solvers, `--tol` controls early termination
of the alternating linear iteration. After each iteration the solver computes

```text
delta = ||T_new - T_previous||_F
```

and reports convergence when `delta < tol`. This is the Frobenius norm of the
full 4 x 4 hand-eye transform difference. Rotation-matrix entries are
dimensionless while translation entries are expressed in millimetres, so this
is an implementation-level matrix-change criterion rather than a single
physically homogeneous error unit. Translation changes usually dominate its
magnitude.

The benchmark commands use `--tol 1e-9` for strict and repeatable convergence
comparison. Useful settings are:

| Setting | Meaning | Recommended use |
|---|---|---|
| `--tol 1e-9` | Stop below a very small transform update | Accuracy and convergence comparisons |
| `--tol 1e-6` | Stop earlier at a practical transform update | Faster large-scale exploratory runs |
| `--tol -1` | Disable early stopping and run exactly `max_iter` iterations | Fixed-iteration diagnostics |

`--max-iter` is only an upper bound when `tol` is non-negative. The joint
solver normally reaches `1e-9` within about 20 iterations. The legacy fitted
iteration contracts much more slowly, which is why the commands below use 30
iterations for joint, 3000 for single-plane fitted and 500 for three-plane
fitted.

The optional nonlinear refinement has separate stopping criteria and does not
use `--tol`. Configure SciPy nonlinear least squares with `--nonlinear-ftol`,
`--nonlinear-xtol` and `--nonlinear-gtol`; all three default to `1e-10`.
`--nonlinear-max-nfev 200` limits its objective evaluations. These nonlinear
options work after either a joint or fitted linear solve, although the
comparison commands below intentionally refine the legacy fitted result.

## Single-plane optimal comparison

Run the following commands from the repository directory. All three use the
same seed, 100 random systems, 0.5 mm profile noise and 105 scans (the strict
81-scan grid plus 24 second-incidence reference scans). The reference scans
are required because the strict fixed-theta grid is rank deficient when the
plane offset is unknown.

### 1. Joint linear update (recommended)

```bash
PYTHONPATH=. python examples/run_single_plane_optimal_benchmark.py \
  --systems 100 \
  --seed 7 \
  --mode unknown \
  --profile-points 100 \
  --noise-std 0.5 \
  --heights-mm 60 90 120 \
  --theta-deg 30 \
  --beta-deg 60 90 120 \
  --pose-geometry paper_incidence \
  --reference-scans 24 \
  --reference-theta-deg 60 \
  --reference-heights-mm 60 90 120 \
  --reference-beta-deg 60 90 120 \
  --plane-offset-mode joint \
  --init-mode carlson \
  --init-translation-range-mm 200 \
  --init-angle-range-deg 30 \
  --max-iter 30 \
  --tol 1e-9 \
  --csv results/single_plane_optimal_joint.csv \
  --no-plots \
  --verbose
```

### 2. Legacy fit-then-solve update

```bash
PYTHONPATH=. python examples/run_single_plane_optimal_benchmark.py \
  --systems 100 \
  --seed 7 \
  --mode unknown \
  --profile-points 100 \
  --noise-std 0.5 \
  --heights-mm 60 90 120 \
  --theta-deg 30 \
  --beta-deg 60 90 120 \
  --pose-geometry paper_incidence \
  --reference-scans 24 \
  --reference-theta-deg 60 \
  --reference-heights-mm 60 90 120 \
  --reference-beta-deg 60 90 120 \
  --plane-offset-mode fitted \
  --init-mode carlson \
  --init-translation-range-mm 200 \
  --init-angle-range-deg 30 \
  --max-iter 3000 \
  --tol 1e-9 \
  --csv results/single_plane_optimal_fitted.csv \
  --no-plots \
  --verbose
```

### 3. Legacy update followed by nonlinear refinement

```bash
PYTHONPATH=. python examples/run_single_plane_optimal_benchmark.py \
  --systems 100 \
  --seed 7 \
  --mode unknown \
  --profile-points 100 \
  --noise-std 0.5 \
  --heights-mm 60 90 120 \
  --theta-deg 30 \
  --beta-deg 60 90 120 \
  --pose-geometry paper_incidence \
  --reference-scans 24 \
  --reference-theta-deg 60 \
  --reference-heights-mm 60 90 120 \
  --reference-beta-deg 60 90 120 \
  --plane-offset-mode fitted \
  --init-mode carlson \
  --init-translation-range-mm 200 \
  --init-angle-range-deg 30 \
  --max-iter 3000 \
  --tol 1e-9 \
  --nonlinear-refine \
  --nonlinear-plane-mode refit \
  --nonlinear-max-nfev 200 \
  --nonlinear-ftol 1e-10 \
  --nonlinear-xtol 1e-10 \
  --nonlinear-gtol 1e-10 \
  --nonlinear-loss linear \
  --csv results/single_plane_optimal_fitted_nonlinear.csv \
  --no-plots \
  --verbose
```

`joint` estimates the hand-eye translation and plane offset in the same linear
update and rejects unobservable data. The legacy `fitted` mode first fits the
plane offset and then solves the hand-eye update, so it generally needs a much
higher iteration limit; 3000 is a conservative cap for this single-plane
configuration. For unknown-plane nonlinear refinement, `refit` re-estimates
the plane for every candidate hand-eye transform.

The default linear multistart safeguard is not nonlinear refinement. It only
retries six deterministic linear/PCA starts when the first solution's plane
RMS exceeds `max(1 mm, 3*noise_std)`, then selects the lowest-residual linear
result. Disable it with `--no-linear-multistart` for a literal single-start
study.

## Three-plane comparison

All three commands below use the same seed, 100 random systems, 0.5 mm profile
noise and `3 x 35 = 105` scans.

### 1. Joint linear update (recommended)

```bash
PYTHONPATH=. python examples/run_three_plane_benchmark.py \
  --systems 100 \
  --seed 7 \
  --mode unknown \
  --poses-per-plane 35 \
  --profile-points 100 \
  --noise-std 0.5 \
  --plane-offset-mode joint \
  --init-mode carlson \
  --init-translation-range-mm 200 \
  --init-angle-range-deg 30 \
  --max-iter 30 \
  --tol 1e-9 \
  --csv results/three_plane_joint.csv \
  --no-plots \
  --verbose
```

### 2. Legacy fit-then-solve update

```bash
PYTHONPATH=. python examples/run_three_plane_benchmark.py \
  --systems 100 \
  --seed 7 \
  --mode unknown \
  --poses-per-plane 35 \
  --profile-points 100 \
  --noise-std 0.5 \
  --plane-offset-mode fitted \
  --init-mode carlson \
  --init-translation-range-mm 200 \
  --init-angle-range-deg 30 \
  --max-iter 500 \
  --tol 1e-9 \
  --csv results/three_plane_fitted.csv \
  --no-plots \
  --verbose
```

### 3. Legacy update followed by nonlinear refinement

```bash
PYTHONPATH=. python examples/run_three_plane_benchmark.py \
  --systems 100 \
  --seed 7 \
  --mode unknown \
  --poses-per-plane 35 \
  --profile-points 100 \
  --noise-std 0.5 \
  --plane-offset-mode fitted \
  --init-mode carlson \
  --init-translation-range-mm 200 \
  --init-angle-range-deg 30 \
  --max-iter 500 \
  --tol 1e-9 \
  --nonlinear-refine \
  --nonlinear-max-nfev 200 \
  --nonlinear-ftol 1e-10 \
  --nonlinear-xtol 1e-10 \
  --nonlinear-gtol 1e-10 \
  --nonlinear-loss linear \
  --csv results/three_plane_fitted_nonlinear.csv \
  --no-plots \
  --verbose
```

The joint solver normally converges within about 20 iterations, so 30 is a
practical limit. The legacy fitted solver has required roughly 300 iterations
in the checked sweeps, so 500 is used for both legacy comparisons. The
three-plane nonlinear implementation freezes the three planes estimated by
the preceding legacy solve while refining the six-parameter SE(3) hand-eye
transform.

For the two joint commands above, the checked seed-7, 100-system result is:

| Method | Translation median / mean / max (mm) | Rotation median / mean / max (deg) |
|---|---:|---:|
| Single plane, 81+24 | 0.03438 / 0.03727 / 0.10700 | 0.02844 / 0.03026 / 0.07313 |
| Three planes, 3x35 | 0.04248 / 0.04591 / 0.11617 | 0.02852 / 0.03026 / 0.07283 |

The two commands use independent per-trial random streams for ground truth,
scene/noise and initialization. Equal `system_idx` values therefore have the
same true hand-eye and exactly the same initial perturbation in both methods.
The observable single-plane configuration is better in all three translation
statistics; rotation is effectively identical (single is slightly lower in
median and slightly higher in mean/max). In a separate 300-system stress run,
two single-plane trials invoked the linear multistart safeguard and all 300
finished below 0.108 mm translation and 0.081 deg rotation error.

## Strict-grid diagnostic

Use this to verify that the missing degree of freedom is detected instead of
silently reporting an arbitrary translation:

```bash
PYTHONPATH=. python examples/run_single_plane_optimal_benchmark.py \
  --systems 1 \
  --seed 7 \
  --mode unknown \
  --noise-std 0 \
  --heights-mm 60 90 120 \
  --theta-deg 30 \
  --beta-deg 60 90 120 \
  --pose-geometry paper_incidence \
  --reference-scans 0 \
  --plane-offset-mode joint \
  --init-mode carlson \
  --max-iter 30 \
  --debug-diagnostics \
  --no-plots \
  --verbose
```

The command is expected to exit with status 2 after reporting a
`translation/plane-offset observability` failure. For comparison,
`--plane-offset-mode fitted` retains the legacy fit-then-solve iteration and
can converge to a low plane residual while preserving a large arbitrary
translation along the gauge.

To verify both the residual invariance and the direction of the actual
translation error, run the legacy solver only as a diagnostic:

```bash
PYTHONPATH=. python examples/run_single_plane_optimal_benchmark.py \
  --systems 10 \
  --seed 7 \
  --mode unknown \
  --noise-std 0 \
  --heights-mm 60 90 120 \
  --theta-deg 30 \
  --beta-deg 60 90 120 \
  --pose-geometry paper_incidence \
  --reference-scans 0 \
  --plane-offset-mode fitted \
  --init-mode carlson \
  --init-translation-range-mm 200 \
  --init-angle-range-deg 0 \
  --max-iter 30 \
  --tol -1 \
  --no-linear-multistart \
  --debug-gauge-plot fixed_theta_gauge_proof.png \
  --csv fixed_theta_gauge_results.csv \
  --plot-dir fixed_theta_gauge_plots \
  --verbose
```

`fixed_theta_gauge_proof.png` sweeps translation along the true sensor-Z axis
and a transverse sensor-X control direction. The sensor-Z residual remains
constant while the fitted offset follows
`delta_d = -delta_t_z*cos(theta)`. The automatically saved
`translation_error_vs_sensor_z_gauge.png` decomposes every solver error into
components parallel and perpendicular to `R_ef_s_true @ e_z`. The same values
are written to the CSV as `gauge_parallel_error_mm`,
`gauge_perpendicular_error_mm`, `gauge_axis_angle_deg` and
`gauge_parallel_fraction`.

## Geometry and paper-reproduction notes

- `paper_incidence` implements theta as the angle between the plane normal and
  sensor `-Z`, consistent with the paper's theta=0/beta=90 statement and the
  deposited target trajectory. The paper does not print the full vector/sign
  construction, so this convention is explicit in `simulation.py`.
- `observable_dihedral` keeps the repository's older theta convention only as
  a named engineering alternative; it is not the faithful paper-incidence
  interpretation.
- Infeasible paper-incidence pairs, such as theta=0/beta=60, are skipped per
  pose instead of aborting the entire dataset.
- The noisy 100-system benchmark samples plane rotations continuously in the
  paper's appropriate non-near-zero range. The paper's separate Sec. 4.2
  orientation sweep enumerates the 512 triples in
  `{-5,-4,-3,-2,2,3,4,5}^3`; these are different experiments.

## Tests

```bash
PYTHONPATH=. python -m pytest -q tests/test_paper_incidence_observability.py
```

The tests cover the incidence geometry, the strict-grid rank defect, an exact
sensor-Z/plane-offset gauge transformation, recovery after adding a second
theta, non-finite profile points, and per-pose skipping of infeasible angle
pairs.

Sources:

- Paper DOI: <https://doi.org/10.1016/j.rcim.2019.101823>
- Public calibration data: <https://doi.org/10.17028/rd.lboro.7365020>
- Local copy: `Robust hand-eye calibration of 2D laser sensors using a single-plane.pdf`
