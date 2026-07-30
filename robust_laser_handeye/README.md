# Robust 2D Laser Sensor Hand-Eye Calibration

> The new Tan et al. (IEEE TIM 2025) three-step closed-form reproduction is
> isolated in `laser_handeye/tan2025/`; setup, limitations, and commands are in
> [`../paper/README.md`](../paper/README.md). The material below documents the
> existing Sharifzadeh-style alternating single-plane implementation.

Reusable noise-free acquisitions for translation/composite, single-plane
circular, and three-plane experiments are generated with
`examples/generate_calibration_dataset.py`. The portable manifest/NPZ contract
and real-robot handoff are documented in
[`laser_handeye/calibration_dataset/README.md`](laser_handeye/calibration_dataset/README.md).

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

For `--plane-offset-mode joint`, `difference`, and `fitted`, `--tol` controls
early termination of the alternating linear iteration. After each iteration
the solver computes

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
options work after a joint, difference, or fitted linear solve, although the
comparison commands below intentionally refine the legacy fitted result.

### Interpreting the convergence plots

Both single-plane and three-plane benchmark commands save exactly five default
figures:

| File | Contents |
|---|---|
| `translation_error_by_iteration.png` | Translation norm error [mm] by solver step |
| `rotation_error_by_iteration.png` | Geodesic rotation error [deg] by solver step |
| `fitted_plane_rms_by_iteration.png` | RMS distance to the fitted plane(s) [mm] by solver step |
| `final_translation_rotation_error_boxplot.png` | Before/after translation and rotation error distributions |
| `final_error_sensor_axis_components.png` | Final signed translation and rotation-vector errors along true sensor X/Z/Y |

Both histories record the same solver states: state 0 is the initial estimate,
followed by one state per completed linear update. When nonlinear refinement
is enabled, its final transform and recomputed self-fitted RMS are appended as
the final solver state. Thus matching x coordinates now refer to the same
hand-eye estimate in both plots.

### Saved CSV artifacts

Every single-plane and three-plane benchmark run writes four CSV files.
For `--csv results/example.csv`, the defaults are:

| File | Contents |
|---|---|
| `results/example.csv` | One complete row per generated trial |
| `results/example_iterations.csv` | Long format: one row per trial and solver step |
| `results/example_summary.csv` | One aggregate row for the execution |
| `results/example_failures.csv` | One row per failed trial; header-only when no trial failed |

Override the companion paths with `--summary-csv` and `--failures-csv`. The
summary remains available even if every trial fails. It records requested,
completed and failed counts; convergence and success rates; the complete CLI
configuration and elapsed time; and aggregate statistics for translation,
rotation, component errors, iterations, condition number, plane RMS,
Frobenius error and nonlinear refinement.

The main error metrics include `min`, `p05`, `p25`, `median`, `mean`, `std`,
`rmse`, `p75`, `p95`, `p99` and `max`. Min/max columns also include the
corresponding `system_idx`, while signed component errors additionally report
the largest absolute error and its trial ID. The failure CSV records
`system_idx`, exception type and exception message.

The three-plane noise sweep keeps one summary row per noise level and now also
includes translation/rotation min, max, p99 and extreme trial IDs, plus
iteration and condition-number ranges. Its failure log additionally records
the noise level of each failed trial. It also writes
`paper_style_noise_sweep_gt_errors.png`, a two-panel translation/rotation view
corresponding to one fixed-scan-count slice of Fig. 5 in `single_plane.pdf`.
It is deliberately not drawn as the paper's 3-D surface because the current
sweep varies noise while holding `--poses-per-plane` fixed.

Translation and rotation-vector errors are rotated into the true sensor frame
and stored as `err_t_sensor_{x,y,z}_mm` and
`err_r_sensor_{x,y,z}_deg`. The iteration CSV exposes translation error,
rotation error, plane RMS, Frobenius error and available gauge components as
ordinary numeric columns, so no JSON parsing is needed for follow-up analysis.
Single-plane optimal trials also store fitted/GT plane offsets, their signed
error, the mean fitted-normal/sensor-Z dot product, and aligned per-iteration
histories in `iter_plane_offset_estimate_mm`, `iter_plane_offset_error_mm`,
`iter_err_t_sensor_z_mm`, and `iter_normal_sensor_z_dot_mean`.

To inspect an optimal single-plane scene interactively in 3-D, pass an HTML
path to `--debug-scene-plot`:

```bash
PYTHONPATH=. python examples/run_single_plane_optimal_benchmark.py \
  --systems 1 \
  --seed 7 \
  --reference-scans 24 \
  --debug-scene-plot results/single_plane_optimal_scene.html \
  --no-plots
```

The HTML contains the GT plane and normal, reconstructed laser profiles and
base-frame axes. It supports mouse rotation, zoom, pan, hover coordinates and
legend-based trace toggling. The representative scene uses `--debug-scene-seed`
when provided, otherwise the main `--seed`. Supplying an image suffix such as
`.png` keeps the existing static scene plot.

When regular plots are enabled, the optimal benchmark also writes
`translation_error_direction_3d.html` inside `--plot-dir`. It displays one
interactive translation-error vector per trial in the true sensor X/Y/Z frame,
colors endpoints by total error magnitude, and overlays the `+/-` sensor-Z
gauge axis with translucent 15-degree gauge cones. Hovering an endpoint shows
its `system_idx`, signed components, norm and angle to the gauge axis.

## Bootstrap-planned single-plane experiment

The original optimal simulator creates each requested sensor pose using the GT
plane and converts it to a flange pose using the GT hand-eye. To model a real
setup in which neither is available for exact trajectory planning, enable
`--trajectory-planning bootstrap`:

1. Acquire four preliminary parallel profiles in a short manually aimed raster.
2. Reconstruct those profiles in the robot base frame using the initial
   hand-eye and fit a plane with PCA.
3. Build the requested `(d, theta, beta)` circular trajectory on that estimated
   plane and convert every sensor command to a flange command with the same
   initial hand-eye.
4. Simulate measurements using the physical GT plane and GT hand-eye, then run
   the existing unknown-plane calibration unchanged.

The four bootstrap profiles locate the plane only; they are not added to the
calibration scan set, so `n_scans` remains directly comparable to the original
benchmark. In the simulator, their poses represent an operator manually aiming
at the board. The GT plane is used only to ensure that these preliminary
profiles hit the physical plane, not to compute the calibrated trajectory.

```bash
PYTHONPATH=. python3 examples/run_single_plane_optimal_benchmark.py \
  --systems 100 \
  --seed 7 \
  --mode unknown \
  --trajectory-planning bootstrap \
  --bootstrap-scans 4 \
  --bootstrap-line-half-length-mm 50 \
  --bootstrap-height-mm 120 \
  --bootstrap-theta-deg 30 \
  --bootstrap-beta-deg 90 \
  --init-mode relative \
  --rel-offset 0.1 \
  --noise-std 0.5 \
  --reference-scans 24 \
  --plane-offset-mode joint \
  --max-iter 30 \
  --tol 1e-9 \
  --csv results/single_plane_bootstrap_planned.csv \
  --verbose
```

The trial CSV records `bootstrap_plane_normal_error_deg`,
`bootstrap_plane_offset_error_mm`, and the mean commanded-versus-actual sensor
pose errors. Large initial hand-eye errors can produce a poor bootstrap plane;
use `--rel-offset` (or the Carlson angle/translation ranges) to match the
quality of the initial calibration available in the real system.

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
  --verbose
```

### Why `joint` and `difference` give the same hand-eye update

For sensor point `i` acquired at flange pose `k`, the reconstructed base-frame
point is

```text
p_b,i = R_be,k (x_i R1 + z_i R3 + t) + t_be,k.
```

The 2-D profile has sensor `y=0`, so the linear unknown vector is
`w = [R1, R3, t]`. With `a_k = n^T R_be,k`, define the point row and right-hand
side

```text
b_i = [x_i a_k, z_i a_k, a_k]
c_i = -n^T t_be,k.
```

The unknown-plane constraint `n^T p_b,i = l` is therefore

```text
b_i w - l = c_i.
```

For one physical plane, stacking all points gives the `joint` least-squares
problem

```text
min_(w,l) ||B w - 1 l - c||^2.                         (1)
```

For fixed `w`, differentiating (1) with respect to `l` gives

```text
l*(w) = mean(B w - c) = mean(B) w - mean(c).
```

Substituting `l*(w)` back into (1) analytically eliminates the offset:

```text
min_w ||(B - 1 mean(B)) w - (c - 1 mean(c))||^2.       (2)
```

Now subtract the constraint for point `j` from the constraint for point `i`:

```text
(b_i - b_j) w = c_i - c_j
<=> n^T (p_b,i - p_b,j) = 0.                           (3)
```

The all-pairs objective from (3) satisfies

```text
sum_(i<j) (residual_i - residual_j)^2
    = N sum_i (residual_i - mean(residual))^2.
```

The constant factor `N` does not change the least-squares minimizer. Therefore
the centered system (2), all point-pair differences (3), and the `joint`
problem (1) return the same `w`, up to floating-point roundoff. `joint` keeps
one explicit offset column per plane; `difference` is the Schur-complement or
variable-eliminated form of the same problem.

The implementation does not create the `N(N-1)/2` point pairs. In
`_build_grouped_difference_system_from_current_T` it instead performs the
following independently for every physical plane:

1. Reconstruct all points with the current hand-eye estimate and fit `n` by
   PCA.
2. Build `B` and `c` from all valid points in all scans of that plane.
3. Compute `B_centered = B - mean(B, axis=0)` and
   `c_centered = c - mean(c)`.
4. Stack the centered plane systems and solve the nine hand-eye variables.
5. Project the recovered rotation to SO(3), then re-solve translation with
   that rotation fixed using the same centered equations.

Use this explicit offset-free formulation with:

```bash
--solver-update-mode simultaneous --plane-offset-mode difference
```

Points from different flange poses must participate in the same plane-level
centering. If differences are formed only between points from the same scan,
`R_be,k`, `t_be,k`, and the hand-eye translation contribution are identical on
both sides; translation cancels and cannot be calibrated.

Offset elimination also does not add pose information. If the fixed-theta
trajectory has a translation/offset gauge, `joint` exposes it in the augmented
`[B, -1]` system while `difference` exposes the corresponding null or
near-null direction in centered `B`. A second incidence magnitude or another
independent pose excitation is still required.

In the checked seed-7, 50-system observable benchmark, `joint` and
`difference` had identical iteration counts. Their maximum differences were
`3.2e-12 mm` in translation error, `1.2e-10 deg` in rotation error, and
`2.4e-15 mm` in final plane RMS.

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
  --verbose
```

`joint` estimates the hand-eye translation and plane offset in the same linear
update. `difference` analytically eliminates that offset and returns the same
hand-eye update. The legacy `fitted` mode first fits the plane offset and then
solves the hand-eye update, so it generally needs a much higher iteration
limit; 3000 is a conservative cap for this single-plane configuration. For
unknown-plane nonlinear refinement, `refit` re-estimates the plane for every
candidate hand-eye transform.

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
median and slightly higher in mean/max).

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

## Strict theta-30 ring followed by Fisher NBV

The separate hybrid experiment starts from the strict 9-line theta=30 grid
(81 profiles, with the original benchmark's three distances and three beta
values), adds one conservative theta=45 Fisher bridge, then selects further
views from the full Fisher candidate bank:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 \
  examples/run_theta30_then_fisher_nbv.py \
  --trials 20 \
  --seed 7 \
  --added-views 5 \
  --noise-std-mm 0.5 \
  --initial-translation-range-mm 200 \
  --initial-angle-range-deg 20 \
  --output-dir results/theta30_then_fisher_nbv_verified
```

In the verified seed-7 run, one added view changes the structural joint rank
from 8 to 9; two added views first put both median errors below 0.1 mm / 0.1
deg; five put all 20 trials below both thresholds. See
`fisher_next_view/HYBRID_THETA30.md` for the complete table and limitations.

The initial 81 poses retain the original simulator's oracle acquisition: they
are constructed from the GT plane and GT hand-eye. Only the added bridge/NBV
poses are commanded from the current estimate. This experiment therefore
measures calibration-stage scan efficiency, not autonomous hardware
acquisition from a 200 mm / 20 deg initial error.

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
