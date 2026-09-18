# Sanchez WPCA Python Wrapper

A small pybind11 wrapper around the **authors' MEPP2 implementation** of the
iterative weighted-PCA normal estimator from Sanchez et al., ISPRS JPRS 2020.

This project does **not** copy the MEPP2 algorithm source. `scripts/install.sh`
clones the required official `WeightedPCANormals` subtree and pins it to a
specific MEPP2 commit, then compiles a thin Python binding around `core.h` /
`core.inl`.

## 1. System dependencies (Ubuntu)

```bash
sudo apt update
sudo apt install -y build-essential cmake ninja-build git libeigen3-dev libflann-dev liblz4-dev
```

Activate your Conda environment first, e.g.

```bash
conda activate laser_handeye
```

## 2. Build and install into the active Python environment

```bash
cd sanchez_wpca_wrapper
./scripts/install.sh
```

The script also installs Python-side build dependencies (`pybind11`, `numpy`),
clones the official MEPP2 source, builds the module, and copies
`sanchez_wpca*.so` into the active environment's `site-packages`.

## 3. Verify

```bash
python -c "import sanchez_wpca; print(sanchez_wpca.__doc__)"
python examples/quick_test.py
```

## 4. Basic use for point clouds in millimetres

```python
import numpy as np
import sanchez_wpca

result = sanchez_wpca.estimate_normals(
    np.asarray(points, dtype=np.float32),
    k=50,
    noise=0.0,
    curvature_radius=np.inf,
    unit_scale_to_meter=1e-3,
    orient_to_origin=False,
)

normals = np.asarray(result["normals"])
optimized = np.asarray(result["optimized_mask"], dtype=bool)

print(result["elapsed_ms"])
print(optimized.sum(), "/", len(points))
```

## Parameters

- `k`: KNN count. The authors' README suggests roughly 100 normally, 50 for
  scattered point clouds, and 300 for very noisy point clouds. For a fair
  comparison against a fixed-radius estimator, also test a `k` matched to the
  median number of neighbours inside that radius.
- `noise`: estimated sensor noise standard deviation, in the same unit as
  `points`.
- `curvature_radius`: minimum tolerated smooth-curvature radius, in the same
  unit as `points`; use `np.inf` only for piecewise-planar data.
- `unit_scale_to_meter`: converts your coordinates to the units expected by the
  official implementation. Use `1e-3` for millimetres, `1.0` for metres.
- `orient_to_origin`: reproduces the MEPP2 outer wrapper's final normal flip
  when enabled. It is not necessary for Harris3D because the second moment uses
  `n n^T`, which is invariant to normal sign.

## Returned diagnostics

- `normals`: `(N,3)` estimated normals.
- `optimized_mask`: points that passed the authors' preselection and underwent
  robust iterative optimization.
- `second_init_mask`: points for which the second initialization was used.
- `nan_mask`: invalid results.
- `elapsed_ms`: C++ compute time.

## Important for the current project

Your current CAD/point-cloud code uses **millimetres**, so keep
`unit_scale_to_meter=1e-3`. This matters because the official implementation has
an absolute minimum noise constant defined in its meter-scale code.

For the first comparison, keep the authors' algorithm unchanged. Compare:

1. PCL PCA normal
2. Gaussian weighted PCA normal
3. Sanchez official WPCA normal
4. Harris response using each normal field

Do not add Gaussian spatial weights to Sanchez until the official baseline has
been measured.


## Ubuntu FLANN/LZ4 note

On Ubuntu, `libflann_cpp` uses LZ4 symbols. The wrapper explicitly links `liblz4`; install `liblz4-dev` before building.
