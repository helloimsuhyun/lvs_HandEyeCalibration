#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <Eigen/Core>
#include <flann/flann.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

// Official MEPP2 implementation by Sanchez et al.
// This header includes core.inl at the end.
#include "core.h"

namespace py = pybind11;

namespace {

py::dict estimate_normals(
    py::array_t<float, py::array::c_style | py::array::forcecast> points,
    int k,
    float noise,
    float curvature_radius,
    float unit_scale_to_meter,
    bool orient_to_origin,
    bool verbose)
{
    auto info = points.request();
    if (info.ndim != 2 || info.shape[1] != 3) {
        throw std::invalid_argument("points must have shape (N, 3)");
    }

    const int n = static_cast<int>(info.shape[0]);
    if (n < 4) {
        throw std::invalid_argument("at least 4 points are required");
    }
    if (k < 3) {
        throw std::invalid_argument("k must be >= 3");
    }
    if (k >= n) {
        throw std::invalid_argument("k must be smaller than the number of points");
    }
    if (!(unit_scale_to_meter > 0.0f) || !std::isfinite(unit_scale_to_meter)) {
        throw std::invalid_argument("unit_scale_to_meter must be finite and > 0");
    }
    if (noise < 0.0f || !std::isfinite(noise)) {
        throw std::invalid_argument("noise must be finite and >= 0");
    }
    if (!(curvature_radius > 0.0f) && !std::isinf(curvature_radius)) {
        throw std::invalid_argument("curvature_radius must be > 0 or +inf");
    }

    const float* in_ptr = static_cast<const float*>(info.ptr);

    // The authors' implementation and README use meters.  Converting here
    // also preserves the meaning of the official hard-coded noise floor.
    Eigen::Matrix<float, Eigen::Dynamic, 3> pc(n, 3);
    std::vector<float> flann_storage(static_cast<size_t>(n) * 3u);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < 3; ++j) {
            const float v_m = in_ptr[3 * i + j] * unit_scale_to_meter;
            pc(i, j) = v_m;
            flann_storage[static_cast<size_t>(3 * i + j)] = v_m;
        }
    }

    flann::Matrix<float> dataset(flann_storage.data(), n, 3);
    flann::Index<flann::L2<float>> tree(dataset, flann::KDTreeSingleIndexParams(15));
    tree.buildIndex();

    float noise_m = noise * unit_scale_to_meter;
    noise_m = std::max(noise_m, noise_min);  // official constant from core.h

    float curvature_m = curvature_radius;
    if (std::isfinite(curvature_radius)) {
        curvature_m = curvature_radius * unit_scale_to_meter;
    }

    py::array_t<float> normals({n, 3});
    py::array_t<std::uint8_t> optimized_mask({n});
    py::array_t<std::uint8_t> second_init_mask({n});
    py::array_t<std::uint8_t> nan_mask({n});

    auto nbuf = normals.mutable_unchecked<2>();
    auto ebuf = optimized_mask.mutable_unchecked<1>();
    auto sbuf = second_init_mask.mutable_unchecked<1>();
    auto zbuf = nan_mask.mutable_unchecked<1>();

    int optimized_count = 0;
    int second_init_count = 0;
    int nan_count = 0;

    const auto t0 = std::chrono::high_resolution_clock::now();

    {
        py::gil_scoped_release release;

        for (int i = 0; i < n; ++i) {
            CApp app(&pc, &tree, i, noise_m);
            const Eigen::Vector3f point_ref = app.getPoint();

            app.setParams(div_fact, curvature_m);
            app.selectNeighborsKnn(k);
            app.init1();

            const bool optimize = app.isOnEdge();
            bool used_second = false;

            if (optimize) {
                ++optimized_count;

                bool first = true;
                app.Optimize(first);
                app.OptimizePos(first, thresh_weight);

                app.reinitPoint();
                app.init2();
                first = false;

                if (app.SuspectedOnEdge_) {
                    used_second = true;
                    ++second_init_count;
                    app.Optimize(first);
                    app.OptimizePos(first, thresh_weight);
                }
            }

            Eigen::Vector3f normal = app.finalNormal_;
            bool bad = app.isNan() || !normal.allFinite() || normal.norm() < 1e-12f;

            if (bad) {
                ++nan_count;
                nbuf(i, 0) = std::numeric_limits<float>::quiet_NaN();
                nbuf(i, 1) = std::numeric_limits<float>::quiet_NaN();
                nbuf(i, 2) = std::numeric_limits<float>::quiet_NaN();
            } else {
                normal.normalize();

                // The original MEPP2 wrapper flips normals toward the origin.
                // Harris3D is sign-invariant, so this is optional here.
                if (orient_to_origin && normal.dot(point_ref) > 0.0f) {
                    normal = -normal;
                }

                nbuf(i, 0) = normal.x();
                nbuf(i, 1) = normal.y();
                nbuf(i, 2) = normal.z();
            }

            ebuf(i) = optimize ? 1u : 0u;
            sbuf(i) = used_second ? 1u : 0u;
            zbuf(i) = bad ? 1u : 0u;

            if (verbose && (i % 1000 == 0 || i + 1 == n)) {
                std::cout << "[sanchez_wpca] "
                          << (100.0 * static_cast<double>(i + 1) / static_cast<double>(n))
                          << "%\n";
            }
        }
    }

    const auto t1 = std::chrono::high_resolution_clock::now();
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    py::dict out;
    out["normals"] = std::move(normals);
    out["optimized_mask"] = std::move(optimized_mask);
    out["second_init_mask"] = std::move(second_init_mask);
    out["nan_mask"] = std::move(nan_mask);
    out["optimized_count"] = optimized_count;
    out["second_init_count"] = second_init_count;
    out["nan_count"] = nan_count;
    out["elapsed_ms"] = elapsed_ms;
    out["k"] = k;
    out["noise"] = noise;
    out["curvature_radius"] = curvature_radius;
    out["unit_scale_to_meter"] = unit_scale_to_meter;
    return out;
}

}  // namespace

PYBIND11_MODULE(sanchez_wpca, m)
{
    m.doc() = R"doc(
Python wrapper around the authors' MEPP2 implementation of:
Sanchez et al., Robust normal vector estimation in 3D point clouds
through iterative principal component analysis, ISPRS JPRS 163 (2020).

The wrapper intentionally keeps the official KNN neighborhood and the
original init1 -> edge preselection -> Optimize -> OptimizePos -> init2 ->
second Optimize -> normal selection flow from MEPP2.
)doc";

    m.def(
        "estimate_normals",
        &estimate_normals,
        py::arg("points"),
        py::arg("k") = 50,
        py::arg("noise") = 0.0f,
        py::arg("curvature_radius") = std::numeric_limits<float>::infinity(),
        py::arg("unit_scale_to_meter") = 1e-3f,
        py::arg("orient_to_origin") = false,
        py::arg("verbose") = false,
        R"doc(
Estimate robust normals using the Sanchez/MEPP2 weighted-PCA implementation.

Parameters
----------
points : (N,3) float array
    Input point cloud.
k : int
    Number of nearest neighbours. Must be < N.
noise : float
    Estimated sensor noise standard deviation in the SAME UNIT as points.
curvature_radius : float
    Minimum tolerated smooth curvature radius in the SAME UNIT as points.
    Use +inf for piecewise-planar data.
unit_scale_to_meter : float
    Converts input units to meters before calling the official algorithm.
    1e-3 for millimetres, 1.0 for metres.
orient_to_origin : bool
    If True, reproduce the outer wrapper's orientation-to-origin step.
    False is usually fine for Harris3D because n n^T is sign-invariant.
verbose : bool
    Print progress.

Returns
-------
dict with:
    normals            : (N,3) float32
    optimized_mask     : points that passed the authors' edge preselection
    second_init_mask   : points that used the second initialization
    nan_mask           : invalid outputs
    elapsed_ms         : C++ compute time
)doc");
}
