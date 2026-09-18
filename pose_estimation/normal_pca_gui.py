#!/usr/bin/env python3
"""
Local Normal Inspector GUI
==========================

Interactive tool for comparing ordinary PCA, one-shot soft-weighted PCA, and iterative soft-weighted PCA around a selected point.

Main workflow
-------------
1) Generate Smooth plane / Smooth cylinder / Edge / Corner.
2) Click a point in the left 3D geometry view to select the query point.
3) Inspect all neighbors within the PCA radius:
   - neighbor estimated normals as 3D arrows,
   - distance from query,
   - local Gauss-map distribution colored by query distance,
   - normal components / angular deviation versus query distance.

Views
-----
LEFT:
    Whole geometry + GT-normal angular-error heatmap.
    Click a point to select it.

TOP RIGHT:
    Local neighborhood within radius.
    Every neighbor normal is shown as an arrow.
    Query estimated normal and GT normal are emphasized.

BOTTOM CENTER:
    Gauss map of neighbor estimated normals.
    Endpoint color encodes distance to query.

BOTTOM RIGHT:
    Normal components nx, ny, nz versus distance from query.
    A secondary panel shows angular difference from query GT normal.

Run
---
    python local_normal_inspector_gui.py

Dependencies
------------
    numpy scipy matplotlib tkinter

Ubuntu:
    sudo apt install python3-tk
"""

import os
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from scipy.spatial import cKDTree

import matplotlib
if "MPLBACKEND" not in os.environ:
    matplotlib.use("TkAgg")

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)


# ============================================================
# Utility
# ============================================================

def normalize_rows(v):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.clip(n, 1e-12, None)


def angular_deg(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)):
        return np.nan
    c = np.clip(np.dot(a, b), -1.0, 1.0)
    return np.degrees(np.arccos(c))


def orient_normals_to_reference(normals, reference):
    """
    Flip normal signs only for visualization so every normal lies in the
    same hemisphere as `reference`.

    PCA normals are axial quantities: n and -n represent the same local
    tangent plane. This removes visually distracting sign flips without
    changing the estimated plane itself.
    """
    normals = np.asarray(normals, dtype=float).copy()
    reference = np.asarray(reference, dtype=float)

    if np.any(~np.isfinite(reference)):
        return normals

    ref_norm = np.linalg.norm(reference)
    if ref_norm < 1e-12:
        return normals

    reference = reference / ref_norm

    valid = np.all(np.isfinite(normals), axis=1)
    dots = normals[valid] @ reference
    flip = dots < 0.0

    vv = normals[valid]
    vv[flip] *= -1.0
    normals[valid] = vv

    return normals


# ============================================================
# Synthetic geometry
# ============================================================

def make_plane(extent, spacing):
    a = np.arange(-extent, extent + 0.5 * spacing, spacing)
    X, Y = np.meshgrid(a, a)

    P = np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)]
    GT = np.tile([0.0, 0.0, 1.0], (len(P), 1))
    face = np.zeros(len(P), dtype=int)

    return P, GT, face


def make_cylinder(extent, spacing, radius):
    theta_max = min(math.radians(70.0), extent / max(radius, 1e-6))
    dtheta = max(spacing / max(radius, 1e-6), math.radians(1.0))

    theta = np.arange(
        -theta_max,
        theta_max + 0.5 * dtheta,
        dtheta,
    )
    z = np.arange(-extent, extent + 0.5 * spacing, spacing)

    TH, Z = np.meshgrid(theta, z)

    X = radius * np.cos(TH)
    Y = radius * np.sin(TH)

    P = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    GT = normalize_rows(
        np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)]
    )
    face = np.zeros(len(P), dtype=int)

    return P, GT, face


def make_edge(extent, spacing, angle_deg):
    """
    Two half-planes sharing x-axis.

    Face 0:
        z = 0, y > 0
        GT normal = +z

    Face 1:
        Face 0 rotated around x-axis by angle_deg.

    Exact mathematical edge samples are excluded so every point
    has one unambiguous analytic GT normal.
    """
    phi = math.radians(angle_deg)

    x = np.arange(-extent, extent + 0.5 * spacing, spacing)
    u = np.arange(spacing, extent + 0.5 * spacing, spacing)

    X, U = np.meshgrid(x, u)

    A = np.c_[
        X.ravel(),
        U.ravel(),
        np.zeros(X.size),
    ]
    GTA = np.tile([0.0, 0.0, 1.0], (len(A), 1))

    B = np.c_[
        X.ravel(),
        U.ravel() * math.cos(phi),
        U.ravel() * math.sin(phi),
    ]

    nB = np.array([
        0.0,
        -math.sin(phi),
        math.cos(phi),
    ])
    GTB = np.tile(nB, (len(B), 1))

    P = np.vstack([A, B])
    GT = np.vstack([GTA, GTB])

    face = np.r_[
        np.zeros(len(A), dtype=int),
        np.ones(len(B), dtype=int),
    ]

    return P, GT, face


def make_corner(extent, spacing, angle_deg):
    """
    Three surface patches meeting at the origin.

    Exact boundaries are not sampled.
    """
    phi = math.radians(angle_deg)

    a = np.arange(spacing, extent + 0.5 * spacing, spacing)
    U, V = np.meshgrid(a, a)

    u = U.ravel()
    v = V.ravel()

    # Face 0
    A = np.c_[
        u,
        v,
        np.zeros_like(u),
    ]
    GTA = np.tile([0.0, 0.0, 1.0], (len(A), 1))

    # Face 1: rotate about y
    B = np.c_[
        u * math.cos(phi),
        v,
        u * math.sin(phi),
    ]
    nB = np.array([
        math.sin(phi),
        0.0,
        math.cos(phi),
    ])
    GTB = np.tile(nB, (len(B), 1))

    # Face 2: rotate about x
    C = np.c_[
        u,
        v * math.cos(phi),
        v * math.sin(phi),
    ]
    nC = np.array([
        0.0,
        -math.sin(phi),
        math.cos(phi),
    ])
    GTC = np.tile(nC, (len(C), 1))

    P = np.vstack([A, B, C])
    GT = np.vstack([GTA, GTB, GTC])

    face = np.r_[
        np.zeros(len(A), dtype=int),
        np.ones(len(B), dtype=int),
        np.full(len(C), 2, dtype=int),
    ]

    return P, GT, face


# ============================================================
# Ordinary PCA normal estimation
# ============================================================


def weighted_pca_normal(points, weights, reference=None):
    """
    Weighted PCA normal from 3D points.

    `reference` is used only to choose the sign of the eigenvector.
    """
    points = np.asarray(points, dtype=float)
    weights = np.asarray(weights, dtype=float)

    valid = (
        np.all(np.isfinite(points), axis=1)
        & np.isfinite(weights)
        & (weights > 1e-12)
    )

    if np.sum(valid) < 3:
        return np.full(3, np.nan)

    X = points[valid]
    w = weights[valid]

    sw = np.sum(w)
    if sw <= 1e-12:
        return np.full(3, np.nan)

    c = np.sum(X * w[:, None], axis=0) / sw
    Y = X - c
    C = (Y * w[:, None]).T @ Y / sw

    vals, vecs = np.linalg.eigh(C)
    n = vecs[:, 0]

    if reference is not None and np.all(np.isfinite(reference)):
        if np.dot(n, reference) < 0:
            n = -n

    return n / max(np.linalg.norm(n), 1e-12)


def estimate_pca_normals(points, gt_normals, radius, min_neighbors=6):
    """
    Ordinary radius-PCA normals.

    Returns:
        tree, est, err, support
    """
    tree = cKDTree(points)

    N = len(points)
    est = np.full((N, 3), np.nan)
    err = np.full(N, np.nan)
    support = np.zeros(N, dtype=int)

    for i, p in enumerate(points):
        ids = np.asarray(
            tree.query_ball_point(p, radius),
            dtype=int,
        )

        support[i] = len(ids)

        if len(ids) < min_neighbors:
            continue

        X = points[ids]
        c = X.mean(axis=0)
        Y = X - c
        C = (Y.T @ Y) / len(X)

        vals, vecs = np.linalg.eigh(C)
        n = vecs[:, 0]

        # Sign only for analytic visualization/evaluation.
        if np.dot(n, gt_normals[i]) < 0:
            n = -n

        est[i] = n

        dot = np.clip(
            np.dot(n, gt_normals[i]),
            -1.0,
            1.0,
        )
        err[i] = np.degrees(np.arccos(dot))

    return tree, est, err, support


def estimate_oneshot_normals(
    points,
    gt_normals,
    radius,
    sigma_e_deg=15.0,
    sigma_r=0.25,
    spatial_sigma_ratio=0.5,
    min_neighbors=6,
):
    """
    Proposed one-shot soft local-surface weighting.

    Stage 0: ordinary PCA produces rough normals n_i^rough.

    For every query p:
      d_i      = ||q_i - p||
      theta_i  = angle(n_i^rough, n_p^rough)
      kappa_hat = median(theta_i / d_i)
      e_i      = max(0, theta_i - kappa_hat * d_i)

      residual:
          r_i = |(q_i-p)^T n_p^rough|

      weights:
          w_d = exp(-d_i^2 / (2 sigma_d^2))
          w_e = 1 / (1 + (e_i / sigma_e)^2)
          w_r = 1 / (1 + (r_i / sigma_r)^2)

          w_i = w_d * w_e * w_r

    Final normal = ONE weighted PCA only.
    There is no iterative reweighting.

    Notes:
    - theta/e are calculated in radians internally.
    - sigma_d = spatial_sigma_ratio * radius.
    - Neighbor rough normals are sign-aligned to the query rough normal
      before angular differences are calculated.
    """
    tree, rough, _, support = estimate_pca_normals(
        points,
        gt_normals,
        radius,
        min_neighbors=min_neighbors,
    )

    N = len(points)
    est = np.full((N, 3), np.nan)
    err = np.full(N, np.nan)

    sigma_e = np.deg2rad(max(float(sigma_e_deg), 1e-6))
    sigma_r = max(float(sigma_r), 1e-9)
    sigma_d = max(float(spatial_sigma_ratio) * radius, 1e-9)

    for i, p in enumerate(points):
        ids = np.asarray(
            tree.query_ball_point(p, radius),
            dtype=int,
        )

        if len(ids) < min_neighbors:
            continue

        n0 = rough[i]
        if np.any(~np.isfinite(n0)):
            continue

        X = points[ids]
        local_rough = rough[ids]

        valid_n = np.all(np.isfinite(local_rough), axis=1)
        if np.sum(valid_n) < min_neighbors:
            continue

        # ------------------------------------------------------
        # Geometry / distance
        # ------------------------------------------------------
        delta = X - p
        d = np.linalg.norm(delta, axis=1)

        # ------------------------------------------------------
        # Sign-align rough local normals to query rough normal.
        # n and -n represent the same plane.
        # ------------------------------------------------------
        aligned = local_rough.copy()
        valid_idx = np.flatnonzero(valid_n)

        dots = aligned[valid_n] @ n0
        flip = dots < 0.0

        tmp = aligned[valid_n]
        tmp[flip] *= -1.0
        aligned[valid_n] = tmp

        # ------------------------------------------------------
        # Normal angular change theta_i
        # ------------------------------------------------------
        cosang = np.clip(
            aligned @ n0,
            -1.0,
            1.0,
        )
        theta = np.arccos(cosang)

        # Ignore query itself / near-zero distances for kappa.
        valid_kappa = (
            valid_n
            & np.isfinite(theta)
            & (d > 1e-9)
        )

        if np.sum(valid_kappa) >= 3:
            kappa_hat = np.median(
                theta[valid_kappa] / d[valid_kappa]
            )
        else:
            kappa_hat = 0.0

        # "Excess" normal change beyond smooth continuation.
        expected_theta = kappa_hat * d
        e = np.maximum(
            0.0,
            theta - expected_theta,
        )

        # ------------------------------------------------------
        # Rough query tangent-plane residual
        # ------------------------------------------------------
        residual = np.abs(delta @ n0)

        # ------------------------------------------------------
        # Soft weights
        # ------------------------------------------------------
        w_d = np.exp(
            -(d ** 2) / (2.0 * sigma_d ** 2)
        )

        w_e = 1.0 / (
            1.0 + (e / sigma_e) ** 2
        )

        w_r = 1.0 / (
            1.0 + (residual / sigma_r) ** 2
        )

        w = w_d * w_e * w_r
        w[~valid_n] = 0.0

        # ONE weighted PCA only.
        n = weighted_pca_normal(
            X,
            w,
            reference=n0,
        )

        if np.any(~np.isfinite(n)):
            continue

        # For analytic GT error / visualization only.
        if np.dot(n, gt_normals[i]) < 0:
            n = -n

        est[i] = n

        dot = np.clip(
            np.dot(n, gt_normals[i]),
            -1.0,
            1.0,
        )
        err[i] = np.degrees(np.arccos(dot))

    return tree, est, err, support


def estimate_iterative_normals(
    points,
    gt_normals,
    radius,
    sigma_e_deg=15.0,
    sigma_r=0.25,
    spatial_sigma_ratio=0.5,
    max_iterations=5,
    convergence_deg=0.05,
    min_neighbors=6,
):
    """
    Iterative soft-weighted PCA refinement.

    Initialization:
        n_p^(0) = ordinary PCA normal at query p.

    At iteration k:
        1) Sign-align each rough neighbor normal to n_p^(k).
        2) theta_i^(k) = angle(n_i^rough, n_p^(k))
        3) kappa_hat^(k) = median(theta_i^(k) / d_i)
        4) e_i^(k) = max(0, theta_i^(k) - kappa_hat^(k) d_i)
        5) r_i^(k) = |(q_i - p)^T n_p^(k)|
        6) w_i^(k) = w_d * w_e * w_r
        7) n_p^(k+1) = weighted PCA using w_i^(k)

    Stop when:
        angle(n_p^(k+1), n_p^(k)) < convergence_deg

    Important:
    - Rough neighbor normals are computed once and remain fixed.
    - Only the query normal / weights are iteratively updated.
    - This is IRLS-like, but the robust term includes the proposed
      smooth-normal-field excess e_i.
    """
    tree, rough, _, support = estimate_pca_normals(
        points,
        gt_normals,
        radius,
        min_neighbors=min_neighbors,
    )

    N = len(points)
    est = np.full((N, 3), np.nan)
    err = np.full(N, np.nan)

    sigma_e = np.deg2rad(max(float(sigma_e_deg), 1e-6))
    sigma_r = max(float(sigma_r), 1e-9)
    sigma_d = max(float(spatial_sigma_ratio) * radius, 1e-9)

    max_iterations = max(1, int(max_iterations))
    convergence_deg = max(float(convergence_deg), 0.0)

    for i, p in enumerate(points):
        ids = np.asarray(
            tree.query_ball_point(p, radius),
            dtype=int,
        )

        if len(ids) < min_neighbors:
            continue

        n_current = rough[i].copy()
        if np.any(~np.isfinite(n_current)):
            continue

        X = points[ids]
        local_rough = rough[ids]

        valid_n = np.all(np.isfinite(local_rough), axis=1)
        if np.sum(valid_n) < min_neighbors:
            continue

        delta = X - p
        d = np.linalg.norm(delta, axis=1)

        # Spatial weight stays fixed during iteration.
        w_d = np.exp(
            -(d ** 2) / (2.0 * sigma_d ** 2)
        )

        for _ in range(max_iterations):
            # --------------------------------------------------
            # Align rough local normals to current query normal.
            # --------------------------------------------------
            aligned = local_rough.copy()

            tmp = aligned[valid_n]
            dots = tmp @ n_current
            tmp[dots < 0.0] *= -1.0
            aligned[valid_n] = tmp

            # --------------------------------------------------
            # Angular deviation from current query normal.
            # --------------------------------------------------
            cosang = np.clip(
                aligned @ n_current,
                -1.0,
                1.0,
            )
            theta = np.arccos(cosang)

            # --------------------------------------------------
            # Robust local smooth-change scale.
            # --------------------------------------------------
            valid_kappa = (
                valid_n
                & np.isfinite(theta)
                & (d > 1e-9)
            )

            if np.sum(valid_kappa) >= 3:
                kappa_hat = np.median(
                    theta[valid_kappa] / d[valid_kappa]
                )
            else:
                kappa_hat = 0.0

            expected_theta = kappa_hat * d

            e = np.maximum(
                0.0,
                theta - expected_theta,
            )

            # --------------------------------------------------
            # Current query-plane residual.
            # --------------------------------------------------
            residual = np.abs(
                delta @ n_current
            )

            # --------------------------------------------------
            # Soft weights
            # --------------------------------------------------
            w_e = 1.0 / (
                1.0 + (e / sigma_e) ** 2
            )

            w_r = 1.0 / (
                1.0 + (residual / sigma_r) ** 2
            )

            w = w_d * w_e * w_r
            w[~valid_n] = 0.0

            n_new = weighted_pca_normal(
                X,
                w,
                reference=n_current,
            )

            if np.any(~np.isfinite(n_new)):
                break

            # Convergence uses axial normal angle.
            c = np.clip(
                abs(np.dot(n_new, n_current)),
                -1.0,
                1.0,
            )
            change_deg = np.degrees(
                np.arccos(c)
            )

            n_current = n_new

            if change_deg < convergence_deg:
                break

        # Analytic sign only for display / GT error.
        if np.dot(n_current, gt_normals[i]) < 0:
            n_current = -n_current

        est[i] = n_current

        dot = np.clip(
            np.dot(n_current, gt_normals[i]),
            -1.0,
            1.0,
        )
        err[i] = np.degrees(np.arccos(dot))

    return tree, est, err, support


# ============================================================
# GUI
# ============================================================

class LocalNormalInspector(tk.Tk):

    def __init__(self):
        super().__init__()

        self.title("Local PCA Normal Inspector")
        self.geometry("1600x900")
        self.minsize(1150, 720)

        self.P = None
        self.GT = None
        self.face = None

        self.tree = None
        self.est = None
        self.err = None
        self.support = None

        self.selected_idx = None
        self.global_scatter = None

        # -----------------------------
        # Parameters
        # -----------------------------
        self.geometry_var = tk.StringVar(value="Edge")
        self.angle_var = tk.DoubleVar(value=90.0)
        self.curvature_var = tk.DoubleVar(value=5.0)

        self.spacing_var = tk.DoubleVar(value=0.5)
        self.radius_var = tk.DoubleVar(value=2.0)
        self.noise_var = tk.DoubleVar(value=0.02)

        # Normal-estimation mode
        self.normal_mode_var = tk.StringVar(value="PCA")

        # ONESHOT parameters
        self.oneshot_sigma_e_var = tk.DoubleVar(value=15.0)   # deg
        self.oneshot_sigma_r_var = tk.DoubleVar(value=0.25)   # mm

        # ITERATIVE parameters
        self.iterations_var = tk.IntVar(value=5)

        self.local_arrow_stride_var = tk.IntVar(value=1)
        self.status_var = tk.StringVar(
            value="Generate geometry, then click a point."
        )

        self._build_ui()
        self._update_controls()

        self.after(100, self.regenerate)

    # --------------------------------------------------------
    # UI
    # --------------------------------------------------------

    def _build_ui(self):

        outer = ttk.Frame(self)
        outer.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(
            outer,
            padding=10,
        )
        controls.pack(
            side=tk.LEFT,
            fill=tk.Y,
        )

        plot_frame = ttk.Frame(outer)
        plot_frame.pack(
            side=tk.RIGHT,
            fill=tk.BOTH,
            expand=True,
        )

        row = 0

        ttk.Label(
            controls,
            text="Geometry",
            font=("", 11, "bold"),
        ).grid(
            row=row,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(0, 6),
        )
        row += 1

        ttk.Label(
            controls,
            text="Type",
        ).grid(
            row=row,
            column=0,
            sticky="w",
        )

        combo = ttk.Combobox(
            controls,
            textvariable=self.geometry_var,
            state="readonly",
            width=18,
            values=[
                "Smooth plane",
                "Smooth cylinder",
                "Edge",
                "Corner",
            ],
        )
        combo.grid(
            row=row,
            column=1,
            sticky="ew",
            pady=2,
        )
        combo.bind(
            "<<ComboboxSelected>>",
            lambda _e: self._geometry_changed(),
        )
        row += 1

        ttk.Label(
            controls,
            text="Edge / corner angle",
        ).grid(
            row=row,
            column=0,
            sticky="w",
        )

        self.angle_spin = ttk.Spinbox(
            controls,
            from_=5,
            to=170,
            increment=5,
            textvariable=self.angle_var,
            width=9,
        )
        self.angle_spin.grid(
            row=row,
            column=1,
            sticky="ew",
            pady=2,
        )
        row += 1

        ttk.Label(
            controls,
            text="Cylinder radius",
        ).grid(
            row=row,
            column=0,
            sticky="w",
        )

        self.curvature_spin = ttk.Spinbox(
            controls,
            from_=1,
            to=100,
            increment=0.5,
            textvariable=self.curvature_var,
            width=9,
        )
        self.curvature_spin.grid(
            row=row,
            column=1,
            sticky="ew",
            pady=2,
        )
        row += 1

        ttk.Separator(controls).grid(
            row=row,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=8,
        )
        row += 1

        ttk.Label(
            controls,
            text="Normal estimation",
            font=("", 11, "bold"),
        ).grid(
            row=row,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(0, 6),
        )
        row += 1

        ttk.Label(
            controls,
            text="Mode",
        ).grid(
            row=row,
            column=0,
            sticky="w",
        )

        self.mode_combo = ttk.Combobox(
            controls,
            textvariable=self.normal_mode_var,
            state="readonly",
            values=["PCA", "ONESHOT", "ITERATIVE"],
            width=12,
        )
        self.mode_combo.grid(
            row=row,
            column=1,
            sticky="ew",
            pady=2,
        )
        self.mode_combo.bind(
            "<<ComboboxSelected>>",
            lambda _e: self._mode_changed(),
        )
        row += 1

        row = self._spin_row(
            controls,
            row,
            "Point spacing [mm]",
            self.spacing_var,
            0.1,
            2.0,
            0.1,
        )

        row = self._spin_row(
            controls,
            row,
            "Radius [mm]",
            self.radius_var,
            0.2,
            6.0,
            0.1,
        )

        row = self._spin_row(
            controls,
            row,
            "Noise σ [mm]",
            self.noise_var,
            0.0,
            0.5,
            0.01,
        )

        ttk.Label(
            controls,
            text="ONESHOT σe [deg]",
        ).grid(
            row=row,
            column=0,
            sticky="w",
        )
        self.sigma_e_spin = ttk.Spinbox(
            controls,
            from_=1.0,
            to=90.0,
            increment=1.0,
            textvariable=self.oneshot_sigma_e_var,
            width=9,
        )
        self.sigma_e_spin.grid(
            row=row,
            column=1,
            sticky="ew",
            pady=2,
        )
        row += 1

        ttk.Label(
            controls,
            text="ONESHOT σr [mm]",
        ).grid(
            row=row,
            column=0,
            sticky="w",
        )
        self.sigma_r_spin = ttk.Spinbox(
            controls,
            from_=0.01,
            to=5.0,
            increment=0.05,
            textvariable=self.oneshot_sigma_r_var,
            width=9,
        )
        self.sigma_r_spin.grid(
            row=row,
            column=1,
            sticky="ew",
            pady=2,
        )
        row += 1

        ttk.Label(
            controls,
            text="ITERATIVE max iter",
        ).grid(
            row=row,
            column=0,
            sticky="w",
        )
        self.iter_spin = ttk.Spinbox(
            controls,
            from_=1,
            to=30,
            increment=1,
            textvariable=self.iterations_var,
            width=9,
        )
        self.iter_spin.grid(
            row=row,
            column=1,
            sticky="ew",
            pady=2,
        )
        row += 1

        row = self._spin_row(
            controls,
            row,
            "Local arrow stride",
            self.local_arrow_stride_var,
            1,
            20,
            1,
        )

        ttk.Button(
            controls,
            text="Regenerate",
            command=self.regenerate,
        ).grid(
            row=row,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(10, 3),
        )
        row += 1

        ttk.Button(
            controls,
            text="Export figure",
            command=self.export_figure,
        ).grid(
            row=row,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=3,
        )
        row += 1

        ttk.Label(
            controls,
            text=(
                "Selection\n"
                "---------\n"
                "Left 3D view에서\n"
                "point를 클릭하세요."
            ),
            justify=tk.LEFT,
        ).grid(
            row=row,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(12, 4),
        )
        row += 1

        ttk.Label(
            controls,
            textvariable=self.status_var,
            wraplength=260,
            justify=tk.LEFT,
        ).grid(
            row=row,
            column=0,
            columnspan=2,
            sticky="ew",
        )

        controls.columnconfigure(
            1,
            weight=1,
        )

        # ----------------------------------------------------
        # Figure layout
        # ----------------------------------------------------
        self.fig = Figure(
            figsize=(13, 8),
            constrained_layout=True,
        )

        gs = self.fig.add_gridspec(
            2,
            3,
            width_ratios=[1.25, 1.0, 1.0],
        )

        # Whole geometry uses full left column.
        self.ax_global = self.fig.add_subplot(
            gs[:, 0],
            projection="3d",
        )

        # Local neighborhood spans top right two panels.
        self.ax_local = self.fig.add_subplot(
            gs[0, 1:],
            projection="3d",
        )

        # Gauss map bottom center.
        self.ax_gauss = self.fig.add_subplot(
            gs[1, 1],
            projection="3d",
        )

        # Normal-vs-distance bottom right.
        self.ax_dist = self.fig.add_subplot(
            gs[1, 2],
        )
        self.ax_dist_angle = self.ax_dist.twinx()

        self.canvas = FigureCanvasTkAgg(
            self.fig,
            master=plot_frame,
        )
        self.canvas.get_tk_widget().pack(
            fill=tk.BOTH,
            expand=True,
        )

        toolbar = NavigationToolbar2Tk(
            self.canvas,
            plot_frame,
            pack_toolbar=False,
        )
        toolbar.update()
        toolbar.pack(
            side=tk.BOTTOM,
            fill=tk.X,
        )

        # One pick handler for interest-point selection.
        self.canvas.mpl_connect(
            "pick_event",
            self._on_pick,
        )

    def _spin_row(
        self,
        parent,
        row,
        text,
        variable,
        lo,
        hi,
        step,
    ):
        ttk.Label(
            parent,
            text=text,
        ).grid(
            row=row,
            column=0,
            sticky="w",
        )

        ttk.Spinbox(
            parent,
            from_=lo,
            to=hi,
            increment=step,
            textvariable=variable,
            width=9,
        ).grid(
            row=row,
            column=1,
            sticky="ew",
            pady=2,
        )

        return row + 1

    def _geometry_changed(self):
        self._update_controls()
        self.regenerate()

    def _mode_changed(self):
        self._update_controls()
        self.regenerate()

    def _update_controls(self):
        g = self.geometry_var.get()

        self.angle_spin.configure(
            state="normal"
            if g in ("Edge", "Corner")
            else "disabled"
        )

        self.curvature_spin.configure(
            state="normal"
            if g == "Smooth cylinder"
            else "disabled"
        )

        mode = self.normal_mode_var.get()
        weighted_mode = mode in ("ONESHOT", "ITERATIVE")

        self.sigma_e_spin.configure(
            state="normal" if weighted_mode else "disabled"
        )
        self.sigma_r_spin.configure(
            state="normal" if weighted_mode else "disabled"
        )
        self.iter_spin.configure(
            state="normal" if mode == "ITERATIVE" else "disabled"
        )

    # --------------------------------------------------------
    # Data generation
    # --------------------------------------------------------

    def regenerate(self):

        try:
            spacing = max(
                0.1,
                float(self.spacing_var.get()),
            )
            radius = max(
                0.2,
                float(self.radius_var.get()),
            )
            noise = max(
                0.0,
                float(self.noise_var.get()),
            )

            angle = float(
                np.clip(
                    float(self.angle_var.get()),
                    5.0,
                    170.0,
                )
            )

            curvature = max(
                1.0,
                float(self.curvature_var.get()),
            )

            extent = 7.0
            g = self.geometry_var.get()

            if g == "Smooth plane":
                P, GT, face = make_plane(
                    extent,
                    spacing,
                )

            elif g == "Smooth cylinder":
                P, GT, face = make_cylinder(
                    extent,
                    spacing,
                    curvature,
                )

            elif g == "Edge":
                P, GT, face = make_edge(
                    extent,
                    spacing,
                    angle,
                )

            elif g == "Corner":
                P, GT, face = make_corner(
                    extent,
                    spacing,
                    angle,
                )

            else:
                raise ValueError(
                    f"Unknown geometry: {g}"
                )

            rng = np.random.default_rng(0)
            P = P + rng.normal(
                0.0,
                noise,
                size=P.shape,
            )

            mode = self.normal_mode_var.get()

            if mode == "PCA":
                tree, est, err, support = estimate_pca_normals(
                    P,
                    GT,
                    radius,
                )

            elif mode == "ONESHOT":
                tree, est, err, support = estimate_oneshot_normals(
                    P,
                    GT,
                    radius,
                    sigma_e_deg=float(self.oneshot_sigma_e_var.get()),
                    sigma_r=float(self.oneshot_sigma_r_var.get()),
                    spatial_sigma_ratio=0.5,
                )

            elif mode == "ITERATIVE":
                tree, est, err, support = estimate_iterative_normals(
                    P,
                    GT,
                    radius,
                    sigma_e_deg=float(self.oneshot_sigma_e_var.get()),
                    sigma_r=float(self.oneshot_sigma_r_var.get()),
                    spatial_sigma_ratio=0.5,
                    max_iterations=int(self.iterations_var.get()),
                    convergence_deg=0.05,
                )

            else:
                raise ValueError(
                    f"Unknown normal mode: {mode}"
                )

            self.P = P
            self.GT = GT
            self.face = face

            self.tree = tree
            self.est = est
            self.err = err
            self.support = support

            # Default query:
            # choose a Face-0 point nearest to geometry center / feature.
            f0 = np.flatnonzero(face == 0)

            if g == "Edge":
                score = (
                    np.abs(P[f0, 0])
                    + np.abs(P[f0, 1] - radius * 0.5)
                    + np.abs(P[f0, 2])
                )

            elif g == "Corner":
                target = radius / math.sqrt(2.0)
                score = (
                    np.abs(P[f0, 0] - target)
                    + np.abs(P[f0, 1] - target)
                    + np.abs(P[f0, 2])
                )

            else:
                center = np.mean(P[f0], axis=0)
                score = np.linalg.norm(
                    P[f0] - center,
                    axis=1,
                )

            self.selected_idx = int(
                f0[np.argmin(score)]
            )

            self.refresh()

        except Exception as exc:
            messagebox.showerror(
                "Error",
                str(exc),
            )

    # --------------------------------------------------------
    # Picking
    # --------------------------------------------------------

    def _on_pick(self, event):
        """
        Point selection from the whole-geometry 3D scatter.

        Matplotlib returns one or more candidate indices in the plotted
        Path3DCollection. Pick the first candidate.
        """
        if self.global_scatter is None:
            return

        if event.artist is not self.global_scatter:
            return

        inds = np.atleast_1d(event.ind)

        if len(inds) == 0:
            return

        # global_scatter contains only finite-normal points.
        plotted_ids = self._global_plot_ids

        local_index = int(inds[0])

        if local_index >= len(plotted_ids):
            return

        self.selected_idx = int(
            plotted_ids[local_index]
        )

        self.refresh()

    # --------------------------------------------------------
    # Plotting
    # --------------------------------------------------------

    def refresh(self):

        if self.P is None:
            return

        # Remove old colorbars before recreating axes content.
        for attr in (
            "_global_cb",
            "_gauss_cb",
        ):
            cb = getattr(
                self,
                attr,
                None,
            )
            if cb is not None:
                try:
                    cb.remove()
                except Exception:
                    pass
                setattr(
                    self,
                    attr,
                    None,
                )

        self.ax_global.clear()
        self.ax_local.clear()
        self.ax_gauss.clear()
        self.ax_dist.clear()
        self.ax_dist_angle.clear()

        self._plot_global()
        self._plot_local()
        self._plot_gauss()
        self._plot_distance_normal()

        self._update_status()

        self.canvas.draw_idle()

    # --------------------------------------------------------
    # Whole geometry
    # --------------------------------------------------------

    def _plot_global(self):

        valid = (
            np.isfinite(self.err)
            & np.all(
                np.isfinite(self.est),
                axis=1,
            )
        )

        ids = np.flatnonzero(valid)
        self._global_plot_ids = ids

        self.global_scatter = self.ax_global.scatter(
            self.P[ids, 0],
            self.P[ids, 1],
            self.P[ids, 2],
            c=np.clip(
                self.err[ids],
                0,
                45,
            ),
            s=14,
            vmin=0,
            vmax=45,
            picker=True,
            pickradius=5,
        )

        i = self.selected_idx

        if i is not None:

            p = self.P[i]

            self.ax_global.scatter(
                [p[0]],
                [p[1]],
                [p[2]],
                marker="*",
                s=200,
                label="query",
            )

            R = float(
                self.radius_var.get()
            )

            # Approximate radius sphere around query to show support region.
            uu = np.linspace(
                0,
                2 * np.pi,
                22,
            )
            vv = np.linspace(
                0,
                np.pi,
                12,
            )

            xs = (
                p[0]
                + R
                * np.outer(
                    np.cos(uu),
                    np.sin(vv),
                )
            )
            ys = (
                p[1]
                + R
                * np.outer(
                    np.sin(uu),
                    np.sin(vv),
                )
            )
            zs = (
                p[2]
                + R
                * np.outer(
                    np.ones_like(uu),
                    np.cos(vv),
                )
            )

            self.ax_global.plot_wireframe(
                xs,
                ys,
                zs,
                rstride=3,
                cstride=3,
                linewidth=0.35,
                alpha=0.15,
            )

        self.ax_global.set_title(
            f"Global geometry — {self.normal_mode_var.get()}\n"
            "click a point to select query"
        )

        self.ax_global.set_xlabel("x [mm]")
        self.ax_global.set_ylabel("y [mm]")
        self.ax_global.set_zlabel("z [mm]")

        self._equal_3d(
            self.ax_global,
            self.P[ids],
        )

        self._global_cb = self.fig.colorbar(
            self.global_scatter,
            ax=self.ax_global,
            shrink=0.52,
            pad=0.07,
        )
        self._global_cb.set_label(
            "GT angular error [deg]"
        )

    # --------------------------------------------------------
    # Local neighborhood + arrows
    # --------------------------------------------------------

    def _neighbor_ids(self):

        if self.selected_idx is None:
            return np.array([], dtype=int)

        p = self.P[self.selected_idx]

        ids = np.asarray(
            self.tree.query_ball_point(
                p,
                float(self.radius_var.get()),
            ),
            dtype=int,
        )

        return ids

    def _plot_local(self):

        i = self.selected_idx

        if i is None:
            self.ax_local.set_title(
                "No query selected"
            )
            return

        ids = self._neighbor_ids()
        p = self.P[i]

        # Plot faces separately so cross-surface contamination is obvious.
        for f in np.unique(
            self.face[ids]
        ):

            sub = ids[
                self.face[ids] == f
            ]

            self.ax_local.scatter(
                self.P[sub, 0],
                self.P[sub, 1],
                self.P[sub, 2],
                s=24,
                label=f"Face {int(f)}",
            )

        # Query
        self.ax_local.scatter(
            [p[0]],
            [p[1]],
            [p[2]],
            marker="*",
            s=180,
            label="query",
        )

        # Neighbor normal arrows
        # PCA normal sign is ambiguous, so for visualization only we flip
        # each local normal into the same hemisphere as the selected query
        # normal. This makes the arrows visually coherent.
        stride = max(
            1,
            int(
                self.local_arrow_stride_var.get()
            ),
        )

        arrow_ids = ids[::stride]

        finite = np.all(
            np.isfinite(
                self.est[arrow_ids]
            ),
            axis=1,
        )

        arrow_ids = arrow_ids[finite]

        query_ref = self.est[i]
        if np.any(~np.isfinite(query_ref)):
            query_ref = self.GT[i]

        arrow_normals = orient_normals_to_reference(
            self.est[arrow_ids],
            query_ref,
        )

        L = max(
            0.25,
            min(
                0.8,
                0.35
                * float(
                    self.radius_var.get()
                ),
            ),
        )

        if len(arrow_ids):

            self.ax_local.quiver(
                self.P[arrow_ids, 0],
                self.P[arrow_ids, 1],
                self.P[arrow_ids, 2],
                arrow_normals[:, 0],
                arrow_normals[:, 1],
                arrow_normals[:, 2],
                length=L,
                normalize=True,
                linewidth=0.8,
                alpha=0.8,
            )

        # Query estimated normal
        n = self.est[i]

        if np.all(
            np.isfinite(n)
        ):
            self.ax_local.quiver(
                p[0],
                p[1],
                p[2],
                n[0],
                n[1],
                n[2],
                length=1.5 * L,
                normalize=True,
                linewidth=3.0,
                label="query estimated",
            )

        # Query GT normal
        ngt = self.GT[i]

        self.ax_local.quiver(
            p[0],
            p[1],
            p[2],
            ngt[0],
            ngt[1],
            ngt[2],
            length=1.5 * L,
            normalize=True,
            linewidth=3.0,
            label="query GT",
        )

        qdist = np.linalg.norm(
            self.P[ids] - p,
            axis=1,
        )

        other = np.sum(
            self.face[ids]
            != self.face[i]
        )

        self.ax_local.set_title(
            f"Local neighborhood normals — {self.normal_mode_var.get()}\n"
            f"R={self.radius_var.get():.2f} mm, "
            f"N={len(ids)}, "
            f"other-face={100*other/max(len(ids),1):.1f}%"
        )

        self.ax_local.set_xlabel("x [mm]")
        self.ax_local.set_ylabel("y [mm]")
        self.ax_local.set_zlabel("z [mm]")

        self.ax_local.legend(
            fontsize=8,
            ncol=2,
        )

        self._equal_3d(
            self.ax_local,
            self.P[ids],
        )

    # --------------------------------------------------------
    # Local Gauss map
    # --------------------------------------------------------

    def _plot_gauss(self):

        i = self.selected_idx

        if i is None:
            return

        ids = self._neighbor_ids()
        p = self.P[i]

        normals = self.est[ids]

        valid = np.all(
            np.isfinite(normals),
            axis=1,
        )

        ids = ids[valid]
        normals = normals[valid]

        if len(ids) == 0:
            return

        query_ref = self.est[i]
        if np.any(~np.isfinite(query_ref)):
            query_ref = self.GT[i]

        normals = orient_normals_to_reference(
            normals,
            query_ref,
        )

        d = np.linalg.norm(
            self.P[ids] - p,
            axis=1,
        )

        # Unit sphere
        u = np.linspace(
            0,
            2 * np.pi,
            34,
        )
        v = np.linspace(
            0,
            np.pi,
            18,
        )

        xs = np.outer(
            np.cos(u),
            np.sin(v),
        )
        ys = np.outer(
            np.sin(u),
            np.sin(v),
        )
        zs = np.outer(
            np.ones_like(u),
            np.cos(v),
        )

        self.ax_gauss.plot_wireframe(
            xs,
            ys,
            zs,
            rstride=3,
            cstride=3,
            linewidth=0.3,
            alpha=0.12,
        )

        sc = self.ax_gauss.scatter(
            normals[:, 0],
            normals[:, 1],
            normals[:, 2],
            c=d,
            s=24,
        )

        # Draw only a few representative arrows.
        order = np.argsort(d)

        reps = sorted(
            set([
                int(order[0]),
                int(order[len(order)//2]),
                int(order[-1]),
            ])
        )

        for pos in reps:

            n = normals[pos]

            self.ax_gauss.quiver(
                0,
                0,
                0,
                n[0],
                n[1],
                n[2],
                length=1.0,
                normalize=True,
                linewidth=1.4,
                alpha=0.8,
            )

        # Query normal
        nq = self.est[i]

        if np.all(
            np.isfinite(nq)
        ):

            self.ax_gauss.quiver(
                0,
                0,
                0,
                nq[0],
                nq[1],
                nq[2],
                length=1.0,
                normalize=True,
                linewidth=3.0,
            )

            self.ax_gauss.scatter(
                [nq[0]],
                [nq[1]],
                [nq[2]],
                marker="*",
                s=100,
            )

        # Query GT
        ngt = self.GT[i]

        self.ax_gauss.quiver(
            0,
            0,
            0,
            ngt[0],
            ngt[1],
            ngt[2],
            length=1.0,
            normalize=True,
            linewidth=3.0,
        )

        self.ax_gauss.set_title(
            "Local Gauss map (sign-aligned)\n"
            "color = distance from query"
        )

        self.ax_gauss.set_xlabel("nx")
        self.ax_gauss.set_ylabel("ny")
        self.ax_gauss.set_zlabel("nz")

        self.ax_gauss.set_xlim(
            -1.05,
            1.05,
        )
        self.ax_gauss.set_ylim(
            -1.05,
            1.05,
        )
        self.ax_gauss.set_zlim(
            -1.05,
            1.05,
        )
        self.ax_gauss.set_box_aspect(
            (1, 1, 1)
        )

        self._gauss_cb = self.fig.colorbar(
            sc,
            ax=self.ax_gauss,
            shrink=0.60,
            pad=0.08,
        )
        self._gauss_cb.set_label(
            "distance to query [mm]"
        )

    # --------------------------------------------------------
    # Normal versus query distance
    # --------------------------------------------------------

    def _plot_distance_normal(self):

        i = self.selected_idx

        if i is None:
            return

        ids = self._neighbor_ids()
        p = self.P[i]

        normals = self.est[ids]

        valid = np.all(
            np.isfinite(normals),
            axis=1,
        )

        ids = ids[valid]
        normals = normals[valid]

        if len(ids) == 0:
            return

        query_ref = self.est[i]
        if np.any(~np.isfinite(query_ref)):
            query_ref = self.GT[i]

        normals = orient_normals_to_reference(
            normals,
            query_ref,
        )

        d = np.linalg.norm(
            self.P[ids] - p,
            axis=1,
        )

        order = np.argsort(d)

        d = d[order]
        ids = ids[order]
        normals = normals[order]

        # Three normal components
        self.ax_dist.scatter(
            d,
            normals[:, 0],
            s=15,
            label="nx",
        )
        self.ax_dist.scatter(
            d,
            normals[:, 1],
            s=15,
            label="ny",
        )
        self.ax_dist.scatter(
            d,
            normals[:, 2],
            s=15,
            label="nz",
        )

        # Query estimated components as horizontal reference.
        nq = self.est[i]

        if np.all(
            np.isfinite(nq)
        ):

            self.ax_dist.axhline(
                nq[0],
                linestyle=":",
                linewidth=0.8,
            )
            self.ax_dist.axhline(
                nq[1],
                linestyle=":",
                linewidth=0.8,
            )
            self.ax_dist.axhline(
                nq[2],
                linestyle=":",
                linewidth=0.8,
            )

        # Also encode angular difference from query GT using point size.
        ngt = self.GT[i]

        delta = np.array([
            angular_deg(n, ngt)
            for n in normals
        ])

        # Overlay angular difference from query GT on the persistent
        # secondary y-axis.
        ax2 = self.ax_dist_angle

        ax2.plot(
            d,
            delta,
            linewidth=1.0,
            alpha=0.55,
            label="angle to query GT",
        )

        ax2.set_ylabel(
            "angle to query GT [deg]"
        )

        self.ax_dist.set_title(
            "Neighbor normal vs distance to query (sign-aligned)"
        )

        self.ax_dist.set_xlabel(
            "distance to query [mm]"
        )
        self.ax_dist.set_ylabel(
            "normal component"
        )

        self.ax_dist.set_ylim(
            -1.05,
            1.05,
        )

        self.ax_dist.grid(
            alpha=0.22,
        )

        self.ax_dist.legend(
            fontsize=8,
            loc="lower left",
        )

        ax2.legend(
            fontsize=8,
            loc="upper right",
        )

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    def _update_status(self):

        if self.selected_idx is None:
            return

        i = self.selected_idx
        ids = self._neighbor_ids()

        p = self.P[i]
        n = self.est[i]
        ngt = self.GT[i]

        other = np.sum(
            self.face[ids]
            != self.face[i]
        )

        text = (
            f"Mode: {self.normal_mode_var.get()}\n"
            f"Query index: {i}\n"
            f"Face: {self.face[i]}\n"
            f"p = [{p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}] mm\n"
            f"Support: {len(ids)} points\n"
            f"Other-face: "
            f"{100*other/max(len(ids),1):.1f}%\n"
            f"Normal error: "
            f"{self.err[i]:.3f} deg"
        )

        if self.normal_mode_var.get() in ("ONESHOT", "ITERATIVE"):
            text += (
                f"\nσe={self.oneshot_sigma_e_var.get():.1f} deg"
                f", σr={self.oneshot_sigma_r_var.get():.3f} mm"
                f", σd={0.5*self.radius_var.get():.3f} mm"
            )

        if self.normal_mode_var.get() == "ITERATIVE":
            text += (
                f"\nmax iter={self.iterations_var.get()}, "
                f"stop Δn<0.05 deg"
            )

        self.status_var.set(text)

    # --------------------------------------------------------
    # Figure formatting
    # --------------------------------------------------------

    @staticmethod
    def _equal_3d(ax, P):

        if len(P) == 0:
            return

        lo = np.min(
            P,
            axis=0,
        )
        hi = np.max(
            P,
            axis=0,
        )

        center = 0.5 * (
            lo + hi
        )

        half = 0.5 * np.max(
            hi - lo
        )

        if (
            not np.isfinite(half)
            or half <= 1e-9
        ):
            half = 1.0

        ax.set_xlim(
            center[0] - half,
            center[0] + half,
        )
        ax.set_ylim(
            center[1] - half,
            center[1] + half,
        )
        ax.set_zlim(
            center[2] - half,
            center[2] + half,
        )

        ax.set_box_aspect(
            (1, 1, 1)
        )

    # --------------------------------------------------------
    # Export
    # --------------------------------------------------------

    def export_figure(self):

        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"),
                ("PDF", "*.pdf"),
            ],
        )

        if not path:
            return

        self.fig.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
        )

        self.status_var.set(
            self.status_var.get()
            + f"\nSaved: {path}"
        )


if __name__ == "__main__":
    LocalNormalInspector().mainloop()