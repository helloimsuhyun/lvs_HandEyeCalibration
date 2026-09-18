"""Drop-in usage snippet for the user's existing pcl_harris3d.py.

Assumes `points` is an (N,3) NumPy array in millimetres.
"""
import numpy as np
import sanchez_wpca

# points = ...  # your sampled CAD point cloud, unit: mm

result = sanchez_wpca.estimate_normals(
    np.asarray(points, dtype=np.float32),
    k=50,
    noise=0.0,                  # ideal STL sampling; same unit as points (mm)
    curvature_radius=np.inf,    # use only when the object is piecewise planar
    unit_scale_to_meter=1e-3,   # mm -> m internally
    orient_to_origin=False,
    verbose=False,
)

sanchez_normals = np.asarray(result["normals"], dtype=np.float64)
optimized_mask = np.asarray(result["optimized_mask"], dtype=bool)

print("Sanchez optimized:", optimized_mask.sum(), "/", len(points))
print("Sanchez elapsed ms:", result["elapsed_ms"])

# Example with the user's existing PCL Harris wrapper:
# R_points, R = pcl.harris3d(
#     np.asarray(points, dtype=np.float32),
#     radius=2.0,
#     threshold=0.0,
#     nonmax=False,
#     refine=False,
#     method="HARRIS",
#     normals=np.asarray(sanchez_normals, dtype=np.float32),
# )
