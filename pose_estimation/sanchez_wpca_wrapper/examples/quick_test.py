import numpy as np
import sanchez_wpca

# Two perpendicular planes, coordinates in millimetres.
rng = np.random.default_rng(0)
n = 35
u = np.linspace(-5.0, 5.0, n)
v = np.linspace(-5.0, 5.0, n)
U, V = np.meshgrid(u, v)

plane_a = np.c_[U.ravel(), V.ravel(), np.zeros(U.size)]
plane_b = np.c_[U.ravel(), np.zeros(U.size), V.ravel()]
points = np.vstack([plane_a, plane_b]).astype(np.float32)
points += rng.normal(0.0, 0.02, points.shape).astype(np.float32)

result = sanchez_wpca.estimate_normals(
    points,
    k=50,
    noise=0.02,                 # mm
    curvature_radius=np.inf,    # piecewise planar
    unit_scale_to_meter=1e-3,   # input is mm
    orient_to_origin=False,
    verbose=False,
)

normals = np.asarray(result["normals"])
edge_mask = np.asarray(result["optimized_mask"], dtype=bool)
second_mask = np.asarray(result["second_init_mask"], dtype=bool)

print("points            :", points.shape)
print("normals           :", normals.shape)
print("optimized points  :", edge_mask.sum())
print("second init points:", second_mask.sum())
print("nan count         :", result["nan_count"])
print("elapsed           : %.2f ms" % result["elapsed_ms"])
print("first 5 normals:\n", normals[:5])
