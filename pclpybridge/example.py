"""
pclpybridge example.py

각 함수의 최소 실행 예제.
기본 예제는 NumPy 배열을 사용하며, Open3D legacy PointCloud도 대부분 그대로 입력할 수 있습니다.

주의:
- 좌표가 meter 단위라고 가정한 예시입니다.
- 실제 데이터에서는 radius / threshold / PPF 파라미터를 물체 크기에 맞게 조정하세요.
"""

import numpy as np
import pclpybridge as pcl


# ---------------------------------------------------------------------
# 예제용 point cloud 생성
# ---------------------------------------------------------------------
rng = np.random.default_rng(0)

# 약 10 cm 크기의 간단한 3D point cloud
points = rng.uniform(-0.05, 0.05, size=(3000, 3)).astype(np.float32)

# PPF 예제를 위한 model / scene
model_points = points.copy()

theta = np.deg2rad(15.0)
R = np.array(
    [
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0,            0.0,           1.0],
    ],
    dtype=np.float32,
)
t = np.array([0.02, -0.01, 0.015], dtype=np.float32)
scene_points = model_points @ R.T + t


# ---------------------------------------------------------------------
# 1. VoxelGrid downsample
# ---------------------------------------------------------------------
down = pcl.voxel_downsample(
    points,
    leaf_size=0.005,  # 5 mm
)
print("voxel_downsample:", down.shape)


# ---------------------------------------------------------------------
# 2. Normal estimation
# ---------------------------------------------------------------------
normals, curvature = pcl.estimate_normals(
    points,
    radius=0.010,  # 10 mm
)
print("estimate_normals:", normals.shape, curvature.shape)


# ---------------------------------------------------------------------
# 3. Harris3D keypoint
# ---------------------------------------------------------------------
harris_points, harris_response = pcl.harris3d(
    points,
    radius=0.010,
    threshold=1e-6,
    nonmax=True,
    refine=False,
    method="HARRIS",
)
print("harris3d:", harris_points.shape, harris_response.shape)


# ---------------------------------------------------------------------
# 4. ISS3D keypoint
# ---------------------------------------------------------------------
iss_points = pcl.iss3d(
    points,
    salient_radius=0.012,
    nonmax_radius=0.008,
    gamma21=0.975,
    gamma32=0.975,
    min_neighbors=5,
)
print("iss3d:", iss_points.shape)


# ---------------------------------------------------------------------
# 5. SIFT3D keypoint
# ---------------------------------------------------------------------
# PCL SIFTKeypoint는 intensity가 필요합니다.
intensity = np.linalg.norm(points, axis=1).astype(np.float32)

sift_points = pcl.sift3d(
    points,
    intensity=intensity,
    min_scale=0.005,
    n_octaves=3,
    n_scales_per_octave=4,
    min_contrast=0.001,
)
# 결과 열: [x, y, z, scale]
print("sift3d:", sift_points.shape)


# ---------------------------------------------------------------------
# 6. FPFH descriptor
# ---------------------------------------------------------------------
fpfh = pcl.fpfh(
    points,
    radius=0.015,
    normals=normals,
    normal_radius=0.010,
)
# shape: (N, 33)
print("fpfh:", fpfh.shape)


# ---------------------------------------------------------------------
# 7. SHOT descriptor
# ---------------------------------------------------------------------
shot = pcl.shot(
    points,
    radius=0.020,
    normals=normals,
    normal_radius=0.010,
)
# shape: (N, 352)
print("shot:", shot.shape)


# ---------------------------------------------------------------------
# 8. PPF Registration
# ---------------------------------------------------------------------
# 실제 laser scan에서는 scan마다 sensor viewpoint를 기준으로 normal을
# orientation한 뒤 scene_normals로 직접 넘기는 것을 권장합니다.

model_normals, _ = pcl.estimate_normals(
    model_points,
    radius=0.010,
)

scene_normals, _ = pcl.estimate_normals(
    scene_points,
    radius=0.010,
)

result = pcl.ppf_register(
    model_points,
    scene_points,
    model_normals=model_normals,
    scene_normals=scene_normals,
    normal_radius=0.010,
    angle_step_deg=12.0,
    distance_step=0.005,              # 5 mm
    scene_reference_rate=5,
    position_cluster_threshold=0.010, # 10 mm
    rotation_cluster_threshold_deg=20.0,
    max_candidates=20,
)

print("ppf_register converged:", result.converged)
print("PPF candidate backend:", result.candidate_backend)
print("PPF Top1 transform:")
print(result.top_transform)
print("PPF votes:", result.votes)
print("PPF relative votes:", result.relative_votes)


# ---------------------------------------------------------------------
# Open3D PointCloud 사용 예시
# ---------------------------------------------------------------------
# import open3d as o3d
#
# cloud = o3d.io.read_point_cloud("scan.ply")
#
# kp, response = pcl.harris3d(
#     cloud,
#     radius=0.004,
#     threshold=1e-6,
# )
#
# desc = pcl.shot(
#     cloud,
#     radius=0.012,
#     normal_radius=0.004,
# )
