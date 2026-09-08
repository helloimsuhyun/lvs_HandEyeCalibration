# pclpybridge

Python에서 일부 PCL 기능을 NumPy/Open3D point cloud로 호출하기 위한 pybind11 wrapper.

## 검증 환경

- Ubuntu 22.04
- Python 3.10
- PCL 1.12

## 설치 / 실행

Ubuntu:

```bash
sudo apt update
sudo apt install -y libpcl-dev python3-dev cmake ninja-build

cd pclpybridge_pcl112_fix
rm -rf build
python -m pip install --no-cache-dir .
```

설치 확인:

```bash
python -c "import pclpybridge; print('pclpybridge OK')"
```

예제 실행:

```bash
python example.py
```

## 포함 함수

- `voxel_downsample()` : PCL VoxelGrid
- `estimate_normals()` : PCL NormalEstimationOMP
- `harris3d()` : PCL HarrisKeypoint3D
- `iss3d()` : PCL ISSKeypoint3D
- `sift3d()` : PCL SIFTKeypoint
- `fpfh()` : PCL FPFHEstimationOMP
- `shot()` : PCL SHOTEstimation
- `ppf_register()` : PCL PPF registration

`ppf_register()`은 `result.transforms`, `result.votes`로 pose 후보와 vote를 반환합니다.

- PCL >= 1.14: PCL의 `getBestPoseCandidates()` 직접 사용
- PCL 1.12/1.13: 해당 getter가 없어서 PCL 1.12 voting/clustering 방식의 compatibility path 사용

> 거리/radius 파라미터 단위는 입력 point cloud 좌표 단위와 동일합니다.
