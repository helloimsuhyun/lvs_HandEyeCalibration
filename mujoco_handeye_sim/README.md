# 교체형 Hand-eye MuJoCo 환경

로봇 URDF, 센서 CAD, 초기 `T_tcp_sensor`만 받아 다음 구조의 MuJoCo 모델을
생성한다.

```text
URDF의 movable link ── URDF fixed joint ── TCP ── T_tcp_sensor ── sensor origin
                                                              ├─ 실제 CAD visual
                                                              └─ 빠른 collision proxy
```

현재 예제는 Rainbow Robotics 공식 `rb5_850e` URDF/충돌 메시와 사용자가 제공한
Keyence LJ-V7080 STEP을 사용한다. 생성 결과는 이 폴더의 `build/`에만 생기므로
현재 `real_laser_handeye` 코드는 수정하지 않는다.

## 1. 바로 빌드

저장소 루트에서 실행한다.

```bash
python3 -m pip install -e ./mujoco_handeye_sim
handeye-mujoco build \
  --config mujoco_handeye_sim/configs/rb5_ljv7080.yaml
handeye-mujoco check \
  --model mujoco_handeye_sim/build/rb5_ljv7080/model.xml
handeye-mujoco view \
  --model mujoco_handeye_sim/build/rb5_ljv7080/model.xml
```

설치 없이도 다음처럼 실행할 수 있다.

```bash
PYTHONPATH=mujoco_handeye_sim/src python3 -m handeye_mujoco build \
  --config mujoco_handeye_sim/configs/rb5_ljv7080.yaml
```

`view`는 그래픽 세션이 필요하다. 서버에서는 `build`, `check`, Python 충돌 API는
GUI 없이 동작한다.

충돌 형상을 확실하게 보려면 센서 CAD를 숨기고 로봇 충돌 메시를 파란색, 센서
충돌 프록시를 빨간색으로 표시한다.

```bash
PYTHONPATH=mujoco_handeye_sim/src python3 -m handeye_mujoco view \
  --collision-only \
  --model mujoco_handeye_sim/build/rb5_ljv7080/model.xml
```

## 2. 장비 교체: 세 입력만 변경

`configs/rb5_ljv7080.yaml`을 복사하고 아래를 바꾼다.

1. `robot.urdf`: 일반 URDF 파일
2. `sensor.cad`: STEP/STP, STL 또는 OBJ 파일
3. `handeye.path`: 4x4 `T_tcp_sensor` JSON/CSV/TXT/NPY

추가로 이름/단위에 해당하는 다음 메타데이터만 맞춘다.

- `robot.tcp_link`: URDF에서 TCP를 나타내는 link 이름
- `robot.package_roots`: URDF가 `package://...` 메시 URI를 쓸 때만 필요
- `sensor.cad_units`: `mm` 또는 `m`
- `handeye.translation_units`: 행렬 translation의 `mm` 또는 `m`

즉 코드 변경은 필요 없다. URDF에 TCP fixed link가 있으면 builder가 그 fixed
transform을 자동으로 접고, MuJoCo에 남는 가장 가까운 link 아래에 센서를 부착한다.

### STEP 센서를 새로 넣는 경우

MuJoCo는 STEP을 직접 읽지 않는다. 현재 LJ-V7080에는 미리 변환한 캐시가 함께
있다. 다른 STEP으로 교체할 때 `preconverted_mesh`를 지우고 한 번만 CAD 옵션을
설치해 삼각분할한다.

캐시는 원본 STEP의 SHA-256과 함께 검증되므로 STEP만 바꿔도 예전 LJ-V7080
메시를 잘못 재사용하지 않는다.

```bash
python3 -m pip install -e './mujoco_handeye_sim[cad]'
handeye-mujoco build --force-cad --config path/to/new_config.yaml
```

FreeCAD의 `FreeCADCmd`가 설치되어 있으면 CadQuery 없이도 자동 변환한다. STL/OBJ
입력에는 어느 쪽도 필요 없다.

## 3. 좌표계 계약

입력 행렬은 반드시 다음 의미여야 한다.

```text
p_tcp = T_tcp_sensor @ p_sensor
T_world_sensor = T_world_tcp @ T_tcp_sensor
```

현재 `real_laser_handeye/initial_T_tcp_sensor.json`은 translation이 mm이므로 예제
설정에 `translation_units: mm`가 지정되어 있다. 회전행렬은 직교행렬이고
determinant가 +1인지 빌드 시 검증한다.

CAD 원점과 측정 원점이 다르면 `sensor.frame`으로 정합한다. 이 값은 CAD 자체나
`T_tcp_sensor`를 바꾸지 않고, CAD visual/collision만 측정 프레임에 배치한다.
workflow 경로 계획에서는 hand-eye JSON의 `T_sensor_physical = ^S T_P`로
물리 원점과 측정 원점의 차이를 지정한다. 방향과 거리는 JSON 값을 그대로
신뢰한다. 원형 pose pattern은 `P`를 기준으로 만든 다음
`S = P @ inv(T_S_P)`로 측정 좌표계를 복원한다.

현재 LJ-V7080은 제공된 도면과 STEP의 측정영역을 대조해 다음과 같이 설정했다.

- 측정 원점: 사다리꼴 중앙 (`X=±16 mm`, `Z=±23 mm`가 만나는 십자점)
- STEP 좌표의 원점 위치: `[-52.6428785, -97.6682626, 12.7832449] mm`
- 측정축: `+X = CAD +Z`, `+Y = CAD +X`, `+Z = CAD +Y`
- Keyence 부호: NEAR 측은 `+Z`, FAR 측은 `-Z`
- 물리 헤드의 앞쪽 표면: 측정 원점으로부터 `Z=+80 mm`

따라서 Keyence profile의 `[x, 0, z]`와 시뮬레이션 `sensor_origin`이 같은 프레임을
쓴다. `T_tcp_sensor`도 이 사다리꼴 중심을 원점으로 하는 변환으로 유지된다.

LJ-V7080 STEP에는 실제 헤드뿐 아니라 사다리꼴 측정영역 도형과 케이블도 들어 있다.
전체 CAD bounding box를 충돌 형상으로 쓰면 측정공간 자체가 장애물로 판정되므로,
예제는 도면의 96×71×42 mm 물리 헤드를 collision box로 사용한다. STEP에 고정된
커넥터와 케이블 경로는 별도의 capsule 체인으로 근사한다. 실제 설치에서 케이블
배선이 CAD와 다르면 YAML의 capsule 끝점을 실제 경로에 맞게 수정해야 한다.

## 4. 충돌 검사 API

`examples/collision_api.py`에 최소 예제가 있다.

```python
from handeye_mujoco import HandEyeSimulation

sim = HandEyeSimulation("mujoco_handeye_sim/build/rb5_ljv7080/model.xml")
sim.reset_home()
sim.set_joint_positions([0, -1.57, 1.57, 0, 1.57, 0])

report = sim.collision_report()
T_world_sensor = sim.sensor_pose_world()
safe, first_bad_index, report = sim.trajectory_is_collision_free(q_trajectory)
```

`scene.safety_margin_m` 안으로 들어온 contact도 경로 계획에서는 unsafe로 처리한다.
센서는 기본적으로 실제 CAD를 visual로, CAD bounding box에 padding을 더한 box를
collision으로 사용한다. 이 보수적 프록시는 캘리브레이션 pose 후보를 대량으로
검사할 때 빠르고 안전 측으로 치우친다.

현재 범위는 self/environment collision 판정과 센서 pose 제공까지다. 다음 단계의
캘리브레이션 궤적 생성기는 이 API 위에서 다음 순서로 붙이면 된다.

1. 원하는 `T_world_sensor` 후보 생성
2. `T_world_tcp = T_world_sensor @ inv(T_tcp_sensor)`로 TCP 목표 변환
3. RB5 IK로 joint 후보 생성 및 joint limit 확인
4. 각 구간 interpolation 후 `trajectory_is_collision_free` 검사
5. 실제 로봇 실행 전 속도/가속도/특이점/작업공간 제한을 추가 검증

## 5. 생성물

- `build/.../model.xml`: 최종 MJCF
- `build/.../robot_prepared.urdf`: package URI가 해소된 중간 URDF
- `build/.../meshes/`: 로봇 및 센서 로컬 메시
- `build/.../manifest.json`: 사용한 입력, 합성 transform, 센서 bounds, MuJoCo 통계

`manifest.json`의 `T_attachment_sensor_m`로 실제 합성 결과를 추적할 수 있다.
RB5 원본 자산의 출처와 Apache-2.0 라이선스는 `assets/rb5_850e`에 보존했다.

## 안전 주의

이 시뮬레이션의 collision-free 결과만으로 실제 로봇을 움직이면 안 된다. URDF
충돌 메시 오차, 브래킷/케이블/워크홀더 누락, TCP 및 hand-eye 초기값 오차를 포함한
추가 여유를 두고, 실제 RB5 controller의 joint/속도/안전 제한과 저속 dry-run을
별도로 적용해야 한다.
