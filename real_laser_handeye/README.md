# Real Laser Hand-Eye Calibration

모든 명령은 저장소 루트에서 실행한다.

```bash
cd /home/choisuhyun/lvs_HandEyeCalibration
```

## 1. 의존성 설치

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e ./mujoco_handeye_sim
```

REAL 모드에서 RB5를 연결할 경우:

```bash
python3 -m pip install -e ./real_laser_handeye/rbpodo
sudo apt-get update
sudo apt-get install python3-tk
```

ROS 2 Humble 및 MoveIt 2 설치:

```bash
sudo apt-get update
sudo apt-get install \
  ros-humble-moveit \
  ros-humble-moveit-planners-ompl \
  ros-humble-pilz-industrial-motion-planner
```

## 2. MuJoCo 모델 빌드

```bash
PYTHONPATH="$(pwd)/mujoco_handeye_sim/src${PYTHONPATH:+:$PYTHONPATH}" \
python3 -m handeye_mujoco build \
  --config mujoco_handeye_sim/configs/rb5_ljv7080.yaml
```

UR5e 모델은 ROS의 `ur_description`을 사용한다. 공식 DAE visual mesh는 빌드할
때 MuJoCo가 읽을 수 있는 OBJ로 자동 변환하며 `ros-humble-ur`가 제공하는 Assimp
런타임(`libassimp5`)을 사용한다.

```bash
source /opt/ros/humble/setup.bash
python3 mujoco_handeye_sim/scripts/prepare_ur5e.py
python3 -m handeye_mujoco build \
  --config mujoco_handeye_sim/configs/ur5e_ljv7080.yaml
```

모델 확인:

```bash
PYTHONPATH="$(pwd)/mujoco_handeye_sim/src${PYTHONPATH:+:$PYTHONPATH}" \
python3 -m handeye_mujoco check \
  --model mujoco_handeye_sim/build/rb5_ljv7080/model.xml
```

## 3. MoveIt 2 workspace 빌드

```bash
./moveit2_ws/build_workspace.sh
```

`real_laser_handeye/initial_T_tcp_sensor.json`을 수정한 경우 위 빌드를 다시
실행한다.

빌드 확인:

```bash
source /opt/ros/humble/setup.bash
source moveit2_ws/install/setup.bash

ros2 pkg prefix rb5_laser_moveit_config
ros2 pkg prefix rbpodo_description
```

## 4. 통합 GUI 실행

```bash
source /opt/ros/humble/setup.bash
source moveit2_ws/install/setup.bash

export PYTHONPATH="$(pwd):$(pwd)/mujoco_handeye_sim/src${PYTHONPATH:+:$PYTHONPATH}"
python -m real_laser_handeye.workflow_gui
```

시작 창에서 `mode = sim | real`, `robot = rb5 | ur5e`를 고른다. `real`일
때만 robot IP와 laser IP가 활성화되며 둘 다 유효한 IP여야 시작할 수 있다.
같은 선택을 CLI로 미리 넘겨 시작 창을 건너뛸 수도 있다.

```bash
# SIM
python -m real_laser_handeye.workflow_gui --mode sim --robot ur5e

# REAL
python -m real_laser_handeye.workflow_gui --mode real --robot rb5 \
  --robot-ip 169.254.186.20 --laser-ip 169.254.186.182
```

연결 직후 `0 Mount check` 탭에서 adapter가 읽은 6축 joint와 TCP position/RPY를
좌측 그래프로 확인할 수 있다. 우측 MuJoCo 모델은 같은 measured joint로 실시간
갱신되며, 모델 FK와 measured TCP의 위치/회전 차이도 표시한다. 이 탭은 로봇
명령을 전혀 보내지 않으므로 `Enable real motion`을 켜지 말고 pendant/freedrive로
천천히 각 관절을 움직여 실제 로봇·센서 장착 방향과 모델을 비교한다.

GUI 실행 순서:

```text
Connect
-> REAL이면 "Enable real motion" 선택 후 zero-displacement 검증
-> Capture Initial Line x 4
-> Estimate Plane
-> Record Stable Pose
-> 반경 입력
-> Generate & Validate
```

Stage 2의 `Route mode`에서 두 순서 결정 방식을 선택할 수 있다.

- `Circular greedy (existing)`: 기록된 START에서 가장 가까운 approach를 첫
  pose로 고르고, 인접 pose의 joint cost로 원주 진행 방향을 정한다.
- `Global order + alpha`: 각 pose의 `alpha`/`alpha+180` collision-aware IK를
  후보로 두고, 최대 단일 관절 변화 제한을 만족하면서 전체 가중 관절 이동량이
  최소인 열린 경로를 동적계획법으로 선택한다. 기본 제한은 90 deg이며
  `planning.global_alpha_max_joint_delta_deg`에서 설정한다.

두 모드 모두 선택한 순서에 대해 아래 실행 경로를 반복한다.

```text
RRTConnect MoveJ -> APPROACH
MoveIt Cartesian -> SCAN -> capture
MoveIt Cartesian -> APPROACH
```

모든 구간은 동일한 MoveIt PlanningScene의 self/sensor/plane/floor collision을
사용한다. 계획이 실패하면 `collision_distance_fallback_step_mm` 간격으로 scan
거리를 늘리고, 필요하면 tilt도 줄여 다시 계획한다. 마지막 scan의 retract가
끝나면 해당 APPROACH에서 종료하며 자동 START 복귀는 하지 않는다.

## 5. REAL ROS driver 자동 실행과 장비 self-check

`Connect`를 누르면 GUI가 입력한 robot IP를 사용해 선택한 공식 ROS 2
driver를 자식 프로세스로 실행하고, GUI 종료 시 함께 정리한다.

- RB5: 공식 `rbpodo_ros2` hardware/description을 사용하고 이 프로젝트의
  `real_driver.launch.py`가 `joint_trajectory_controller`를 활성화한다.
- UR5e: 공식 `ur_robot_driver/ur_control.launch.py`를 실행한다. 해당 IP의
  factory kinematics YAML이 없으면 먼저 `ur_calibration`으로 추출하고 이후
  연결에서는 캐시를 재사용한다. 같은 YAML을 센서 collision geometry가 포함된
  프로젝트 MoveGroup에도 전달해 driver TF와 계획 모델이 같은 kinematics를 쓴다.
- driver 출력은 각 real session의 `ros_driver/driver.log`에 저장된다. UR
  calibration 출력은 같은 디렉터리의 `ur_calibration.log`에 저장된다.

연결 성공은 trajectory action, 6개 joint state, `T_base_tcp`, Keyence 제어/
고속 프로파일 채널이 모두 열렸다는 뜻이다. 실제 이동은 여전히 잠겨 있다.
`Enable real motion`을 선택하고 경고를 확인하면 현재 joint pose와 같은 목표를
저속 trajectory로 한 번 보내 action accept/result와 실제 상태 drift를 확인한
뒤에만 scan/return 동작이 허용된다. `STOP` 또는 재연결 시 이 승인은 취소된다.

RB5의 공식 ros2_control joint state는 현재 reference joint를 내보내므로,
RB5 adapter는 trajectory 명령은 ROS action으로 유지하고 도착 확인 및 TCP는
`rbpodo.CobotData`의 measured `jnt_ang`/`tcp_pos`에서 읽는다.

필수 TF는 RB5 `link0 -> tcp`, UR5e `base_link -> tool0`이다. UR5e driver를
일반 controller로 실행하려면 `initial_joint_controller:=joint_trajectory_controller`
를 지정할 수 있다.

UR5e는 teach pendant에서 로봇 전원/브레이크를 해제하고 **External Control**
프로그램을 시작해야 한다. RB5는 arm initialization 및 E-stop 해제 후 연결한다.
두 장비 모두 최초 `Enable real motion` 검증 시 물리 E-stop을 즉시 누를 수 있는
상태에서 작업 영역을 비우고 수행한다.

## 6. REAL 초기 프로파일 캡처

수동으로 로봇 위치를 교시한 뒤 필요한 횟수만큼 반복 실행한다.

```bash
PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}" \
python3 -m real_laser_handeye.estimate_initial_plane capture \
  --robot-host 169.254.186.20 \
  --laser-ip 169.254.186.182 \
  --handeye real_laser_handeye/initial_T_tcp_sensor.json \
  --dataset-dir runs/real/initial_plane/dataset \
  --max-stationarity-translation-mm 0.2 \
  --max-stationarity-rotation-deg 0.2
```

## 7. 저장된 프로파일로 평면 추정

```bash
PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}" \
python3 -m real_laser_handeye.estimate_initial_plane estimate \
  --handeye real_laser_handeye/initial_T_tcp_sensor.json \
  --dataset-dir runs/real/initial_plane/dataset \
  --estimate-dir runs/real/initial_plane/estimate \
  --no-show-plot
```

## 8. 테스트

```bash
python3 -m compileall -q \
  real_laser_handeye \
  mujoco_handeye_sim/src \
  moveit2_ws/src

PYTHONPATH="$(pwd)/mujoco_handeye_sim/src${PYTHONPATH:+:$PYTHONPATH}" \
python3 -m pytest -q mujoco_handeye_sim/tests
```
