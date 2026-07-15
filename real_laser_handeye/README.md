# 실제 로봇–2D 레이저 single-plane 캘리브레이션

이 폴더는 `run_single_plane_optimal_benchmark.py`에서 검증한 실제 실행 구성을 장비 수집까지 연결한다.

- 기본 81 pose: 9개 circular target ray × `(d=60/90/120, theta=30, beta=60/90/120)`
- 관측가능성 보강 24 pose: 동일 패턴에 `theta=60` reference pose
- 해법: `plane_offset_mode=joint`인 linear iteration
- nonlinear refinement: 사용하지 않음
- 기본 동작: dry-run. 명시적인 세 가지 잠금을 모두 풀기 전에는 로봇 명령을 보내지 않음

엄밀한 `theta=30` 81 pose만 사용하면 unknown plane offset과 TCP–sensor translation 한 방향의 rank가 분리되지 않는다. 그래서 plan 단계가 기본적으로 이를 거부하며, 24개의 `theta=60` reference를 포함한 105 pose가 실제 캘리브레이션 기본값이다. `joint` solver는 이 관측성을 만들어내는 것이 아니라, 이미 관측 가능한 translation과 plane offset을 같은 선형계에서 함께 푼다.

## 좌표계 규약

모든 길이는 mm이고 `T_A_B`는 B 좌표를 A 좌표로 변환한다.

```text
T_base_sensor = T_base_tcp @ T_tcp_sensor
```

캘리브레이션 중 로봇에 설정된 TCP를 바꾸면 안 된다. 로봇 API가 flange pose를 주거나 m/rad 단위를 쓰면 adapter에서 반드시 위 규약의 `T_base_tcp`, mm, proper rotation matrix로 바꿔야 한다.

## 장비 연결부

1. [robot_adapter_template.py](robot_adapter_template.py)를 로봇 모델명으로 복사한다.
2. 아래 두 함수에 제조사 SDK 호출을 넣는다.
   - `current_T_base_tcp()`: 현재 TCP의 4×4 transform 반환
   - `move_tcp(...)`: 명령 TCP로 이동하고 컨트롤러가 정지한 뒤 반환
3. `connect`, `stop`, `close`에 제조사 연결/정지/해제를 넣고, 별도 통신 경로의 `stop()`이 blocking `move_tcp()`를 실제로 끊을 수 있을 때만 `supports_independent_stop()`이 `True`를 반환하게 한다.
4. 기본 설정의 `require_controller_collision_check=true`를 사용할 때는 `controller_path_is_safe`에 controller IK, joint-limit, full-link collision query를 반드시 연결하고, 검사한 motion mode와 `move_tcp`의 motion mode를 동일하게 유지한다.

템플릿의 Euler 변환은 extrinsic XYZ degree 가정이다. 로봇이 ZYX, rotation vector, quaternion을 쓰면 변환 코드를 해당 규약으로 변경한다.

Keyence LJ-X는 [keyence_adapter.py](keyence_adapter.py)가 번들된 `libljxacom.so`를 필요할 때만 로드한다. IP, control/high-speed port, batch 수와 median 집계는 JSON에서 설정한다.

## 안전 설정

[real_config.example.json](real_config.example.json)을 복사한다.

```bash
cp real_laser_handeye/real_config.example.json real_laser_handeye/real_config.json
```

실제 셀을 측정한 값으로 다음 항목을 반드시 수정한다.

- `workspace_mm`: TCP가 허용되는 보수적인 AABB
- `no_go_boxes_mm`: 지그, 테이블, 평면 뒤쪽 등 TCP 금지 AABB
- `safe_transit_T_base_tcp`: teach pendant로 검증한 4×4 안전 대기 pose
- `initial_handeye_uncertainty_mm/deg`: 초기 hand-eye 오차의 상한
- 속도, step, 접근 거리와 readback tolerance
- `max_final_plane_rms_mm`: 수렴 여부와 별개로 최종 결과를 저장/활성화할 수 있는 plane-fit RMS 상한
- robot adapter 경로/주소

TCP 경로 검사는 로봇 링크, 케이블, elbow의 충돌을 수학적으로 보장하지 못한다. 제조사 controller의 joint limit, singularity, collision monitoring, safety PLC를 활성화하고 저속 T1/teach mode에서 첫 경로를 검증해야 한다. 첫 검증이 끝나기 전에는 `live_enabled=false`를 유지한다.

plan에는 전체 내용의 SHA-256 `plan_id`와 검토 당시 safety snapshot이 저장된다. 실행 설정이나 plan 내용이 바뀌거나 기존 dataset이 다른 plan에서 만들어졌다면 수집/resume을 거부하므로 plan을 다시 생성해 검토해야 한다. no-go 검사는 waypoint뿐 아니라 각 TCP 선분과 AABB의 교차도 확인한다.
현재 bootstrap quality metadata가 없거나 현재 config의 RMS/span/센서 거리 한계를 통과하지 못한 boundary로는 UI와 CLI 모두 live motion plan을 생성하지 않는다. boundary에는 평면을 재구성할 때 사용한 초기 `T_tcp_sensor`도 저장되며 plan 입력 hand-eye와 정확히 대조한다. 이 품질/provenance 증거는 plan ID 안에 포함되고 live session 시작 때 다시 검사되므로 구버전 plan으로 우회할 수 없다.

## 실시간 UI

Tkinter와 Matplotlib 기반 UI는 추가 pip 패키지 없이 실행된다. Ubuntu에서 Tk가 빠져 있다면 OS 패키지 `python3-tk`가 필요하다.

```bash
python3 -m real_laser_handeye.run_calibration_ui \
  --config real_laser_handeye/real_config.json \
  --handeye initial_T_tcp_sensor.csv \
  --plane-boundary runs/bootstrap/plane_boundary.json \
  --plan runs/motion_plan.json \
  --dataset-dir runs/dataset \
  --output runs/T_tcp_sensor_calibrated.csv
```

UI는 다음 네 화면을 제공한다.

- `Live / Capture`: 실시간 sensor X–Z profile, 직전 또는 계획표에서 선택한 이전 profile, 다음 `d/theta/beta`, target TCP 행렬과 XYZ/RPY
- `Accumulated plane / 3D plan`: 전체 TCP 계획, 완료 pose, 다음 safe-transit→approach→target 경로, TCP/센서 XYZ 축, workspace/no-go box, bootstrap 경계, 누적 base-frame point cloud와 provisional plane/RMS
- `Capture plan`: 105개 scan의 pending/next/captured 상태와 line, parameter, reference 여부
- `Calibration`: joint-linear solver 실행, 최종 `T_tcp_sensor`, rank/condition/RMS, convergence history

`Connect preview`는 로봇 이동을 전혀 명령하지 않고 센서 profile과 현재 TCP capture 기능만 연결한다. UI bootstrap은 teach pendant로 수동 이동한 네 위치에서 `Capture current bootstrap view`를 누른 뒤 `Finalize 4 views / plane`을 실행한다.

장비 없이 화면과 전체 scan-by-scan 흐름을 먼저 확인할 수 있다.

```bash
python3 -m real_laser_handeye.run_calibration_ui --mock-demo
```

이후 `Generate 105-pose plan` → `Connect preview`를 누르고 acknowledgement에 `I_CHECKED_THE_ROBOT_CELL`을 입력한다. `Capture next 1`은 mock robot으로 한 scan만 수행하며, `Start automatic`은 전체 계획을 수행한다. mock 결과는 기본적으로 `runs/mock_ui` 아래에 저장된다.

실제 자동 이동은 CLI와 동일하게 다음 조건을 모두 통과해야 한다.

- `live_enabled=true`
- `controller_collision_check_acknowledged=true`
- `controller_path_is_safe(...) == True` (`require_controller_collision_check=true`일 때)
- acknowledgement 입력란에 `I_CHECKED_THE_ROBOT_CELL`
- `return_to_safe_between_scans=true` — UI pause가 항상 capture→retreat→safe-transit 뒤에 적용되도록 강제
- robot adapter의 `supports_independent_stop()==true` — UI thread와 별개로 blocking 이동을 중단할 수 있어야 함

`Capture next 1`은 한 scan만 안전 경로로 수집하고, `Start automatic`은 나머지를 진행한다. `Pause after safe return`은 현재 scan을 저장하고 안전 대기점으로 복귀한 뒤 멈춘다. `SOFTWARE STOP`은 취소 flag와 robot adapter의 controlled stop을 호출하지만 물리 E-stop과 safety PLC를 대체하지 않는다.

실시간 표시와 저장 capture는 센서를 동시에 호출하지 않는다. 단일 `ProfileBroker`만 Keyence SDK 전역 버퍼를 소비하며, UI는 latest immutable frame을 읽고 capture는 로봇 settle 이후 생성된 새 sequence를 기다린다.
profile 오류와 frame age는 sequence가 갱신되지 않아도 `ERROR/STALE`로 표시된다. 누적 CSV 로드와 provisional plane SVD는 background snapshot worker에서 수행하고, Tk/Matplotlib 갱신만 UI thread에서 수행한다.

## 실행 순서

### 1. 평면 bootstrap

자동 이동 전에 평면 위치를 알 수 없으므로 이 단계만은 teach pendant로 네 위치를 수동 jog한다. 네 profile이 평면의 사용할 영역을 넓게 둘러싸도록 하고 각 위치에서 Enter를 누른다.

```bash
python3 -m real_laser_handeye.run_real_single_plane_calibration bootstrap \
  --config real_laser_handeye/real_config.json \
  --handeye initial_T_tcp_sensor.csv \
  --output-dir runs/bootstrap \
  --margin-mm 20
```

출력 `runs/bootstrap/plane_boundary.json`에는 초기 hand-eye로 재구성한 plane frame, fit RMS와 안전 UV 경계가 저장된다. `capture.max_bootstrap_plane_rms_mm`, `min_bootstrap_span_mm`, `min_bootstrap_sensor_plane_distance_mm` 품질 검사를 모두 통과한 경우에만 파일을 저장한다. 거부되면 네 view를 더 넓게 분산해 다시 측정하고 단위, TCP, profile 축 및 평면만 측정했는지 확인한다.

### 2. 105 pose 계획과 정적 검사

이 명령은 장비에 연결하지 않고 `motion_plan.json`과 사람이 검토하기 쉬운 `motion_plan.csv`를 만든다. CSV에는 각 scan의 approach/target TCP가 mm와 extrinsic XYZ Euler degree로 기록되므로 controller simulation에 넣어 확인할 수 있다. 실제 명령은 Euler CSV를 다시 읽지 않고 JSON의 4×4 행렬을 사용한다.

```bash
python3 -m real_laser_handeye.run_real_single_plane_calibration plan \
  --config real_laser_handeye/real_config.json \
  --plane-boundary runs/bootstrap/plane_boundary.json \
  --handeye initial_T_tcp_sensor.csv \
  --heights-mm 60 90 120 \
  --theta-deg 30 \
  --beta-deg 60 90 120 \
  --reference-scans 24 \
  --reference-theta-deg 60 \
  --reference-heights-mm 60 90 120 \
  --reference-beta-deg 60 90 120 \
  --pose-geometry paper_incidence \
  --output runs/motion_plan.json
```

계획 생성 시 다음을 검사한다.

- translation/plane-offset matrix rank 4와 condition
- 모든 target/approach/interpolated TCP의 workspace 및 no-go box
- 초기 hand-eye 불확실성을 뺀 sensor–plane clearance
- target line endpoint가 bootstrap 안전 경계 안에 있는지
- 모든 transform이 유효한 SE(3)인지

엄밀한 논문 81 pose가 왜 거부되는지 확인만 하려면 `--reference-scans 0 --allow-unobservable`을 쓸 수 있다. 이 옵션으로 만든 데이터도 `calibrate` 단계는 의도적으로 거부한다.

### 3. dry-run 확인

```bash
python3 -m real_laser_handeye.run_real_single_plane_calibration collect \
  --config real_laser_handeye/real_config.json \
  --plan runs/motion_plan.json \
  --dataset-dir runs/dataset
```

`--execute`가 없으므로 로봇/레이저 연결과 이동은 전혀 발생하지 않는다.

### 4. 실제 수집

teach pendant/controller simulation으로 `safe_transit → approach → target → retreat → safe_transit` 경로를 확인한 후에만:

1. JSON의 `live_enabled=true`
2. full-link controller 검사를 확인하고 `controller_collision_check_acknowledged=true`
3. 아래 명시적 acknowledgement 전달

```bash
python3 -m real_laser_handeye.run_real_single_plane_calibration collect \
  --config real_laser_handeye/real_config.json \
  --plan runs/motion_plan.json \
  --dataset-dir runs/dataset \
  --execute \
  --acknowledge-risk I_CHECKED_THE_ROBOT_CELL
```

각 target에서 이동 종료와 settle을 기다린 뒤 전/후 TCP가 정지 tolerance 안인지 확인하고 fresh profile을 저장한다. profile을 초기 평면으로 재구성한 RMS가 제한을 넘으면 즉시 controlled stop을 호출한다. profile/TCP 파일과 `manifest.json`은 임시 파일 후 replace 방식으로 저장하며, 동일 plan ID의 명령으로만 완료된 scan을 건너뛰고 재개할 수 있다.

### 5. nonlinear refinement 없는 캘리브레이션

```bash
python3 -m real_laser_handeye.run_real_single_plane_calibration calibrate \
  --dataset-dir runs/dataset \
  --handeye initial_T_tcp_sensor.csv \
  --output runs/T_tcp_sensor_calibrated.csv \
  --max-iter 30 \
  --tol 1e-9 \
  --max-final-plane-rms-mm 2.0
```

결과와 함께 `T_tcp_sensor_calibrated.diagnostics.json`에 실제 TCP pose의 관측가능성, rank/condition history, 초기/최종 plane RMS, iteration 및 plane offset이 저장된다.
solver가 수렴했더라도 최종 plane RMS가 acceptance limit를 넘으면 diagnostics만 저장하고 transform CSV는 쓰거나 UI에서 활성화하지 않는다. 이 self-fit gate는 잘못된 결과를 거르는 최소 조건이며, 별도 검증 pose/치수로 최종 hand-eye 정확도를 확인해야 한다.

첫 joint-linear 결과의 self-fit plane RMS가 기본 1 mm를 넘을 때만 초기 회전을 각 축 ±30°로 바꾼 여섯 linear/PCA start를 추가 실행하고, 수렴한 결과 중 RMS가 가장 작은 것을 고른다. 이는 benchmark의 conditional linear multistart이며 nonlinear refinement가 아니다. `--no-linear-multistart`로 끌 수 있고 threshold/각도는 `--linear-multistart-threshold-mm`, `--linear-multistart-angle-deg`로 조정한다.

bootstrap이 이미 끝났다면 plan–collect–calibrate를 한 명령으로 실행할 수도 있다. `--execute`가 없으면 plan까지만 수행한다.

```bash
python3 -m real_laser_handeye.run_real_single_plane_calibration run \
  --config real_laser_handeye/real_config.json \
  --plane-boundary runs/bootstrap/plane_boundary.json \
  --handeye initial_T_tcp_sensor.csv \
  --work-dir runs/calibration_001
```

## 테스트

장비 SDK를 로드하지 않는 mock robot/planar laser로 105 pose 수집부터 joint calibration, scan 단위 resume/stop, 단일 profile broker, 누적 평면 모델과 headless UI renderer까지 검사한다.

```bash
PYTHONPATH=robust_laser_handeye:. python3 -m pytest -q \
  robust_laser_handeye/tests real_laser_handeye/tests
```

현재 결과: `29 passed`.
