# MuJoCo RB5–Keyence 단일 평면 hand-eye 환경

MuJoCo 환경은 실물 수동 캡처 UI와 분리된 CLI로 유지한다. 자동 bootstrap,
충돌 필터, preflight, simulated capture와 calibration을 한 번에 실행할 수 있다.

```bash
python3 -m mujoco_laser_handeye \
  --work-dir runs/mujoco_rb5_auto_bootstrap
```

장면과 scan 파라미터는
[`config/rb5_keyence_single_plane.json`](config/rb5_keyence_single_plane.json)
및 CLI 옵션에서 변경한다. `--viewer`를 지정하면 native MuJoCo viewer를 연다.

이 패키지는 자체 `compat` 모듈의 simulation 수집 인터페이스를 사용해
RB5-850E, Keyence 2D 레이저 프로파일 센서, 단일 평면 타겟을 MuJoCo에서
검증한다. 결과 dataset은 기존 simulation 형식인
`profile CSV + T_base_tcp CSV + manifest.json` 형식이므로 기존 joint-linear
single-plane solver에 그대로 들어간다.

기본 계획은 `analyze_single_dual_theta_gt_jacobian.py`의 결론을 반영한다.

- main: 9 line × `d=60/90/120`, `theta=30`, `beta=60/90/120` = 81 pose
- reference: `theta=60` pose 24개
- 이론 계획: 총 105 pose, translation/plane-offset rank 4
- MuJoCo 필터: IK, 관절 제한, self collision, 센서/타겟/테이블 충돌을 통과한
  pose만 유지하고 rank 4를 다시 확인

경로계획 전에 RB5가 네 개 bootstrap view로 자동 이동하고 noisy Keyence
profile을 수집한다. 평면과 안전 경계는 이 profile, TCP readback 및 초기
hand-eye만으로 추정하며 미리 만든 boundary를 사용하지 않는다.

기본 셀에서는 105개 중 48개가 안전 경로로 남고(환경/버전에 따라 재검증),
9개 target line과 `theta=30/60`이 모두 유지된다. 거부된 pose와 접촉 링크는
`collision_filter_report.json`에 남는다.

## 구현 범위

- Rainbow Robotics 공식 `rbpodo_ros2`의 RB5-850E 관절 원점, 질량·관성과
  collision STL 사용
- damped least-squares + bounded least-squares RB5 IK
- 각 Cartesian waypoint 사이를 2 deg 이하 joint step으로 다시 나눠 충돌 검사
- RB5 self collision과 Keyence housing–robot/target/table/fixture 충돌 검사
- Keyence 형식의 sensor-frame `[x, 0, z]` profile, X/Z noise, dropout,
  측정 Z 범위 및 타겟 경계 clipping
- 충돌 검사된 4-view bootstrap 자동 이동·캡처 및 실제 workflow의 평면/경계 추정
- headless 실행과 MuJoCo interactive viewer
- bootstrap → theoretical plan → collision filter → full preflight → capture → calibration
  일괄 실행

RB5 자산 출처와 Apache-2.0 라이선스는
[`assets/rb5_850e/NOTICE.md`](assets/rb5_850e/NOTICE.md) 및
[`assets/rb5_850e/LICENSE`](assets/rb5_850e/LICENSE)에 있다.

## 설치와 실행

저장소 루트에서:

```bash
python3 -m pip install -r requirements.txt

python3 -m mujoco_laser_handeye \
  --work-dir runs/mujoco_rb5_auto_bootstrap
```

bootstrap만 먼저 확인하려면:

```bash
python3 -m mujoco_laser_handeye \
  --work-dir runs/mujoco_rb5_bootstrap \
  --bootstrap-only
```

기존에 검토한 boundary를 재사용할 때만 명시적으로 지정한다.

```bash
python3 -m mujoco_laser_handeye \
  --plane-boundary path/to/plane_boundary.json \
  --work-dir runs/mujoco_rb5_existing_boundary
```

화면을 보면서 확인하려면 Linux desktop의 `DISPLAY`가 있는 터미널에서:

```bash
python3 -m mujoco_laser_handeye \
  --work-dir runs/mujoco_rb5_viewer \
  --viewer \
  --realtime-scale 0.05
```

충돌 필터와 preflight까지만 수행하려면:

```bash
python3 -m mujoco_laser_handeye \
  --work-dir runs/mujoco_rb5_preflight \
  --preflight-only
```

필터가 없을 때 어떤 이론 pose에서 실패하는지 진단하려면 다음 옵션을 쓸 수
있다. 기본 105 pose에는 의도적으로 충돌 경로가 있으므로 이 명령은 첫 실패에서
종료하고 `preflight_report.json`에 원인을 기록한다.

```bash
python3 -m mujoco_laser_handeye \
  --work-dir runs/mujoco_rb5_unfiltered \
  --preflight-only \
  --no-filter-unsafe
```

## 주요 출력

```text
runs/mujoco_rb5_auto_bootstrap/
├── bootstrap/
│   ├── bootstrap_motion_plan.json      # GT 평면으로 제안한 simulation-only view
│   ├── bootstrap_pose_1..4.csv
│   ├── bootstrap_profile_1..4.csv
│   ├── bootstrap_manifest.json
│   ├── bootstrap_report.json
│   └── plane_boundary.json             # profile+초기 hand-eye로 추정한 계획 입력
├── motion_plan_theoretical.json/.csv   # 추정 boundary 기반 105개 기하학 계획
├── collision_filter_report.json       # scan별 PASS/REJECT와 충돌 이유
├── motion_plan_filtered.json/.csv      # 실제 수집에 사용한 안전 계획
├── preflight_report.json               # 전체 안전 계획 재실행 결과/관절 범위
├── dataset/                            # real_laser_handeye 호환 dataset
├── T_tcp_sensor_calibrated.csv
├── T_tcp_sensor_calibrated.diagnostics.json
└── simulation_summary.json             # GT translation/rotation error 포함
```

기본 seed의 현재 검증 결과는 bootstrap RMS `0.603 mm`, 평면 normal GT 오차
`0.065 deg`, 안전한 calibration scan 48개, 최종 plane RMS `0.029 mm`,
ground-truth hand-eye 오차 `0.0028 mm / 0.0016 deg`다. 수치는 MuJoCo 버전,
IK tolerance, noise와 필터 결과에 따라 달라진다.

자동 bootstrap에서 GT 평면은 로봇이 바라볼 네 위치를 제안하는 용도로만
사용한다. 저장되는 `plane_boundary.json`의 point나 plane fit에는 GT를 넣지
않는다. 실제 장비에서는 이 자동 포즈 제안 대신 기존 workflow처럼 작업자가
teach pendant로 네 위치를 정해야 한다.

## 셀과 센서 모델 수정

[`config/rb5_keyence_single_plane.json`](config/rb5_keyence_single_plane.json)의
다음 값을 실제 셀 실측값으로 바꾼다.

- `robot.kwargs.T_tcp_sensor`: 시뮬레이션 ground truth mount
- `plane_center_mm`, `plane_rpy_deg`, `plane_size_mm`
- `table_center_mm`, `table_size_mm`, `obstacles`
- `sensor_housing_size_mm`, `sensor_housing_center_z_mm`
- `collision_margin_mm`: geom 하나의 margin이며 기본 2 mm끼리의 pair는
  최대 4 mm 이내에서 보수적으로 contact 후보가 된다
- `laser.kwargs.x_min_mm/x_max_mm`, `z_min_mm/z_max_mm`, point 수와 noise
- `safety.workspace_mm`, `no_go_boxes_mm`, `safe_transit_T_base_tcp`

평면 위치/크기를 바꾸면 자동 bootstrap이 새 `plane_boundary.json`을 다시
만드므로 기본 실행에서는 예제 boundary를 수정할 필요가 없다. 초기 hand-eye는
[`config/initial_T_tcp_sensor.csv`](config/initial_T_tcp_sensor.csv)에서 바꾸며,
simulation ground truth는 JSON의 `robot.kwargs.T_tcp_sensor`로 분리되어 있다.
새 boundary의 `bootstrap_provenance.T_tcp_sensor_init`도 자동으로 그 값을
기록한다. `--plane-boundary`로 기존 파일을 재사용할 때만 두 값이 정확히
일치해야 한다.

현재 Keyence 모델은 LJ-X 계열의 좌표 규약을 구현한 파라미터식 모델이다.
사용할 정확한 헤드 모델(예: LJ-X8060/8080 등)이 정해지면 데이터시트의
reference distance, X range, Z range, housing 치수로 위 값을 교체해야 한다.

## 테스트

```bash
PYTHONPATH=robust_laser_handeye:. python3 -m pytest -q \
  mujoco_laser_handeye/tests
```

## 안전 한계

이 환경은 실제 로봇 안전 인증이나 controller simulation을 대체하지 않는다.
공식 충돌 메쉬를 쓰지만 케이블, 커넥터, 설치 브래킷, 공구 공차, 사람과 실제
지그는 설정에 넣은 것만 검사한다. MuJoCo IK도 RB controller의 singularity,
torque, safety PLC 및 MoveL 구현과 동일하지 않다. 실기 전에는 필터된 CSV를
RB controller simulation에 넣고, 실제 Keyence CAD와 셀 CAD를 반영한 후,
저속 T1/teach mode와 물리 E-stop 하에서 다시 검증해야 한다.
