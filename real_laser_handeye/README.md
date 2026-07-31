# real laser hand-eye calibration

## 1. 로봇 및 레이저 연결

현재 장비 IP는 다음과 같다.

- Robot: `169.254.186.20`
- Keyence laser: `169.254.186.182`

`robot_adapter.py`에는 사용하는 로봇에 맞게 `T_base_tcp`(4x4,
translation 단위 mm)를 반환하는 함수를 구현해야 한다.
`laser_adapter.py`에는 profile 데이터를 반환하는 `read_profile` 함수를
구현해야 한다.

## 2. 초기 hand-eye 행렬

초기 `T_tcp_sensor` 4x4 행렬은 다음 파일에 저장한다.

```text
real_laser_handeye/initial_T_tcp_sensor.json
```

## 3. 버튼을 이용한 캡처 및 캘리브레이션

로봇 TCP를 수동으로 이동하면서 레이저 profile을 캡처한 다음,
저장된 capture 전체로 캘리브레이션한다.

```bash
PYTHONPATH=. python3 -m real_laser_handeye.main session \
  --robot-host 169.254.186.20 \
  --laser-ip 169.254.186.182 \
  --profile-diagnostic-after-s 3.0 \
  --initial-transform real_laser_handeye/initial_T_tcp_sensor.json \
  --output runs/real/real_initial/T_tcp_sensor_calibrate_initial_value.csv
```

실행 후 PyQtGraph 모니터 창의 버튼을 사용한다.

- `Capture`: 현재 TCP와 새 laser profile 저장
- `Calibrate (RANSAC)`: 저장된 capture를 RANSAC으로 필터링한 후 캘리브레이션
- `Quit`: 세션 종료 및 하드웨어 연결 해제

캡처 파일은 기본적으로 `runs/real/dataset/capture_*.npz`에 저장된다.


## 4. Stop-and-scan 실행

### 스캔 mode 1: 위치 교시

```bash
 PYTHONPATH=. python3 real_laser_handeye/laser_scan_demo/two_point_stop_and_scan.py \
  --handeye /home/choisuhyun/lvs_HandEyeCalibration/runs/real/real_initial/T_tcp_sensor_calibrate_initial_value.csv \
  --save-path runs/real/two_point_stop_and_scan.npz \
  --robot-host 169.254.186.20 \
  --laser-ip 169.254.186.182 \
  --batch-profiles 1 \
  --waypoint-spacing-mm 1.0 \
  --profiles-per-waypoint 10 \
  --capture-aggregate mean \
  --scan-speed-mm-s 5 \
  --scan-accel-mm-s2 5 \
  --alignment-speed-mm-s 10 \
  --alignment-accel-mm-s2 10 \
  --position-tolerance-mm 0.02 \
  --rotation-tolerance-deg 0.2 \
  --arrival-stable-count 10 \
  --settle-at-waypoint-s 0.7 \
  --max-capture-translation-mm 0.2 \
  --max-capture-rotation-deg 0.5 \
  --move-timeout-s 60 \
  --auto-save \
  --save-on-exit
```

### submode : 평면 추정 및 평면 경계 추정

```bash
PYTHONPATH=. python3 -m real_laser_handeye.main session \
  --robot-host 169.254.186.20 \
  --laser-ip 169.254.186.182 \
  --profile-diagnostic-after-s 3.0 \
  --initial-transform real_laser_handeye/initial_T_tcp_sensor.json
```

### submode : 수동 이동 및 캡처, 스캔용

```bash
PYTHONPATH=. python3 -m real_laser_handeye.main session \
  --robot-host 169.254.186.20 \
  --laser-ip 169.254.186.182 \
  --profile-diagnostic-after-s 3.0 \
  --initial-transform real_laser_handeye/initial_T_tcp_sensor.json \
  --output runs/real/real_initial/T_tcp_sensor_calibrate_initial_value.csv
```

## 5. 캘리브레이션 결과로 월드 좌표 실시간 스캔

```bash
PYTHONPATH=. python3 -m real_laser_handeye.laser_scan_demo.main \
  --robot-host 169.254.186.20 \
  --laser-ip 169.254.186.182 \
  --handeye runs/real/T_tcp_sensor_calibrated.csv
```

```text
s : 현재 제한된 point cloud를 runs/real/live_scan.npz로 저장
c : 누적 point cloud와 TCP 궤적 초기화
q : 종료
```
