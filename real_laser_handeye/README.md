# real laser hand-eye calibration

## 1. 로봇 및 레이저 연결

현재 장비 IP는 다음과 같다.

- Robot: `169.254.186.20`
- Keyence laser: `169.254.186.182`

사용하는 장비가 달라진다면 각 adapter에 사용 장비에 맞게 함수를 다시 구현해야한다.

## 2. 초기 hand-eye 행렬

초기 `T_tcp_sensor` 4x4 행렬은 다음 파일에 저장한다.

```text
real_laser_handeye/initial_T_tcp_sensor.json
```

## 3. 캘리브레이션
### 3.1 수동 TCP 조작 및 캘리브레이션

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

## 4. validation을 위한 스캔

### 4.1 Stop-and-scan | 위치 교시 기반의 only translation 스캔, 각 step에서 정지 후 스캔을 진행

```bash
PYTHONPATH=. python3 real_laser_handeye/laser_scan_demo/two_point_stop_and_scan.py \
  --handeye /home/choisuhyun/lvs_HandEyeCalibration/runs/real/real_initial/T_tcp_sensor_calibrate_initial_value.csv \
  --save-path runs/real/two_point_stop_and_scan.npz \
  --robot-host 169.254.186.20 \
  --laser-ip 169.254.186.182 \
  --batch-profiles 1 \
  --scan-speed-mm-s 5 \
  --scan-accel-mm-s2 5 \
  --alignment-speed-mm-s 10 \
  --alignment-accel-mm-s2 10 \
  --waypoint-spacing-mm 1.0 \
  --profiles-per-waypoint 10 \
  --capture-aggregate mean \
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
