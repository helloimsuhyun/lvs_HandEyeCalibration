# LHS random subset screening

고정된 single-plane, GT hand-eye와 초기 오차에서 다음 1차 실험만 수행합니다.

1. Plane-relative 6-DoF LHS candidate bank 생성
2. 각 scan 수 `N`에서 독립적인 random subset calibration
3. 성공한 subset 중 실제 calibration error 상위 비율 저장

Solver는 iterative unknown-plane calibration 결과를 초기값으로 사용하는
joint nonlinear refinement입니다. D-optimal이나 1-exchange는 사용하지
않습니다.

`classification`의 translation/rotation threshold는 promotion label에만
사용합니다. Threshold를 넘거나 solver가 실패한 run과 subset도
`calibration_runs.csv`와 `subset_summary.csv`에 모두 저장합니다.

저장소의 `robust_laser_handeye` 디렉터리에서 실행합니다.

```bash
PYTHONPATH=. python3 experiments/lhs_random_subset_screening/run.py \
  --config experiments/lhs_random_subset_screening/config.json
```

작은 end-to-end 확인:

```bash
PYTHONPATH=. python3 experiments/lhs_random_subset_screening/run.py \
  --config experiments/lhs_random_subset_screening/config.json \
  --smoke \
  --output-dir /tmp/lhs_subset_screening_smoke
```

## tmux에서 본 실험 실행

장시간 실행 전에 먼저 위의 smoke test를 권장합니다. 본 실험은 다음처럼
별도 tmux 세션에서 실행합니다.

```bash
cd /home/choisuhyun/lvs_HandEyeCalibration/robust_laser_handeye
tmux new -s lhs_screening
```

tmux 안에서 실행하고 콘솔 출력도 파일로 보존합니다.

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. \
  python3 experiments/lhs_random_subset_screening/run.py \
  --config experiments/lhs_random_subset_screening/config.json \
  2>&1 | tee lhs_screening_console.log
```

실행을 유지한 채 tmux에서 빠져나오려면 `Ctrl-b`를 누른 다음 `d`를
누릅니다. 세션과 재접속 명령은 다음과 같습니다.

```bash
tmux ls
tmux attach -t lhs_screening
```

다른 터미널에서 진행 로그만 볼 수도 있습니다.

```bash
tail -f /home/choisuhyun/lvs_HandEyeCalibration/robust_laser_handeye/lhs_screening_console.log
```

실험을 중단하려면 tmux 세션 안에서 `Ctrl-c`를 누릅니다. 세션 자체를
종료해야 할 때만 다음 명령을 사용합니다.

```bash
tmux kill-session -t lhs_screening
```

실행기는 비어 있지 않은 기존 output directory를 덮어쓰지 않습니다.
재실행할 때는 `config.json`의 `output_dir`을 새 경로로 변경합니다.

각 N이 끝나면 전체 실험 종료를 기다리지 않고 결과와 plot을 즉시
저장합니다.

```text
<output_dir>/per_n/N005/
├── calibration_runs.csv
├── subset_summary.csv
└── plots/
    ├── error_distributions.png
    └── translation_vs_rotation.png
```

Threshold를 통과하지 못한 run과 subset도 위 CSV에서 제거하지 않습니다.
`manifest.json`의 `completed_scan_counts`에는 디스크 저장까지 완료된 N만
기록됩니다.

상대 pose 파라미터의 순서는 항상
`[u_mm, v_mm, d_mm, tilt_deg, azimuth_deg, normal_azimuth_sensor_deg]`입니다.
마지막 값은 일반적인 sensor roll이 아니라
`atan2((R_BS^T n)_y, (R_BS^T n)_x)`로 정의한 sensor-XY 평면의 plane-normal
방위각입니다. CSV는 사람이
확인하기 위한 파일이고, 후속 실험은 `promoted/promoted_N*.npz`를 입력으로
사용합니다.

이 convention은 `plane_normal_azimuth_sensor_v2`로 저장됩니다. 기존
`results/phase1_lhs_screening`은 legacy roll convention 결과이므로 수정하지
않으며, 기본 config는 새 결과를
`results/phase1_lhs_screening_normal_azimuth_v2`에 생성합니다.
