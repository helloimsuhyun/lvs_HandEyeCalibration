# Phase 2A: initialization robustness

Phase 1에서 N별 `rank_in_N` 상위 비율의 random subset을 가져와, GT
hand-eye·plane·candidate bank·solver는 고정하고 초기값과 measurement noise만
변화시킵니다.

```text
Phase 1 subset 고정
× random initial-error environments
× paired candidate-keyed noise repeats
→ iterative joint-offset
→ hand-eye + plane joint nonlinear refinement
```

`1 mm / 0.1 deg` 기준은 classification label일 뿐입니다. Threshold 밖의
결과와 solver failure도 모두 저장합니다. Phase 1의 `promoted/` 파일은
사용하지 않으므로 absolute threshold 통과 subset이 없는 N도 평가됩니다.

## 실행

저장소의 `robust_laser_handeye` 디렉터리에서 먼저 smoke test를 실행합니다.

```bash
PYTHONPATH=. python3 experiments/lhs_initialization_robustness/run.py \
  --config experiments/lhs_initialization_robustness/config.json \
  --smoke \
  --output-dir /tmp/lhs_phase2a_smoke
```

본 실험은 tmux에서 실행합니다.

```bash
cd /home/choisuhyun/lvs_HandEyeCalibration/robust_laser_handeye
tmux new -s lhs_phase2a

PYTHONUNBUFFERED=1 PYTHONPATH=. \
  python3 experiments/lhs_initialization_robustness/run.py \
  --config experiments/lhs_initialization_robustness/config.json \
  2>&1 | tee lhs_phase2a_console.log
```

Detach는 `Ctrl-b`, `d`, 재접속은 다음 명령입니다.

```bash
tmux attach -t lhs_phase2a
```

각 N이 끝날 때 `per_n/Nxxx/`의 run CSV, subset summary와 plot을 즉시
저장하고 그 다음 `manifest.json`의 `completed_scan_counts`를 갱신합니다.

## 완료 결과의 geometry 관측

Phase 2A가 모두 끝난 후 Phase 1 candidate transform과 Phase 2A calibration
결과만 사용해 survivor geometry를 탐색적으로 비교할 수 있습니다. 이 분석은
새 subset을 만들거나 선택하지 않으며 Phase 2B 데이터도 사용하지 않습니다.

```bash
PYTHONPATH=. python3 \
  experiments/lhs_initialization_robustness/analyze_phase2a_geometry.py \
  --phase1-dir results/phase1_lhs_screening \
  --phase2a-dir results/phase2a_initialization_robustness \
  --output-dir results/phase2a_geometry_observation
```

동일한 출력 위치에 분석을 다시 생성할 때만 `--overwrite`를 추가합니다.

## 완료 결과의 Jacobian 해석

새 calibration을 실행하지 않고, 동결된 Phase 2A label 및 성능 결과에
Phase 1 GT에서 계산한 joint Jacobian 정보를 결합합니다. 센서 좌표계의 평면
법선 `b_i = R_BS_i^T n`, `G_b`, `Var(cos(theta))`, 그리고 plane nuisance를
Schur complement로 제거한 `H_eff`를 N별 survivor/non-survivor와 비교합니다.

```bash
PYTHONPATH=. python3 \
  experiments/lhs_initialization_robustness/analyze_phase2a_jacobian.py \
  --phase1-dir results/phase1_lhs_screening \
  --phase2a-dir results/phase2a_initialization_robustness \
  --output-dir results/phase2a_jacobian_interpretation
```

주 지표인 `scaled_heff_*`는 joint nonlinear solver와 같은 1도/1 mm column
scaling을 사용합니다. `physical_heff_*`는 radian/mm 물리 단위를 그대로 둔
보조 지표입니다. 동일 출력 위치를 다시 생성할 때만 `--overwrite`를
추가합니다.

## Plane nuisance information-loss 분해

완료된 Jacobian 해석을 plane offset loss와 plane-normal loss로 더 분해합니다.
기존 600 subsets와 검증된 analytic Jacobian만 사용하며 pose 생성과
calibration은 실행하지 않습니다. Centered nuisance column space와 기존
Schur complement가 일치하지 않으면 통계를 저장하기 전에 중단합니다.
Translation Schur block의 trace(T), minimum eigenvalue(E), logdet(D)와 실제
Phase 2A success의 N별 Spearman 비교도 함께 저장합니다.

```bash
PYTHONPATH=. python3 \
  experiments/lhs_initialization_robustness/analyze_phase2a_jacobian_decomposition.py \
  --phase1-dir results/phase1_lhs_screening \
  --phase2a-dir results/phase2a_initialization_robustness \
  --output-dir results/phase2a_jacobian_decomposition
```

동일 출력 위치를 다시 생성할 때만 `--overwrite`를 추가합니다.
