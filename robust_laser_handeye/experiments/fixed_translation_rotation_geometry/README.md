# Fixed-translation rotation-geometry experiment

N=9과 정확히 동일한 `b_i = R_BS_i.T @ n`을 유지하면서 `gamma`, `u,v`,
`d`만 바꾸는 controlled experiment입니다. 새로운 optimizer나 subset search는
사용하지 않습니다.

단계별 실행:

```bash
PYTHONPATH=. python3 experiments/fixed_translation_rotation_geometry/run.py \
  --config experiments/fixed_translation_rotation_geometry/config.json --stage r0

PYTHONPATH=. python3 experiments/fixed_translation_rotation_geometry/run.py \
  --config experiments/fixed_translation_rotation_geometry/config.json --stage geometry

# R1/R2 geometry를 본 뒤 필요할 때만
PYTHONPATH=. python3 experiments/fixed_translation_rotation_geometry/run.py \
  --config experiments/fixed_translation_rotation_geometry/config.json --stage r3

PYTHONUNBUFFERED=1 PYTHONPATH=. python3 \
  experiments/fixed_translation_rotation_geometry/run.py \
  --config experiments/fixed_translation_rotation_geometry/config.json \
  --stage calibration 2>&1 | tee fixed_translation_rotation_geometry.log

PYTHONPATH=. python3 experiments/fixed_translation_rotation_geometry/run.py \
  --config experiments/fixed_translation_rotation_geometry/config.json --stage analysis
```

Calibration은 design 하나(90 runs)가 끝날 때마다 상세 run과 summary CSV에
append됩니다. 중단 후 calibration command에 `--resume`을 붙이면 완료 design을
건너뜁니다. 기존 Phase 1/2A 결과는 읽기만 하며 덮어쓰지 않습니다.

## 완료된 discovery run

기본 config의 R0/R1/R2/R3와 nonlinear calibration 5,760회가 완료되어
`results/fixed_translation_rotation_geometry/`에 저장되어 있습니다.

- `report.md`: 수치 결과와 해석
- `calibration_runs.csv`: 실패를 포함한 전체 run
- `calibration_summary.csv`: 64개 design/shift 요약
- `solver_basin_summary.csv`: 저오차/90도 초과 branch와 residual 비교
- `paired_rotation_statistics.csv`: 공통 init/noise scenario paired 검정
- `r3_paired_contrasts.csv`: u/v와 distance-diversity 직접 대비
- `plots/`: R0/R1/R2/R3 및 geometry-error 그림

기존 결과를 실수로 덮어쓰지 않도록 각 stage는 이미 존재하는 주요 산출물이
있으면 중단합니다. 새 실험은 `config.json`의 `output_dir`을 바꿔 실행합니다.
