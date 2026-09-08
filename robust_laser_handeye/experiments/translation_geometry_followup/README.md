# Translation geometry follow-up

Part 1은 기존 N=9 Stage 2 CSV만 사용해 trace와 log-condition의 formal
subset-level effect decomposition을 수행합니다.

```bash
PYTHONPATH=. python3 \
  experiments/translation_geometry_followup/analyze_existing_stage2.py \
  --config experiments/translation_geometry_followup/config.json
```

Part 2는 새로운 LHS candidate bank, N=11 subsets, initialization bank와 noise
streams를 사용하는 독립 replication입니다. Geometry가 강화된 matching 기준을
통과하기 전에는 calibration이 차단됩니다.

```bash
PYTHONPATH=. python3 \
  experiments/translation_geometry_followup/run_fresh_n11_replication.py \
  --config experiments/translation_geometry_followup/config.json \
  --stage geometry

PYTHONUNBUFFERED=1 PYTHONPATH=. python3 \
  experiments/translation_geometry_followup/run_fresh_n11_replication.py \
  --config experiments/translation_geometry_followup/config.json \
  --stage calibration \
  2>&1 | tee fresh_N11_replication.log

PYTHONPATH=. python3 \
  experiments/translation_geometry_followup/run_fresh_n11_replication.py \
  --config experiments/translation_geometry_followup/config.json \
  --stage analysis
```

Calibration은 subset 완료 시마다 run CSV와 subset summary에 append합니다.
중단 후 동일 command에 `--resume`을 추가하면 완료 subset을 건너뜁니다.

완료된 N=9/N=11 subset 결과에서 weakest translation direction을 비교하려면:

```bash
PYTHONPATH=. python3 \
  experiments/translation_geometry_followup/analyze_weakest_direction.py
```

이 분석은 calibration을 다시 실행하지 않으며 `trace(Cov(b))`, `lambda_min`,
condition number, `tr(Cov(b)^-1)`의 subset-level Spearman 관계와 두 N의
eigenvalue 분포를 저장합니다.
