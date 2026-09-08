# Exhaustive N=9 distance assignment study

같은 ordered `b_i`, `(u,v)`, `gamma=0`, 평균 거리와 D3 distance multiset을
고정하고 630개 unique assignment를 전수 조사합니다. 기존 fixed-translation
experiment의 pose/Jacobian/Schur/solver와 Phase-2A scenario bank를 재사용합니다.

```bash
PYTHONPATH=. python3 experiments/distance_assignment_exhaustive_n9/run.py \
  --config experiments/distance_assignment_exhaustive_n9/config.json \
  --stage geometry

PYTHONUNBUFFERED=1 PYTHONPATH=. python3 \
  experiments/distance_assignment_exhaustive_n9/run.py \
  --config experiments/distance_assignment_exhaustive_n9/config.json \
  --stage calibration --calibration-mode all 2>&1 | tee distance_assignment_all.log

PYTHONPATH=. python3 experiments/distance_assignment_exhaustive_n9/run.py \
  --config experiments/distance_assignment_exhaustive_n9/config.json \
  --stage analysis --calibration-mode all
```

Calibration은 assignment 하나가 끝날 때마다 parent process가 CSV에 append합니다.
중단 후 같은 command에 `--resume`을 붙이면 완료 assignment를 건너뜁니다.
`--calibration-mode selected`는 geometry logdet 10분위별 6개와 anchor를 사용합니다.

