# Translation trace–isotropy controlled experiment

기존 Phase 1 candidate bank에서 저장된 Phase 1/2A subset과 겹치지 않는 fresh
random subset을 생성해 `trace(Cov(b))`와 `condition(Cov(b))` 효과를 분리합니다.

Stage 1은 geometry와 matching만 계산하며 calibration을 실행하지 않습니다.

```bash
PYTHONPATH=. python3 \
  experiments/translation_trace_isotropy_controlled/run.py \
  --config experiments/translation_trace_isotropy_controlled/config.json \
  --stage stage1
```

`stage1_manifest.json`의 `matching_passed`가 true인 경우에만 Stage 2를 실행할
수 있습니다. Stage 2는 저장된 Phase 2A initial-error scenario bank와 모든
subset에 동일한 slot-indexed noise realization을 적용합니다.
기본 config는 primary comparison인 A–B(isotropy effect)와 A–C(trace effect)만
calibration하며, Stage 1에는 C–D와 B–D supporting pair도 함께 저장됩니다.

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. python3 \
  experiments/translation_trace_isotropy_controlled/run.py \
  --config experiments/translation_trace_isotropy_controlled/config.json \
  --stage stage2 \
  2>&1 | tee translation_trace_isotropy_stage2.log
```

먼저 별도 경로에서 전체 pipeline smoke test를 할 수 있습니다.

```bash
PYTHONPATH=. python3 experiments/translation_trace_isotropy_controlled/run.py \
  --config experiments/translation_trace_isotropy_controlled/config.json \
  --stage stage1 --smoke --output-dir /tmp/translation_controlled_smoke

PYTHONPATH=. python3 experiments/translation_trace_isotropy_controlled/run.py \
  --config experiments/translation_trace_isotropy_controlled/config.json \
  --stage stage2 --smoke --output-dir /tmp/translation_controlled_smoke
```
