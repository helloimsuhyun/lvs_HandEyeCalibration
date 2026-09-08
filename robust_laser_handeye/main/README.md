# Synthetic Laser Hand–Eye Calibration Dataset Generator

`generate_calibration_dataset.py`는 레이저 프로파일 센서 hand–eye calibration 실험에 사용할 **noise-free synthetic dataset collection**을 생성합니다.

## 반복 선형해 기반 비선형 refinement ablation

반복 교대 선형해를 비선형 초기값으로 사용할 때의 효과와
plane normal을 고정/refit/joint 최적화하는 선택은
[`NONLINEAR_REFINEMENT_ABLATION.md`](NONLINEAR_REFINEMENT_ABLATION.md)에
정리되어 있습니다. 동일 trial에서 여섯 arm을 실행하는 재현 runner는
`run_nonlinear_refinement_ablation.py`입니다.

이 스크립트는 데이터 생성만 수행합니다.

- 측정 노이즈 추가
- robot pose readback noise 추가
- hand–eye calibration
- Monte Carlo 결과 분석

위 과정은 별도의 calibration/experiment 스크립트에서 수행해야 합니다. 따라서 여러 calibration method가 동일한 raw dataset을 공유하여 공정하게 비교할 수 있습니다.

## Single/three-plane online Fisher 대 random 비교

`generate_independent_random_plane_comparison.py`라는 파일명은 기존 실행
경로와의 호환을 위해 유지합니다. 현재 Fisher branch는 GT에서 Jacobian을 한
번 계산해 고정하는 oracle 방식이 아니라 다음 closed-loop를 수행합니다.

1. 네 branch가 동일한 noisy random bootstrap pose로 시작합니다.
2. 획득한 scan만 사용해 hand–eye와 물리적 plane을 joint 최적화합니다.
3. 아직 획득하지 않은 후보에는 robot pose와 plane ID만 노출합니다. 예상
   profile은 현재 hand–eye/plane 추정값으로 계산합니다.
4. 현재 observed information과 후보 expected information을 합쳐 설정된
   D- 또는 E-optimal score가 가장 큰 pose를 고릅니다.
5. 선택한 pose의 noisy scan만 acquire하고, 지금까지의 모든 scan을 다시
   최적화·재선형화합니다.

이 순서는 현재 추정값에서 후보 Jacobian을 계산하고 실제 측정 뒤 전체
파라미터를 재최적화하는 [Yang et al.의 sequential NBV 구조](https://arxiv.org/abs/2303.06766)를
laser point-to-plane 문제에 적용한 것입니다. 해당 논문의 checkerboard
reprojection 문제를 그대로 재현한 것은 아닙니다.

joint local state는 다음과 같습니다.

```text
single: hand–eye right-local SE(3) 6 + plane (normal tangent 2, offset 1) = 9
three : hand–eye right-local SE(3) 6 + 3 planes × 3 = 15
```

한 point의 residual은
`r = nᵀ(T_base_ef T_ef_s p_s) - d`이며, 설정한 sensor `z` 또는 `xz`
Gaussian noise가 plane residual로 투영되는 표준편차로 whiten합니다. 획득
residual Jacobian에는 이 geometry-dependent whitening의 미분도 포함합니다.
후보의 expected residual은 0이므로 후보 Fisher는 통상적인
`F = Jᵀ Σ⁻¹ J` mean-sensitivity 근사입니다.

plane은 알려진 GT로 고정하지 않습니다. [Peng–Sturm의 interest/nuisance
block covariance reduction](https://arxiv.org/abs/1811.03264)을 이 문제의
shared plane nuisance에 적용해

```text
F_handeye = F_hh - F_hp F_pp⁻¹ F_ph
```

를 Cholesky solve로 계산한 뒤 hand–eye 6-DOF만 score합니다. 임의의
`I`를 더하는 information prior는 없습니다.

- D-optimal: `0.5 log det(F_handeye)`
- E-optimal: `lambda_min(F_handeye)`

기본 runner는 전체 hand–eye 불확실성 부피를 줄이는 D-optimal입니다.
E-optimal은 weakest direction을 직접 개선하는 선택적 ablation입니다. E-opt는
회전(rad)과 이동(mm)의 단위에 따라 후보 순위가 달라지므로
[Wilson et al.이 설명한 parameter-unit weighting](https://doi.org/10.1109/TRO.2014.2345918)에
따라 고정 dimensionless metric을 명시합니다. 기본값은
`FISHER_ROTATION_SCALE_DEG=2`, `FISHER_TRANSLATION_SCALE_MM=10`입니다.
이 값은 Gaussian prior가 아닙니다. D-opt의 후보 순위는 고정된 nonsingular
hand–eye scaling에 영향을 받지 않지만 E-opt의 순위는 영향을 받습니다.

no-prior joint Fisher가 bootstrap에서 full rank가 되도록
`INITIAL_RANDOM_SCANS`는 9 이상, 3의 배수, `TOTAL_SCANS`보다 작아야 합니다.
기본값은 27로, three-plane에서 9 scan/plane을 사용합니다. 공통 초기화
27개는 legacy per-axis `±100 mm / ±15°` 조건에서
18·27·36·54 bootstrap을 비교했을 때 초기 추정 안정성과 후속 정책 구간
사이의 가장 나은 절충이었습니다. 현재의 norm-bounded `100 mm / 15°`
기본값에서도 보수적인 값으로 그대로 사용합니다. 총
108 scan 중 81 scan은 후속 정책 비교에 남습니다. 최종 three-plane
Fisher/random branch에도 같은 per-plane quota를 적용합니다.
다만 큰 초기 오차의 nonlinear local minimum은 scan 수만으로 완전히
제거되지 않으므로, bootstrap estimate/error hash와 수렴 상태를 manifest에
계속 기록합니다.

```bash
# 빠른 end-to-end 확인
MAX_TRIALS=3 TOTAL_SCANS=18 INITIAL_RANDOM_SCANS=9 \
CANDIDATE_POOL_SIZE=36 PROFILE_POINTS=16 MAX_ITER=300 \
  bash main/run_fisher_random_policy_comparison.sh

# 기본 실험: D-optimal, 100 trials, 108 scans, initial 27, pool 324
bash main/run_fisher_random_policy_comparison.sh

# E-optimal ablation
FISHER_OBJECTIVE=e_optimal \
  bash main/run_fisher_random_policy_comparison.sh
```

생성되는 네 collection은 다음과 같습니다.

| 경로 | geometry | pose 선택 |
|---|---|---|
| `single_plane/` | single | seeded random |
| `single_plane_fisher/` | single | online Fisher |
| `three_plane/` | three | seeded random |
| `three_plane_fisher/` | three | online Fisher |

저장 profile은 selection 때 실제 사용한 noisy `measured` 값입니다. runner는
최종 calibration에서 `--noise-std-mm 0`을 사용해 noise를 두 번 넣지 않습니다.
공통 후보를 선택한 두 정책에는 candidate-keyed 동일 noise가 적용됩니다.
dataset을 재생성하면 dependent result도 강제로 다시 계산하며, 재사용 result는
입력 `collection.json` SHA-256과 일치해야 합니다.

`comparison_manifest.json`에는 단계별 current-estimate hash, predicted/observed
objective, 재최적화 상태, selected candidate ID, pose/data hash, coordinate
metric, noise model과 Fisher/random 요약 통계가 기록됩니다. 기본 결과는
`results/fisher_random_policy_comparison/online_v4_N108_I27_P324_d_optimal/`에
저장됩니다.

현재 실험의 범위와 한계도 구분해야 합니다.

- Fisher policy는 GT hand–eye, GT plane, future profile을 직접 받지 않습니다.
- 다만 simulator의 공통 candidate robot-pose bank는 GT target/hand–eye로
  feasibility를 보장해 만든 oracle-feasible action bank입니다. 따라서 현재
  결과는 동일 bank 안의 Fisher-vs-random 정책 비교이며, 그대로 실로봇 NBV
  planner라고 해석하면 안 됩니다.
- 초기 최적화는 simulator가 GT에 설정 범위의 오차를 더해 만든 nominal
  transform에서 시작합니다. FIM에 prior precision을 더하지는 않지만,
  완전한 data-only global initialization은 아닙니다.
- robot pose uncertainty는 모델링하지 않으며 독립 Gaussian profile noise와
  local linearization/CRLB 근사를 가정합니다. 높은 Fisher score가 실제
  calibration error 감소를 보장하지는 않습니다.
- online selection은 weighted joint optimizer를 쓰지만 runner의 최종
  benchmark는 기존 `iterative` solver를 사용합니다. 이는 선택 pose가 현재
  calibration solver의 실제 오차를 개선하는지를 별도로 평가하기 위함입니다.

cross-plane effective-normal 중앙값은 audit metric으로만 기록하고 기본값에서는
trial 제거 기준으로 사용하지 않습니다. 별도 dataset 품질검사에만
`--min-effective-cross-plane-angle-deg`를 명시할 수 있습니다. 전체
pose-diversity 비교는 다음으로 실행합니다.

```bash
MAX_TRIALS=100 bash main/run_pose_diversity_comparison.sh
```

## 실험 runner와 plot 호환성

다음 runner는 calibration 완료 후 대응하는 plotter를 자동 실행합니다.
plotter에는 runner에서 실제로 실행한 condition 목록만 전달되므로 같은 result
root에 과거 condition이 남아 있어도 현재 그래프에 섞이지 않습니다.

| Runner | Plotter | 기본 result 구조 |
|---|---|---|
| `run_fisher_random_policy_comparison.sh` | `make_plot/plot_fisher_random_policy_comparison.py` | `<root>/online_v4_N108_I27_P324_<objective>_init_.../{single,three}_{random,fisher}` |
| `run_pose_diversity_comparison.sh` | `make_plot/plot_pose_diversity_comparison.py` | `<root>/<level>/{single_plane,three_plane}` |
| `run_fixed_line_noise_comparison.sh` | `make_plot/plot_fixed_line_noise_boxplots.py` | `<root>/noise_<sigma>/{single_plane,three_plane}` |
| `run_fixed_noise_line_count_comparison.sh` | `make_plot/plot_fixed_noise_line_count_boxplots.py` | `<root>/N<count>/{single_plane,three_plane}` |
| `run_fair_plane_initialization_levels.sh` | `make_plot/plot_initialization_robustness.py` | `<root>/<level>_t<mm>_r<deg>/{single_plane,three_plane}` |

공통 환경 변수:

```bash
# plot 생성을 생략
MAKE_PLOTS=0 bash main/run_pose_diversity_comparison.sh

# 성공 trial만 error boxplot에 포함하고 150 dpi로 저장
PLOT_SUCCESS_ONLY=1 PLOT_DPI=150 \
  bash main/run_pose_diversity_comparison.sh
```

모든 main 실험을 공통 초기화 `‖Δt‖ ≤ 100 mm`,
`d_SO(3)(R_init,R_true) ≤ 15°`로 순서대로 실행하려면:

```bash
bash main/run_all_main_experiments.sh
```

같은 모든 실험에서 alternating 결과를 초기값으로 사용해
`plane_mode="refit"` nonlinear refinement까지 적용하려면:

```bash
bash main/run_all_main_experiments_refit.sh
```

이 전용 runner의 결과와 로그는 기본적으로 `result_refit/` 아래에
저장됩니다. 기존 dataset은 재사용하며, 기존 iterative 결과가 들어 있는
`results/`는 변경하지 않습니다. 예를 들어 빠른 확인은 다음과 같습니다.

```bash
MAX_TRIALS=3 MAKE_PLOTS=0 \
  bash main/run_all_main_experiments_refit.sh
```

Fisher-vs-random을 제외한 네 실험만 모두 실행하려면:

```bash
bash main/run_all_main_experiments_without_fisher.sh
```

이 통합 runner는 initialization, fixed-line noise, fixed-noise line-count,
pose-diversity, Fisher-vs-random 실험을 차례로 실행하고 각 로그를
`results/all_main_experiment_logs/`에 저장합니다. Fisher 실험에는 기본
bootstrap 27개와 D-optimal objective를 전달합니다. `2° / 10 mm` 좌표
metric도 manifest에 기록되며 E-optimal ablation에서 후보 순위에 사용됩니다.

```bash
# 실행할 명령과 경로만 확인
DRY_RUN=1 bash main/run_all_main_experiments.sh

# 빠른 Monte Carlo 실행
MAX_TRIALS=3 MAKE_PLOTS=0 \
  bash main/run_all_main_experiments.sh

# 일부 실험만 생략
RUN_POSE_DIVERSITY=0 RUN_FISHER_RANDOM=0 \
  bash main/run_all_main_experiments.sh
```

개별 runner의 기본 초기화도 모두 translation `direction_norm`,
rotation `axis_angle` sampler를 사용하는 `100 mm / 15°`입니다. 두 크기는
각각 translation norm과 SO(3) geodesic angle의 정확한 상한입니다.
initialization runner의 기본 sweep은
`easy:25:5 medium:100:15 hard:200:30`입니다. 기존 방식은
`INIT_TRANSLATION_PERTURBATION=box_xyz`와
`INIT_ROTATION_PERTURBATION=euler_xyz`로 재현할 수 있습니다.

초기화 범위의 소수점은 폴더명에서 `p`로 정규화됩니다. 예를 들어
`fine:12.5:2.5`는 `fine_t12p5_r2p5`가 되며 plotter는 `p`와 기존 `.` 표기를
모두 읽을 수 있습니다.

각 runner는 기존 dataset/result를 재사용하기 전에
`validate_experiment_artifact.py`로 trial 수, seed, pose 범위, noise,
initialization, solver 설정과 shared-global schema를 검증합니다. 현재 설정과
다르면 조용히 섞어 쓰지 않고 `FORCE_REGENERATE=1` 또는 `FORCE_RERUN=1`을
요구합니다.

---

## 1. 지원하는 dataset mode

| Mode | 설명 |
|---|---|
| `single-plane-circular` | 원형 9-line single-plane acquisition dataset |
| `three-plane` | 서로 직교하는 3개 평면을 사용하는 random 6-DoF dataset |

---

## 2. 파일 위치

예시 저장소 구조:

```text
robust_laser_handeye/
├── examples/
│   └── generate_calibration_dataset.py
└── laser_handeye/
```

저장소 루트로 이동합니다.

```bash
cd /home/choisuhyun/lvs_HandEyeCalibration/robust_laser_handeye
conda activate laser_handeye
```

스크립트 도움말:

```bash
PYTHONPATH=. python examples/generate_calibration_dataset.py --help
```

각 mode별 세부 도움말:

```bash
PYTHONPATH=. python examples/generate_calibration_dataset.py \
  single-plane-circular --help
```

```bash
PYTHONPATH=. python examples/generate_calibration_dataset.py \
  three-plane --help
```

---

# 3. Single-Plane Circular Dataset

하나의 평면 위에 40도 간격의 9개 radial target line을 구성하고, 각 line에 대해 `(d, theta, beta)` 조합을 생성합니다.

## 기본 예제

```bash
PYTHONPATH=. python examples/generate_calibration_dataset.py \
  single-plane-circular \
  --trials 100 \
  --seed 7 \
  --output-dir results/datasets/single_plane_circular_100
```

기본 target scan 구성:

```text
9 target lines
d = 60, 90, 120 mm
theta = 30 deg
beta = 60, 90, 120 deg
```

기본 target scan 수:

```text
9 lines × 3 heights × 1 theta × 3 beta = 81 scans
```

또한 기본값에서는 line ID `1, 2, 5, 6`에 theta 60도의 reference scan이 추가됩니다.

```text
4 lines × 3 heights × 1 theta × 3 beta = 36 additional scans
```

따라서 기본 총 scan 수는 trial당:

```text
81 + 36 = 117 scans
```

## Reference scan 없이 정확히 81 scans 생성

`--reference-line-ids` 뒤에 값을 쓰지 않으면 reference ring이 비활성화됩니다.

```bash
PYTHONPATH=. python examples/generate_calibration_dataset.py \
  single-plane-circular \
  --trials 100 \
  --seed 7 \
  --output-dir results/datasets/single_plane_81_scans \
  --profile-points 100 \
  --radius-mm 100 \
  --heights-mm 60 90 120 \
  --theta-deg 30 \
  --beta-deg 60 90 120 \
  --pose-geometry paper_incidence \
  --reference-line-ids
```

> 고정된 하나의 `|theta|`만 사용하는 unknown-plane 문제는 sensor-Z translation과 plane offset 사이에 관측 불가능성이 생길 수 있습니다.

## Multi-theta dataset 생성

모든 target line에서 theta 30도와 60도를 모두 사용합니다.

```bash
PYTHONPATH=. python examples/generate_calibration_dataset.py \
  single-plane-circular \
  --trials 100 \
  --seed 7 \
  --output-dir results/datasets/single_plane_theta_30_60 \
  --profile-points 100 \
  --radius-mm 100 \
  --heights-mm 60 90 120 \
  --theta-deg 30 60 \
  --beta-deg 60 90 120 \
  --pose-geometry paper_incidence \
  --reference-line-ids
```

이 경우 target scan 수는:

```text
9 lines × 3 heights × 2 theta × 3 beta = 162 scans
```

## 기본 81 scans + 선택 line에 theta 60도 추가

```bash
PYTHONPATH=. python examples/generate_calibration_dataset.py \
  single-plane-circular \
  --trials 100 \
  --seed 7 \
  --output-dir results/datasets/single_plane_reference_theta \
  --profile-points 100 \
  --radius-mm 100 \
  --heights-mm 60 90 120 \
  --theta-deg 30 \
  --beta-deg 60 90 120 \
  --reference-line-ids 1 2 5 6 \
  --reference-heights-mm 60 90 120 \
  --reference-theta-deg 60 \
  --reference-beta-deg 60 90 120
```

## 주요 옵션

| 옵션 | 의미 | 기본값 |
|---|---|---:|
| `--profile-points` | profile당 point 수 | 100 |
| `--profile-half-width-mm` | profile X축 half-width | 25 |
| `--radius-mm` | circular target line 반경 | 100 |
| `--heights-mm` | sensor distance `d` 목록 | 60 90 120 |
| `--theta-deg` | incidence/projection angle 목록 | 30 |
| `--beta-deg` | sensor tilt angle 목록 | 60 90 120 |
| `--pose-geometry` | pose 기하 convention | paper_incidence |
| `--reference-line-ids` | 추가 theta scan을 넣을 line ID | 1 2 5 6 |
| `--reference-theta-deg` | 추가 scan의 theta | 60 |
| `--check-reachability` | 간단한 workspace box 검사 활성화 | 비활성 |

---

# 4. Three-Plane Dataset

서로 직교하는 3개 평면에 대해 random 6-DoF sensor pose와 laser profile을 생성합니다.

## 기본 예제

```bash
PYTHONPATH=. python examples/generate_calibration_dataset.py \
  three-plane \
  --trials 100 \
  --seed 7 \
  --output-dir results/datasets/three_plane_100
```

기본 scan 수:

```text
3 planes × 35 poses per plane = 105 scans per trial
```

## 3개 평면에서 총 81 scans 생성

평면당 27 pose를 사용합니다.

```bash
PYTHONPATH=. python examples/generate_calibration_dataset.py \
  three-plane \
  --trials 100 \
  --seed 7 \
  --output-dir results/datasets/three_plane_81_scans \
  --poses-per-plane 27 \
  --profile-points 100 \
  --profile-half-width-mm 25 \
  --plane-distance-range-mm 650 1000 \
  --plane-min-axis-angle-deg 1 \
  --tangent-range-mm 220 \
  --profile-depth-range-mm 60 150
```

## 주요 옵션

| 옵션 | 의미 | 기본값 |
|---|---|---:|
| `--poses-per-plane` | 평면당 pose 수 | 35 |
| `--profile-points` | profile당 point 수 | 100 |
| `--profile-half-width-mm` | profile half-width | 25 |
| `--plane-distance-range-mm` | 평면 offset 범위 | 650 1000 |
| `--plane-min-axis-angle-deg` | base axis와의 최소 acute angle | 1 |
| `--tangent-range-mm` | 평면 tangent 방향 sampling 범위 | 220 |
| `--profile-depth-range-mm` | profile Z 범위 | 60 150 |
| `--min-view-dot` | 최소 view-direction dot 조건 | 0 |
| `--max-trials-per-plane` | 유효 pose 생성 최대 시도 수 | 50000 |

---

# 5. 공통 옵션

모든 mode에서 다음 옵션을 지원합니다.

| 옵션 | 의미 |
|---|---|
| `--trials N` | 독립적으로 생성할 dataset trial 수 |
| `--seed N` | 전체 collection의 master random seed |
| `--output-dir PATH` | dataset collection 저장 위치 |

예시:

```bash
PYTHONPATH=. python examples/generate_calibration_dataset.py \
  three-plane \
  --trials 100 \
  --seed 42 \
  --output-dir results/datasets/three_plane_seed42
```

---

# 6. 출력 구조

생성 결과는 collection 단위로 저장됩니다.

```text
results/datasets/single_plane_circular_100/
├── collection.json
└── trials/
    ├── trial_000000/
    │   ├── manifest.json
    │   └── ...
    ├── trial_000001/
    │   ├── manifest.json
    │   └── ...
    └── ...
```

`collection.json`에는 다음 정보가 저장됩니다.

- collection schema와 version
- 생성 완료 상태
- master seed
- acquisition mode
- 요청/완료 trial 수
- dataset generation config
- trial별 경로
- trial별 generation seed
- logical dataset SHA-256

각 trial directory에는 다음 정보가 저장됩니다.

- ideal laser profiles
- robot flange poses
- scan metadata
- plane ID/group 정보
- ground-truth hand–eye transform
- ground-truth plane geometry
- acquisition-specific metadata

---

# 7. 재현성

같은 명령에서 다음 두 값이 같으면 동일한 ideal dataset이 생성됩니다.

```text
--seed
trial index
```

예:

```bash
--seed 7 --trials 100
```

의 `trial_000013`은 동일한 코드와 config에서 항상 같은 generation seed를 사용합니다.

각 trial의 logical SHA-256도 `collection.json`에 저장되므로 calibration method 간에 정말 동일한 dataset을 사용했는지 확인할 수 있습니다.

---

# 8. Output Directory 주의사항

스크립트는 기존 결과를 덮어쓰지 않습니다.

다음 조건에서는 실행이 중단됩니다.

- `--output-dir`이 파일인 경우
- `--output-dir` directory가 이미 존재하고 비어 있지 않은 경우

예를 들어 아래 경로가 이미 존재하면:

```text
results/datasets/single_plane_circular_100/
```

새 이름을 사용하거나 기존 directory를 직접 정리해야 합니다.

```bash
rm -rf results/datasets/single_plane_circular_100
```

그 후 다시 실행합니다.

> `rm -rf`는 경로를 반드시 확인한 후 사용하십시오.

---

# 9. 권장 생성 명령

## Single-plane 81-scan baseline

```bash
PYTHONPATH=. python examples/generate_calibration_dataset.py \
  single-plane-circular \
  --trials 100 \
  --seed 7 \
  --output-dir results/datasets/single_plane_81 \
  --profile-points 100 \
  --heights-mm 60 90 120 \
  --theta-deg 30 \
  --beta-deg 60 90 120 \
  --pose-geometry paper_incidence \
  --reference-line-ids
```

## Single-plane multi-theta

```bash
PYTHONPATH=. python examples/generate_calibration_dataset.py \
  single-plane-circular \
  --trials 100 \
  --seed 7 \
  --output-dir results/datasets/single_plane_multi_theta \
  --profile-points 100 \
  --heights-mm 60 90 120 \
  --theta-deg 30 60 \
  --beta-deg 60 90 120 \
  --pose-geometry paper_incidence \
  --reference-line-ids
```

## Three-plane 81-scan comparison

```bash
PYTHONPATH=. python examples/generate_calibration_dataset.py \
  three-plane \
  --trials 100 \
  --seed 7 \
  --output-dir results/datasets/three_plane_81 \
  --poses-per-plane 27 \
  --profile-points 100 \
  --plane-distance-range-mm 650 1000 \
  --plane-min-axis-angle-deg 1
```

---

# 10. Dataset 생성과 Calibration 분리

권장 실험 절차:

```text
1. ideal/raw synthetic dataset collection 생성
2. collection의 logical SHA-256 기록
3. calibration 실행 단계에서 동일한 noise seed 적용
4. iterative / nonlinear-refinement 비교
5. trial별 GT error와 convergence 저장
```

이 구조를 사용하면 각 method가 동일한 다음 입력을 소비합니다.

- 동일한 robot pose
- 동일한 ideal laser profile
- 동일한 GT hand–eye
- 동일한 plane geometry
- 동일한 measurement noise realization

따라서 method 간 비교에서 dataset 차이에 의한 편향을 제거할 수 있습니다.

---

# 11. Alternating 결과를 초기값으로 사용한 Joint Nonlinear Refinement

다음 스크립트는 같은 noisy trial에서 기존 반복 교대 최적화를 한 번
수행한 뒤, 그 결과를 초기값으로 비선형 refinement를 수행합니다.

```bash
bash main/run_alternating_joint_nonlinear_comparison.sh
```

기본 조건은 다음과 같습니다.

- single: 108 scans × 1 plane
- three: 36 scans/plane × 3 planes, 총 108 scans
- profile noise: x/z Gaussian 0.20 mm
- 초기 translation 오차: norm 기준 최대 100 mm
- 초기 rotation 오차: axis-angle 기준 최대 15 deg
- nonlinear loss: linear

비선형 변수는 hand-eye의 local SE(3) 6개와 평면당 3개입니다.

```text
[d_rotation(3), d_translation(3),
 plane_0_normal_tangent(2), plane_0_offset(1), ...]
```

법선은 3차원 unconstrained vector가 아니라 현재 단위 법선의
2차원 tangent space에서 갱신하므로 항상 unit norm을 유지합니다.
즉 single은 총 9개, three는 총 15개 변수를 동시에 추정합니다.

결과의 `trials.csv` 한 행에는 다음 값이 함께 저장됩니다.

- `alternating_*`: 비선형 refinement 직전 결과
- 기본 `translation_error_mm`, `rotation_error_deg`: refinement 후 결과
- `nonlinear_*`: residual, plane 변화량, runtime, Jacobian rank/condition
- `nonlinear_plane_normals_json`, `nonlinear_plane_offsets_json`: 최종 평면

따라서 서로 다른 실행을 사후 pairing하지 않아도 동일한 noise와
동일한 초기값에 대한 before/after 비교가 가능합니다. 생성되는 plot은
pose 오차, signed improvement, 평면 normal/offset 오차를 각각 보여줍니다.

---

# 12. 공통 pose 생성과 unrestricted D-optimal subset 탐색

`laser_handeye.pose_design`이 random, LHS, circular pose 생성과 좌표계 변환을
한 곳에서 제공합니다. 실험 스크립트는 자체 LHS 구현을 만들지 않고 이
공통 API를 사용합니다. `unrestricted_doptimal_pose_design.py`는 circular나
고정 tilt를 강제하지 않는 plane-relative 6DoF 범위에서 random 또는 LHS
candidate bank를 만든 뒤, unknown-plane nuisance를 Schur complement로
제거한 hand-eye 정보의 `logdet(H_eff)`를 multi-start 1-exchange로
최대화합니다.

```bash
python3 main/unrestricted_doptimal_pose_design.py \
  --candidate-design lhs \
  --output-dir result_unrestricted_doptimal
```

빠른 smoke test:

```bash
python3 main/unrestricted_doptimal_pose_design.py \
  --output-dir /tmp/unrestricted_doptimal_smoke \
  --candidate-design random \
  --candidate-count 80 \
  --subset-sizes 3 4 5 \
  --random-starts 3 \
  --random-baseline-subsets 20 \
  --mc-trials 0
```

현재 ideal straight-profile 모델에서는 scan 하나의 Jacobian row space가
최대 2차원이므로, single-plane nuisance 3DoF를 제거한 hand-eye rank의
상한은 `2N-3`입니다. 따라서 N=3,4는 pose와 무관하게 rank 6을 만족할 수
없으며 결과에 `structurally_unobservable`로 기록됩니다. N>=5에는 후보별
실제 profile geometry와 기존 analytic Jacobian을 사용해 정상적으로
D-optimal subset을 탐색합니다.

주요 산출물은 `candidate_bank.csv`, `candidate_information.npz`,
`summary.csv`, `best_subset_N*.csv/json`, start/random-baseline 기록,
geometry plot, `sanity_checks.json`, 그리고 paired Monte Carlo 결과입니다.
