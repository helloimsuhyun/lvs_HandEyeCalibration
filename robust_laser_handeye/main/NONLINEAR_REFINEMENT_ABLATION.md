# 반복 선형해 기반 비선형 refinement 검토

## 결론

현재 데이터와 설정에서는 반복 교대 선형해를 비선형 최적화의 초기값으로
사용하는 편이 더 안전하다. 쉬운 초기값에서는 raw initial에서 시작한
비선형 최적화와 같은 해에 도달하지만, 중간/어려운 초기값에서는 raw
initial joint 최적화에 남아 있던 큰 local-minimum outlier를 제거했다.

다만 `alternating -> nonlinear`이 `alternating only`보다 모든 trial과 모든
hand-eye 지표에서 정확한 것은 아니다.

- three-plane에서는 translation과 rotation이 모두 명확하게 개선됐다.
- single-plane에서는 rotation과 tail error는 크게 개선됐지만 translation의
  중앙 경향은 통계적으로 명확하게 개선되지 않았다.
- 학습 residual은 모든 조건의 100/100 trial에서 감소했지만 GT hand-eye
  error는 일부 trial에서 증가했다.

같은 calibration scan에서 평면을 추정하는 unknown-plane 문제라면 이전
법선을 hard-fix하는 것을 기본값으로 권장하지 않는다.

- `loss="linear"`이고 최종 관심 변수가 hand-eye뿐이면 매 후보 transform에서
  평면을 PCA로 제거하는 6-변수 `refit`이 우선 선택이다.
- plane parameter, joint Jacobian/covariance, plane prior 또는 robust joint
  model이 필요하면 `joint`를 사용한다.
- `fixed_normals`는 법선이 외부 측량 등으로 독립적으로 정확하게 알려진
  경우에 적합하다.

## 실험 설계

공용 immutable dataset의 같은 noisy trial에서 다음 여섯 arm을 짝지어
비교했다.

1. `alternating`
2. `alternating_fixed_planes`
3. `alternating_fixed_normals`: 이전 법선 고정, offset은 후보마다 LS 제거
4. `alternating_refit`: hand-eye 6변수, 후보마다 plane PCA refit
5. `alternating_joint`: hand-eye + plane normal + offset 공동 최적화
6. `initial_joint`: 반복 선형 단계를 거치지 않고 raw initial에서 공동 최적화

공통 설정:

- single-plane 100 trial + three-plane 100 trial
- 초기오차 `easy=25 mm/5 deg`, `medium=100 mm/15 deg`,
  `hard=200 mm/30 deg`
- 총 600 paired task, 3,600 arm row
- trial당 108 scan, profile당 100 point
- sensor X/Z 독립 Gaussian noise `sigma=0.20 mm`
- alternating `max_iter=3000`, `tol=1e-5`
- nonlinear linear loss, `max_nfev=300`, `ftol=xtol=gtol=1e-10`
- outlier: translation `>5 mm` 또는 rotation `>1 deg`
- paired median delta의 bootstrap 95% CI, win fraction의 Wilson 95% CI
- 실용/수치 tie margin: `1e-4 mm`, `1e-4 deg`

모든 3,600 arm이 예외 없이 완료됐다. 각 condition/trial의 arm 수,
dataset/seed/initial hash와 alternating estimate hash도 통계 계산 전에
검증했다.

## 1. 반복 선형해를 비선형 초기값으로 쓸 것인가

아래 outlier 수는 `initial_joint -> alternating_joint`, NFEV는 비선형
단계의 중앙값이다.

| Geometry | 초기오차 | Outlier | Median NFEV |
|---|---:|---:|---:|
| Single | 25 mm / 5 deg | 0 -> 0 | 13 -> 6 |
| Single | 100 mm / 15 deg | 3 -> 0 | 33.5 -> 5 |
| Single | 200 mm / 30 deg | 3 -> 0 | 46.5 -> 5 |
| Three | 25 mm / 5 deg | 0 -> 0 | 15 -> 6 |
| Three | 100 mm / 15 deg | 0 -> 0 | 42 -> 6 |
| Three | 200 mm / 30 deg | 1 -> 0 | 49.5 -> 6 |

outlier가 아닌 trial에서는 두 초기화가 `1e-4 mm/deg` 이내의 같은 해에
도달했다. 즉 중앙 정확도를 더 높이는 효과보다는 basin 안정성과 tail
robustness를 높이는 효과가 핵심이다. SciPy의 `success=True`만으로는
local-minimum outlier를 검출하지 못했다.

반복 선형 단계가 이미 현재 workflow에 포함돼 있다면 그 결과를 비선형
초기값으로 재사용하는 것이 합리적이다. 반면 raw nonlinear과
alternating+nonlinear 중 하나만 새로 실행하는 상황에서는 전체 계산시간도
고려해야 한다. NFEV는 줄지만 alternating 비용이 추가되므로 end-to-end
pipeline이 반드시 더 빠른 것은 아니다.

## 2. Alternating only 대비 nonlinear refinement

### Single-plane

초기오차 단계가 달라도 alternating이 같은 basin으로 수렴해 결과가 거의
같았다. medium 조건의 대표값은 다음과 같다.

| Method | Translation median / p95 [mm] | Rotation median / p95 [deg] | Outlier |
|---|---:|---:|---:|
| Alternating | 0.19676 / 0.38654 | 0.27988 / 0.58759 | 0 |
| Alternating -> joint | 0.20385 / 0.33754 | 0.11838 / 0.22602 | 0 |

paired 기준으로 joint의 translation 개선률은 53%였고 median improvement
95% CI는 `[-0.0244, 0.0384] mm`로 0을 포함했다. 반면 rotation은 85%에서
개선됐고 paired median improvement는 `0.1715 deg`,
95% CI는 `[0.1321, 0.2058] deg`였다.

두 지표가 동시에 개선된 trial은 52%, 동시에 악화된 trial은 14%,
tradeoff는 34%였다. 따라서 single-plane에서 “일반적으로 더 정확하다”를
translation과 rotation 모두에 대해 단정할 수 없다. rotation과 p95/tail을
중시하면 nonlinear refinement의 이점이 크다.

### Three-plane

| 초기오차 | Method | Translation median [mm] | Rotation median [deg] | Outlier |
|---|---|---:|---:|---:|
| Easy | Alternating | 0.08162 | 0.08953 | 0 |
| Easy | Alternating -> joint | 0.03772 | 0.06065 | 0 |
| Medium | Alternating | 0.08320 | 0.09099 | 2 |
| Medium | Alternating -> joint | 0.03772 | 0.06065 | 0 |
| Hard | Alternating | 0.08898 | 0.09445 | 15 |
| Hard | Alternating -> joint | 0.03772 | 0.06065 | 0 |

medium 조건에서 translation/rotation 개선률은 각각 92%/74%였다. paired
median improvement와 95% CI는 각각
`0.0470 [0.0401, 0.0538] mm`,
`0.0271 [0.0215, 0.0431] deg`였다. 두 지표가 동시에 개선된 비율은 70%다.

hard 조건에서는 translation/rotation 개선률이 93%/79%였고 alternating의
15개 outlier가 모두 제거됐다. 테스트한 three-plane 조건에서는 nonlinear
refinement가 명확하게 더 정확하고 안정적이었다.

## 3. 이전 법선을 고정할 것인가

medium 조건의 대표 중앙값:

| Geometry | Plane treatment | Translation [mm] | Rotation [deg] | Outlier |
|---|---|---:|---:|---:|
| Single | Fixed normals | 0.15272 | 0.31061 | 0 |
| Single | Refit planes | 0.20385 | 0.11838 | 0 |
| Single | Joint planes | 0.20385 | 0.11838 | 0 |
| Three | Fixed normals | 0.07353 | 0.08896 | 2 |
| Three | Refit planes | 0.03772 | 0.06065 | 0 |
| Three | Joint planes | 0.03772 | 0.06065 | 0 |

Single-plane에서는 joint가 fixed normals보다 translation이 좋은 trial은
32%뿐이었지만 rotation은 86%에서 좋았다. 두 지표 동시 개선은 32%,
tradeoff는 54%였다. 같은 scan에서 얻은 법선을 고정하면 translation 쪽으로
오차를 재분배하면서 rotation 정확도를 희생하는 경향이 나타났다.

Three-plane medium에서는 joint가 fixed normals보다 translation 92%,
rotation 71%에서 좋았고 두 지표 동시 개선은 69%였다. hard에서는 각각
93%, 76%, 동시 개선 74%였으며 fixed normals에 남아 있던 15개 outlier를
joint가 모두 제거했다.

plane-normal GT error 중앙값도 다음처럼 줄었다.

- Single: `0.28665 -> 0.10621 deg`
- Three medium: `0.10614 -> 0.07439 deg`

따라서 alternating과 같은 데이터에서 얻은 normal은 noise-free truth가
아니며 hard constraint로 취급하면 errors-in-variables bias를 고정할 수 있다.

## 4. Refit과 joint의 관계

linear loss에서는 다음 두 문제가 같은 최소값을 갖는다.

```text
min_T [ min_(unit n_j, d_j) sum r_ijk(T, n_j, d_j)^2 ]
    = min_(T, unit n_j, d_j) sum r_ijk(T, n_j, d_j)^2
```

실험에서도 600/600 trial에서 `1e-4 mm/deg` 이내로 같았다.

- Single 최대 차이: translation `3.10e-5 mm`, rotation `1.33e-5 deg`
- Three 최대 차이: translation `4.27e-6 mm`, rotation `2.14e-6 deg`

따라서 hand-eye만 필요하고 loss가 linear라면 6변수 `refit`으로 충분하다.
최종 plane이 필요하면 종료 transform에서 PCA를 한 번 수행하면 된다.
Joint는 explicit plane estimate와 full joint Jacobian, plane prior 또는 robust
plane model이 필요한 경우 선택한다. 다만 robust loss에서는 현재 `refit`의
PCA와 `fixed_normals`의 arithmetic-mean offset이 robust inner optimum이
아니므로 joint와 목적함수가 더 이상 동일하지 않다.

## 권장 적용 정책

1. Unknown plane을 같은 scan으로 추정하고 hand-eye만 필요:
   `alternating -> refit`.
2. Plane parameter/joint uncertainty/prior가 필요:
   `alternating -> joint`.
3. 외부에서 독립적으로 측정한 정확한 normal이 있음:
   `fixed_normals`; 법선 불확실성이 유한하면 hard-fix보다 soft prior를 권장.
4. Alternating에서 추정한 normal을 그대로 고정:
   기본 정책으로 사용하지 않음.
5. 배포 시 nonlinear 결과 수용 조건:
   finite result, optimizer success, 충분한 Jacobian rank, objective 비증가,
   과도하지 않은 SE(3) update를 확인하고 실패하면 alternating 결과로
   fallback. 가능하면 pose 단위 held-out scan 또는 외부 측량으로 검증.

## 구현 및 재현

추가 구현:

- `laser_handeye/nonlinear_refinement.py`
  - `plane_mode="fixed_normals"`
  - normal 고정, candidate별 `d=mean(p@n)` LS profiling
- `main/run_nonlinear_refinement_ablation.py`
  - 여섯 arm paired runner
  - worker 병렬화
  - failure/outlier 보존 long CSV
  - paired bootstrap/Wilson/Wilcoxon/Pareto 통계
  - pairing hash audit와 source SHA-256

실행 명령:

```bash
cd robust_laser_handeye
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=. python3 main/run_nonlinear_refinement_ablation.py \
  --collection single_plane=dataset/fair_plane_initialization_shared_global/N108/single_plane \
  --collection three_plane=dataset/fair_plane_initialization_shared_global/N108/three_plane \
  --initialization easy:25:5 \
  --initialization medium:100:15 \
  --initialization hard:200:30 \
  --output-dir results/nonlinear_refinement_ablation_noise0p20_init_levels \
  --workers 8 \
  --bootstrap-samples 10000
```

결과 artifact:

- `results/nonlinear_refinement_ablation_noise0p20_init_levels/trials_long.csv`
- `results/nonlinear_refinement_ablation_noise0p20_init_levels/summary.json`
- CSV SHA-256:
  `6113a11eab47f6858c926cd0beb7f951ac78756b41b0cca455339f2a1b7c3e0a`

생성 결과 디렉터리는 repository `.gitignore` 대상이다. 전체 test suite는
`69 passed`다.

## 해석 범위

이 결론은 현재 synthetic dataset, `sigma=0.20 mm`, 108 scan,
linear/unweighted point-to-plane loss에 대한 것이다. 세 초기오차 단계와
single/three-plane geometry를 검사했지만 다음은 아직 일반화 범위 밖이다.

- noise level/scan count/pose-diversity sweep
- robot pose readback noise와 실제 outlier
- 실제 센서의 model mismatch
- held-out scan 또는 독립 측량 기준
- robust/heteroscedastic weighted objective

특히 X/Z noise의 plane-normal 방향 투영분산은 pose마다 다르므로 현재
unweighted objective가 정확한 MLE는 아니다. 따라서 “항상 정확도 향상”이
아니라 “현재 조건에서 three-plane은 명확히 개선, single-plane은
rotation/tail 개선과 translation tradeoff”로 해석해야 한다.
