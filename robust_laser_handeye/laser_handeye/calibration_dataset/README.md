# 범용 CalibrationDataset 취득 형식

이 모듈은 특정 캘리브레이션 알고리즘이 아니라 **취득된 로봇 pose와
레이저 raw profile**을 저장하는 공통 경계입니다. 시뮬레이션에서 만든
데이터와 실제 로봇에서 취득한 데이터를 같은 `CalibrationDataset`으로
표현하고, 캘리브레이션 방법은 저장된 데이터의 adapter만 선택합니다.

공개 loader와 adapter API는 다음과 같습니다.

```python
load_calibration_dataset(...)
dataset.to_tan2025_dataset()
dataset.to_scans_by_plane()
```

## 핵심 원칙

1. 생성 단계에서는 환경, GT hand-eye, robot motion과 **noise-free
   profile**만 확정합니다.
2. 저장된 `ideal`/`raw` profile에는 Gaussian noise, bias, dropout 또는
   pose readback noise를 미리 넣지 않습니다.
3. noise와 outlier는 저장 파일을 로드한 뒤 실험 단계에서 복사본에
   적용합니다. 따라서 서로 다른 알고리즘과 noise level이 정확히 같은
   환경, motion과 원본 profile을 공유합니다.
4. GT와 환경 truth는 평가를 위한 optional 정보입니다. solver 입력의
   필수 조건이 아니며 실제 로봇 dataset에서는 `truth=None`이어도 됩니다.
5. profile의 행 순서만으로 sensor channel을 추측하지 않습니다.
   `channel_ids`를 항상 명시합니다.

Effective GT hand-eye, true/commanded poses와 plane geometry는 `truth`에만
저장합니다. 일반 dataset/scan metadata에는 복제하지 않으므로 blind solver
입력은 `truth`를 전달하지 않는 adapter 경계에서 분리할 수 있습니다.
Collection의 생성 config는 평가·재현용 orchestration 정보이며 solver에
직접 전달하는 입력이 아닙니다.

여기서 `ideal`은 해석 시뮬레이터가 만든 noise-free 측정이고, `raw`는
실제 센서가 반환한 값입니다. 실제 센서의 raw 값에는 물리적인 측정
noise가 이미 포함될 수 있지만, writer가 인위적인 실험 noise를 추가하면
안 됩니다. 추가 noise 설정과 seed는 후단 실험 결과에 따로 기록합니다.

## 취득 모드

### `translation_composite`

한 개의 평면에서 두 종류의 motion을 별도 group으로 취득합니다.

- `translation`: tool orientation을 고정한 pure-translation motion
- `composite`: translation과 rotation이 함께 변하는 motion

두 group은 같은 `plane_id`를 사용하지만 서로 다른 `group_id`를 가집니다.
Tan 2025 closed-form adapter는 이 구분을 사용합니다. 특히 translation
profile 사이에는 동일한 실제 sensor channel을 비교할 수 있어야 하므로
`channel_id_semantics="stable_sensor_channel"`이 필요합니다.

기본 synthetic 설정은 Tan 2025 재현 설정의 translation 36 pose와
composite 30 pose를 사용하며 개수는 생성 설정에서 변경할 수 있습니다.

Plane은 `--plane-mode fixed|random`으로 선택합니다.

- `fixed`(기본): `--plane-center-base-mm`과 `--plane-normal-base` 사용
- `random`: trial마다 plane 하나를 생성하고 translation/composite가 공유

Random mode는 `--plane-angle-range-deg`, `--plane-min-axis-angle-deg`,
`--plane-distance-range-mm`, `--plane-tangent-range-mm`으로 제어합니다.
Normal은 Euler XYZ box sampling에서 유도되므로 구면 위 균일분포는 아닙니다.
정확한 sampled normal, offset과 center frame은 optional truth에 저장됩니다.
Truth의 `normal_sampling_euler_xyz_deg`는 normal을 뽑기 위한 latent 값이며,
`T_base_plane`의 X/Y축은 그 normal에서 만든 canonical tangent basis입니다.
무한평면에서는 normal 주위 roll이 관측되지 않으므로 두 회전을 동일한 board
frame으로 해석하면 안 됩니다.

### `single_plane_circular`

한 개의 평면 위 9개 원형 target line을 따라 profile을 취득하는 기존
single-plane optimal pattern입니다.

- `primary_ring`: 9 lines × 3 heights × 1 theta × 3 beta = **81 scans**
- `reference_ring`: 4 selected lines × 3 heights × 1 theta × 3 beta =
  **36 scans**
- 기본 합계: **81 + 36 = 117 scans**

두 group의 `plane_id`는 모두 0입니다. `reference_ring`은 별도
`acquisition_role="reference"`로 표시하지만 기본적으로 calibration에
포함됩니다. 각 scan metadata에는 가능한 경우 `line_id`, `parameter_id`,
`d_mm`, `theta_deg`, `beta_deg`, branch와 pose-geometry 규약을 보존합니다.

Reference group은 알려진 GT를 solver에 제공한다는 뜻이 아닙니다. 고정
incidence trajectory의 translation/plane-offset gauge를 깨기 위한 추가
motion group이라는 뜻입니다.

### `three_plane_random`

서로 직교하는 세 개의 plane에서 일반 6-DoF random view를 취득합니다.

- `plane_0`, `plane_1`, `plane_2`: plane마다 35 scans
- 기본 합계: **3 × 35 = 105 scans**
- 각 group의 `plane_id`는 각각 0, 1, 2
- 기본 `motion_kind="general_6dof"`

`plane_id`는 **같은 물리 평면에 속한 profile을 묶기 위한 식별자**입니다.
평면의 normal 또는 offset을 solver에 알려 준다는 의미가 아닙니다.
Unknown-plane 반복법은 같은 `plane_id`의 points를 함께 재구성해 하나의
평면을 fitting합니다.

## Group과 plane의 역할

`group_id`와 `plane_id`는 서로 다른 축입니다.

- `group_id`: 왜, 어떤 motion 계획으로 취득했는지 구분
- `acquisition_role`: `calibration`, `reference`, `bootstrap` 등 취득 목적
- `motion_kind`: `pure_translation`, `composite`, `circular`,
  `circular_reference`, `general_6dof` 등 운동학적 종류
- `plane_id`: profile이 닿은 실제 평면의 식별자
- `include_in_calibration`: 기본 adapter 선택에 포함할지 여부

예를 들어 circular mode의 두 group은 `group_id`가 다르지만 동일한
`plane_id=0`을 사용합니다. 반대로 three-plane mode는 group마다 서로 다른
`plane_id`를 사용합니다. 하나의 scan은 정확히 하나의 acquisition group에
속하고 `(scan_group_id, sequence_index_in_group)`은 dataset 안에서
유일해야 합니다.

Adapter의 안정적인 순서는 manifest의 group 선언 순서, 그 안에서는
`sequence_index_in_group` 순서입니다. 실제 capture가 비동기적으로 파일에
합쳐져 scan row 순서가 달라져도 prefix 실험의 의미는 유지됩니다.

## 디스크 계약

한 dataset directory는 다음 두 파일로 구성됩니다.

```text
dataset_directory/
├── manifest.json
└── scans.npz
```

`manifest.json`은 사람이 읽고 버전 관리하기 위한 설명과 integrity 정보를
담고, 큰 수치 배열은 `scans.npz`에 저장합니다. Pickle은 사용하지
않습니다.

### `manifest.json`

필수 논리 항목은 다음과 같습니다.

| 항목 | 의미 |
|---|---|
| `schema_name`, `schema_version` | loader compatibility를 위한 형식과 버전 |
| `conventions` | 단위, frame 이름, transform 방향, profile 좌표 규약 |
| `acquisition_mode` | 위 세 취득 모드 중 하나 |
| `source` | `simulation` 또는 실제 capture source |
| `profile_state` | `ideal` 또는 `raw` |
| `channel_id_semantics` | stable sensor channel인지 per-scan ordinal인지 |
| `groups` | group ID, 이름, role, motion kind, 포함 여부와 metadata |
| `metadata` | generator/capture 설정, 독립 seed, 장비 정보 |
| `truth_metadata` | optional GT/environment 설명 |
| `counts` | scan, point, group, plane 개수 |
| `payload` | NPZ 파일명, byte SHA-256, logical SHA-256 |

`conventions`의 기본 pose 규약은 다음과 같습니다.

```text
T_base_ef: end-effector/flange 좌표의 점을 robot-base 좌표로 변환
p_base = T_base_ef @ p_ef
points_s: sensor frame에서 표현한 [x, y, z], 단위 mm
T_ef_s: sensor 좌표의 점을 end-effector/flange 좌표로 변환
T_base_s = T_base_ef @ T_ef_s
```

회전은 오른손 좌표계의 3×3 SO(3), homogeneous matrix의 마지막 행은
`[0, 0, 0, 1]`입니다. 길이 단위가 다른 실제 장비는 writer에서 mm로
변환해 저장하고 원래 단위와 변환 계수를 metadata에 기록합니다.

### `scans.npz`

필수 key는 다음과 같습니다.

| Key | 의미 |
|---|---|
| `scan_id` | dataset 전체에서 유일한 scan ID |
| `plane_id` | scan이 관측한 물리 평면 ID |
| `scan_group_id` | manifest의 acquisition group ID |
| `sequence_index_in_group` | group 내부의 안정적인 취득 순서 |
| `T_base_ef` | solver가 사용할 동기화된 flange pose, `(N,4,4)` |
| `profile_offsets` | concatenated profile에서 scan별 구간을 지정하는 offsets |
| `points_s` | 모든 profile points를 sensor frame으로 이어 붙인 배열 |
| `channel_ids` | 각 point의 원래 sensor channel ID |
| `valid_mask` | 각 point/channel의 유효 여부 |
| `scan_meta_json` | scan별 JSON metadata |

Optional simulation/evaluation truth key는 다음과 같습니다.

```text
T_ef_s_true
T_base_ef_true
T_base_ef_commanded
truth_plane_id
truth_plane_normal_base
truth_plane_offset_mm
truth_plane_has_frame
truth_T_base_plane
```

이 key가 없다는 이유로 loader나 calibration adapter가 실패해서는 안
됩니다. GT를 사용하는 평가는 먼저 `dataset.truth is not None`인지
확인해야 합니다.

## Channel ID와 invalid sample

고정 폭 센서는 반환 실패 channel을 삭제하거나 profile을 재색인하지 말고
원래 행에 NaN을 유지하는 방식을 권장합니다.

```text
channel_ids = [0, 1, 2, 3, ...]
points_s[invalid] = [NaN, NaN, NaN]
valid_mask[invalid] = false
```

가변 길이 profile도 저장할 수 있지만 반환된 point마다 원래
`channel_ids`를 반드시 보존해야 합니다. 같은 scan 안의 channel ID는
중복될 수 없습니다. `stable_sensor_channel`은 서로 다른 scan에서 같은
ID가 같은 ray/pixel/sample channel을 뜻합니다. Tan translation 단계는 이
규약을 요구합니다. `per_scan_ordinal`은 scan별 순번일 뿐이므로 일반
plane-fitting에는 사용할 수 있지만 Tan translation 대응점에는 사용할 수
없습니다.

Noise/dropout을 후단에서 적용할 때도 array를 축소하지 말고 NaN과
`valid_mask`를 갱신하는 것이 가장 안전합니다. 그러면 ideal dataset의
channel 대응과 prefix scan 순서를 유지할 수 있습니다.

## Dataset 생성 CLI 예시

아래 명령은 저장소 루트에서 실행합니다.

Translation/composite:

```bash
PYTHONPATH=robust_laser_handeye python3 \
  robust_laser_handeye/examples/generate_calibration_dataset.py \
  translation-composite \
  --trials 100 \
  --seed 7 \
  --output-dir datasets/translation_composite
```

Trial마다 random plane과 random hand-eye를 사용하려면:

```bash
PYTHONPATH=robust_laser_handeye python3 \
  robust_laser_handeye/examples/generate_calibration_dataset.py \
  translation-composite \
  --plane-mode random \
  --handeye-preset random \
  --plane-angle-range-deg -30 30 \
  --plane-min-axis-angle-deg 1 \
  --plane-distance-range-mm 350 600 \
  --plane-tangent-range-mm -100 100 \
  --trials 100 \
  --seed 7 \
  --output-dir datasets/translation_composite_random
```

Single-plane circular, 기본 81+36 scans:

```bash
PYTHONPATH=robust_laser_handeye python3 \
  robust_laser_handeye/examples/generate_calibration_dataset.py \
  single-plane-circular \
  --trials 100 \
  --seed 7 \
  --output-dir datasets/single_plane_circular
```

Three-plane random, 기본 105 scans:

```bash
PYTHONPATH=robust_laser_handeye python3 \
  robust_laser_handeye/examples/generate_calibration_dataset.py \
  three-plane \
  --trials 100 \
  --seed 7 \
  --output-dir datasets/three_plane_random
```

각 trial은 다음과 같이 저장됩니다.

```text
<output-dir>/
├── collection.json
└── trials/
    ├── trial_000000/
    │   ├── manifest.json
    │   └── scans.npz
    └── trial_000001/
        ├── manifest.json
        └── scans.npz
```

`collection.json`은 trial 목록과 collection 설정을 기록합니다. 각 trial이
저장될 때 원자적으로 갱신되며, 정상 완료 시 `status="complete"`가 됩니다.
중간 실패 시에는 이미 저장된 trial 목록과 `status="in_progress"`가 남습니다.
Hand-eye, environment, motion seed는 master seed와 trial index에서
독립적으로 파생합니다. 정확한 옵션은 다음 명령으로 확인합니다.

```bash
PYTHONPATH=robust_laser_handeye python3 \
  robust_laser_handeye/examples/generate_calibration_dataset.py \
  translation-composite --help
```

기존 이름을 사용하던 명령을 위해 `generate_tan2025_dataset.py`도 같은
범용 CLI로 연결됩니다. 새 코드에서는 알고리즘 독립적인
`generate_calibration_dataset.py` 이름을 사용합니다.

## Calibration adapter 사용

Dataset을 한 번만 로드한 뒤 실험용 noise 복사본을 만들고, 모든 방법에
같은 복사본을 전달합니다.

```python
dataset = load_calibration_dataset(
    "datasets/translation_composite/trials/trial_000000"
)
measured = apply_noise(dataset, noise_config, seed=noise_seed)
```

Tan 2025:

```python
tan_data = measured.to_tan2025_dataset(
    translation_group="translation",
    composite_group="composite",
)
tan_result = Tan2025ClosedFormEstimator().estimate(tan_data)
```

`to_tan2025_dataset()`은 두 group이 같은 plane에 속하는지,
translation/composite motion tag가 맞는지, stable channel identity가
있는지 검증해야 합니다.

Single/multi-plane 반복법:

```python
scans_by_plane = measured.to_scans_by_plane()
result = calibrate_planes(
    scans_by_plane,
    T_init=initial_transform,
    plane_offset_mode="joint",
)
```

이 adapter는 `include_in_calibration=True`인 group의 scan을 `plane_id`별로
합칩니다. 일부 group만 사용할 때는 가정된 `groups=` 인자로 명시적으로
선택합니다.

```python
primary_only = measured.to_scans_by_plane(groups=["primary_ring"])
all_circular = measured.to_scans_by_plane(
    groups=["primary_ring", "reference_ring"]
)
```

## 실제 로봇으로 이식

실제 시스템에는 simulation의 `Scene`, `PosePlanner`, `ProfileAcquirer`를
그대로 옮길 필요가 없습니다. 로봇 운용 코드는 다음 값만 동기화해 공통
`CalibrationDataset`과 동일한 writer로 저장하면 됩니다.

이식에 필요한 경량 경계는 `laser_handeye/data.py`의 `LaserScan`과
`laser_handeye/calibration_dataset/{models.py,io.py}`입니다. 이 저장 경계는
simulation 모듈을 import하지 않습니다. Synthetic pose/profile 생성기인
`generators.py`는 실제 장비 프로그램에 옮길 필요가 없습니다.

1. 로봇에서 읽은 `T_base_ef`
2. 센서에서 읽은 `points_s`
3. 원래 `channel_ids`와 invalid mask
4. operator가 지정한 `group_id`, `plane_id`, group 내부 순서
5. timestamp, 장비 설정, command/readback 상태 등의 metadata

```python
dataset = CalibrationDataset(
    scans=captured_scans,
    scan_group_ids=group_ids,
    sequence_indices=sequence_indices,
    groups=acquisition_groups,
    acquisition_mode="translation_composite",
    source="real_robot",
    profile_state="raw",
    truth=None,
    metadata=capture_metadata,
)
save_calibration_dataset(dataset, output_directory)
```

중요한 pose는 robot command가 아니라 profile timestamp와 동기화된
readback pose입니다. Commanded pose가 필요하면 optional metadata 또는
`T_base_ef_commanded`에 별도로 남기되 `T_base_ef`를 덮어쓰지 않습니다.
실제 plane normal/offset을 모르면 저장하지 않아도 됩니다. 물리적으로 같은
평면에서 취득했다는 `plane_id` grouping만 정확하면 unknown-plane 반복법을
실행할 수 있습니다.

## 현재 한계

- 해석 simulation은 무한 평면을 사용하며 유한 board boundary를 검사하지
  않습니다.
- 기본 pose 생성기는 robot IK, joint limit, self/environment collision을
  검사하지 않습니다.
- Three-plane random mode는 view direction과 profile depth만 확인합니다.
- Circular mode의 단순 workspace-box 검사는 실제 robot feasibility를
  보장하지 않습니다.
- 실제 장비의 trigger/readback 시간 지연, laser occlusion, saturation과
  multi-path는 별도의 capture backend와 validation이 필요합니다.
- 저장 형식의 observability 진단은 좋은 추가 검증이지만, dataset 생성
  성공 자체가 모든 calibration 방법의 rank와 condition을 보장하지는
  않습니다.

실제 로봇 실행 전에는 별도 motion-preflight 계층에서 IK, joint limit,
충돌, 속도/가속도, cable 및 sensor standoff를 검사해야 합니다. 이 검사는
portable dataset 계약과 분리되어야 하며, 거절된 pose와 이유는 capture
metadata에 남기는 것이 좋습니다.
