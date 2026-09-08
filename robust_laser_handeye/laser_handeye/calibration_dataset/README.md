# CalibrationDataset 형식

이 패키지는 알고리즘과 무관하게 로봇 pose와 레이저 profile을 저장하는
경계입니다. 시뮬레이션과 실제 취득 데이터를 같은 `CalibrationDataset`으로
표현합니다.

```python
from laser_handeye.calibration_dataset import load_calibration_dataset

dataset = load_calibration_dataset("path/to/trial")
scans_by_plane = dataset.to_scans_by_plane()
```

## 지원 취득 모드

- `single_plane_circular`: 한 평면의 9개 target line을 따르는 circular pose
- `three_plane_random`: 서로 직교하는 세 평면의 일반 6-DoF random pose

`group_id`는 취득 목적이나 motion 계획을, `plane_id`는 실제 물리 평면을
구분합니다. 한 평면의 primary/reference ring은 group은 다르지만 같은
`plane_id`를 사용합니다.

## 저장 원칙

1. 생성기는 noise-free `ideal` profile과 pose만 저장합니다.
2. 측정 noise, dropout, pose readback noise는 실험 단계에서 복사본에
   적용합니다.
3. GT hand-eye와 plane geometry는 optional `truth`에만 저장합니다.
4. sensor channel은 행 순서로 추측하지 않고 `channel_ids`로 보존합니다.
5. adapter 순서는 manifest의 group 선언 순서와
   `sequence_index_in_group` 순서를 따릅니다.

기본 pose 규약은 다음과 같습니다.

```text
T_base_ef: end-effector 좌표의 점을 base 좌표로 변환
T_ef_s: sensor 좌표의 점을 end-effector 좌표로 변환
T_base_s = T_base_ef @ T_ef_s
길이 단위: mm
```

## 디스크 구조

```text
dataset_directory/
├── manifest.json
└── scans.npz
```

`manifest.json`에는 schema, 좌표계 규약, acquisition group, metadata,
truth 설명과 payload hash를 저장합니다. `scans.npz`의 핵심 배열은 다음과
같습니다.

| Key | 의미 |
|---|---|
| `scan_id` | dataset 내 고유 scan ID |
| `plane_id` | scan이 관측한 물리 평면 ID |
| `scan_group_id` | acquisition group ID |
| `sequence_index_in_group` | group 내부의 안정적인 순서 |
| `T_base_ef` | 동기화된 flange pose, `(N,4,4)` |
| `profile_offsets` | concatenated profile의 scan별 경계 |
| `points_s` | sensor frame의 profile points |
| `channel_ids` | 원래 sensor channel ID |
| `valid_mask` | point/channel 유효 여부 |

Simulation dataset에는 `T_ef_s_true`, true/commanded flange pose와 plane
truth가 추가될 수 있습니다. 이 값이 없어도 loader와 calibration은
동작해야 합니다.

## 생성 CLI

저장소 루트에서 실행합니다.

```bash
# single-plane circular
PYTHONPATH=robust_laser_handeye python3 \
  robust_laser_handeye/examples/generate_calibration_dataset.py \
  single-plane-circular \
  --trials 10 \
  --seed 7 \
  --output-dir datasets/single_plane_circular

# three-plane random
PYTHONPATH=robust_laser_handeye python3 \
  robust_laser_handeye/examples/generate_calibration_dataset.py \
  three-plane \
  --trials 10 \
  --seed 7 \
  --output-dir datasets/three_plane_random
```

출력은 collection manifest와 trial directory로 구성됩니다.

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

기존 결과는 자동으로 덮어쓰지 않습니다. 같은 master seed와 trial index는
같은 generation seed를 사용하며 logical SHA-256으로 입력 동일성을 확인할
수 있습니다.

## 실제 로봇 데이터

실제 capture 코드는 다음 값만 동기화해 같은 writer로 저장하면 됩니다.

- profile timestamp에 대응하는 `T_base_ef` readback pose
- sensor frame의 `points_s`
- 원래 `channel_ids`와 invalid mask
- `group_id`, `plane_id`, group 내부 순서
- timestamp와 장비 설정 metadata

Robot command pose로 readback pose를 덮어쓰지 않습니다. 실제 plane
normal/offset을 몰라도 같은 물리 평면을 나타내는 `plane_id`만 정확하면
unknown-plane calibration을 실행할 수 있습니다.

## 현재 한계

- 해석 simulation은 무한 평면을 사용합니다.
- pose 생성기는 robot IK, joint limit, collision을 검사하지 않습니다.
- 실제 실행 전 별도 motion preflight에서 reachability와 안전 조건을
  검증해야 합니다.
