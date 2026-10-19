# 확장 모드: 세그먼트 기반 인스턴스와 다시점 합의 (D-124)

*상태: 설계 + 오프라인 상한 실험 단계 (2026-10-19). 논문 모드(`tracking.instance_backend: tracks`, 기본값)는 그대로이며,
이 문서는 그 위에 얹는 확장 모드를 적는다. 영어 번역은 결과가 나온 뒤에 만든다.*

## 1. 왜

논문 파이프라인(Supp. Alg. 1)은 **2D 트랙 id가 곧 인스턴스 id**다. 트랙이 끊기면 새 id가 생기고, 이미 본 면에서만
재활성화(Eq. 5)와 흡수(D-111)가 동작하며, 3D에서 다시 합치거나 가르는 단계가 없다. 그 결과가 README TODO 1번이다:
ScanNet 장면당 활성 인스턴스 46개 vs GT 15개, office0에서 소파 2개가 인스턴스 3개로, 테이블 1개가 2개로 나뉜다.
config 레버(D-122 배경 우선 겹침, D-123 새 트랙 문턱)는 40장면에서 효과가 없거나 손해였다.

이 확장은 관측을 보존하고 id를 나중에 **여러 시점의 합의**로 정한다. 기본 아이디어는 MaskClustering(Yan et al., CVPR 2024)의
view consensus rate이고, 온라인 처리를 위해 반대 증거(conflict)와 누적 카운터를 더했다.

## 2. 정의

### 2.1 기본 단위
- **segment** s: 키프레임 f_s에서 마스크 m_s 하나를 들어올린 3D 점 집합 P_s (2 cm 복셀로 줄임). 라벨 관측 (z_s, score)도 같이 저장.
  코드: `supermap.segments.store.Segment`.
- **같은 프레임 중복 제거**: 한 프레임 안에서 마스크 IoU ≥ 0.8인 마스크는 처음부터 segment 하나로 합친다. 라벨 관측은 전부 투표에 들어간다.
  GroundingDINO의 desk/table 중복 박스가 여기서 정리된다. 코드: `segments.dedup.dedup_groups`.
- **keyframe** f: 1 Hz segmentation 프레임마다 pose, K, 1/4 해상도의 depth·유효 마스크(depth 유효 & depth 가장자리 아님, D-116 규칙)·
  마스크 id 영상을 저장한다. 5분 시퀀스면 약 300장. 코드: `segments.store.Keyframe`.

### 2.2 프레임 f에서 segment s의 판정
- **Visible**: P_s를 f의 pose로 투영해 영상 안이고 `valid`이며 |d_투영 − D_f(u)| < τ_d (0.08 m)인 점을 V(s,f)로 둔다.
  |V| / |P_s| ≥ α (0.3) 이고 |V| ≥ n_min (30)이면 "보인다". 가려졌거나 시야 밖이면 그 프레임은 증거 없음.
- **Contained**: V(s,f)의 점이 f의 마스크들에 어떻게 나뉘는지 센다. 한 마스크 m에 β (0.8) 이상이면 M(s,f) = m, 아니면 M(s,f) = 없음.
  코드: `segments.visibility` (`segment_histograms`, `is_visible`, `contained_mask`).

### 2.3 두 segment(또는 클러스터) a, b의 합의
둘 다 보이는 프레임만 센다.
- support: M(a,f) = M(b,f) ≠ 없음.  conflict: 둘 다 마스크에 들어갔는데 서로 다름.
- consensus(a,b) = support / (support + conflict).
- merge ⇔ support + conflict ≥ k_min (3) 이고 consensus ≥ θ.
- 조각난 table: 전체가 보이는 프레임들에서 한 마스크에 같이 들어가 support가 쌓인다. 붙어 있는 의자 둘: SAM2가 매번 따로 내므로 conflict가 쌓여
  합쳐지지 않는다. 공통 시점이 k_min 미만이면 보류한다. 코드: `segments.consensus`.

### 2.4 클러스터링
- 후보 쌍은 3D bbox를 5 cm 넓혀 겹치는 것만 본다.
- 단순 연결 성분은 쓰지 않는다(A–B, B–C로 A와 C가 엮이는 chaining). θ를 0.95 → 0.9 → 0.85로 낮춰 가며 반복 병합하고,
  병합할 때마다 **클러스터 단위로** 히스토그램(멤버 합)을 다시 계산해 Visible/Contained를 재판정한다.
- 배경 클래스(wall, floor)는 거의 모든 것을 담으므로 합의 대상에서 빼고 복셀 겹침 연결 성분으로만 묶는다.
- 과소분할 마스크(박스 하나가 의자 둘을 덮음) 대책: 이미 conflict로 구분이 확실한 두 클러스터를 한꺼번에 담은 마스크는 그 프레임에서
  support를 주지 않는다(`distrust_split_masks`, 기본 끔, 실험으로 결정).
- ByteTrack은 버리지 않는다. 온라인 모드에서 "연속 프레임에서 같은 track"은 support 1회의 사전 증거로만 더한다. 최종 id는 클러스터가 정한다.
  코드: `segments.cluster.consensus_cluster`, 파라미터 `segments.params.ConsensusParams`.

### 2.5 인스턴스 만들기
클러스터 → `ObjectInstance`: 점은 멤버 합집합(2 cm 복셀 중복 제거), 라벨 분포는 멤버 라벨 관측의 score 합을 정규화,
first/last_seen은 멤버 t의 min/max. 코드: `segments.build.clusters_to_instances`.

## 3. 단계

1. **오프라인 상한 실험** (지금): `experiments/scripts/segments_offline.py`. 검출 → NMS(D-116) → SAM2(점수 ≥ 0.35 박스) → 중복 제거 →
   겹침 해소(D-107) → Unproject(D-106/D-116 정리) 까지는 논문 파이프라인과 같은 객체를 쓰고, 그 뒤를 트랙 대신 세그먼트로 저장한다.
   클러스터링은 전체 키프레임을 한 번에 본다(카운터 대신 배치). 출력은 표준 폴더 형식이라 `eval_replica_semseg.py`를 그대로 쓴다.
   8개 chair 선별 장면에서 θ, α, β, k_min, τ_d를 훑고 트랙 모드(`reports/scannet_ablation/i15_d120`)와 비교한다.
2. **온라인 백엔드**: `tracking.instance_backend: segments`. 새 키프레임마다 시야 안 클러스터만 Visible/Contained를 계산해 쌍별 카운터를 더하고,
   몇 초마다(또는 클러스터 안 consensus가 θ 아래로 떨어지면 그 클러스터만) 다시 클러스터링한다. id 영속성: 이전 클러스터링과 복셀 겹침으로
   매칭해 물려주고, 갈라지면 큰 쪽이 id 유지, 합쳐지면 오래된 id 유지. Eq. 8 점 log-odds, 존재 log-odds(D-113), 사라짐 판정은 세그먼트 단위로
   들고 클러스터가 집계한다.
3. **확장**: 40장면 + 변화 검출 58시퀀스 + 실시간 벤치마크. README 결과 표에 모드별로 나란히 적는다.

## 4. 열린 질문 (실험으로 답한다)
- τ_d: ScanNet 경계 depth가 부정확하다. 0.05 ~ 0.1 중 어디가 안정적인가.
- 마스크를 줄 박스의 점수 하한(0.35). 낮추면 세그먼트가 늘고 합의 증거도 늘지만 벽 오검출도 는다.
- 과소분할 마스크 대책을 켜야 하는가. 켜면 support가 줄어 조각 병합이 늦어질 수 있다.
- 배경 클러스터의 평가 영향(with-bg 지표).
- 메모리: 세그먼트 점을 전부 들고 있을 수 있는가, 아니면 N 프레임 지난 세그먼트를 클러스터에 확정시켜 압축해야 하는가.
