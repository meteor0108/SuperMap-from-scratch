# SuperMap 재구현: 구현 기록과 실험이 정한 기본값

*English (default): [IMPLEMENTATION.md](IMPLEMENTATION.md)*

이 문서는 "SuperMap: A Spatio-Temporal SLAM System for Visual-Language Navigation"(RSS 2026) 재구현의 **엔지니어링 로그**입니다. 대표 결과는 [README.ko.md](../README.ko.md)에 있고, 여기에는 알고리즘 설명과 어떻게 만들었는지, 논문이 비워 둔 세부를 무엇으로 정했는지, 어떤 실험이 각 기본값을
정했는지, 시도했다가 버린 것, 평가 프로토콜, 반복(iteration)별 결과, 한계를 모았습니다.
근거 우선순위는 논문 > 논문에서 뽑은 요구사항 명세 > 결정 기록이며, 공식 코드는 참고하지 않았습니다.
결정 번호(D-xxx), 이슈 번호(I-xx-x), 요구사항 번호(REQ-xxx), iteration 번호는 개발 과정의 기록(10절)을 가리키며 그대로 남겼습니다.

## 목차

1. [알고리즘](#1-알고리즘)
2. [범위 결정](#2-범위-결정)
3. [코드 구조](#3-코드-구조)
4. [파라미터](#4-파라미터)
5. [논문이 비워 둔 세부와 선택](#5-논문이-비워-둔-세부와-선택)
6. [시도했으나 채택하지 않은 것](#6-시도했으나-채택하지-않은-것)
7. [평가 프로토콜](#7-평가-프로토콜)
8. [결과 이력](#8-결과-이력)
9. [한계와 열린 이슈](#9-한계와-열린-이슈)
10. [개발 하네스](#10-개발-하네스)

모든 수치는 별도 표기가 없으면 ScanNet v2 validation 40장면 평균이며 mIoU는 배경(wall, floor) 제외, 라벨 전사 반경 r = 0.05 m입니다.
파이프라인은 결정적이라 같은 설정은 같은 결과를 냅니다.

---

## 1. 알고리즘

프레임마다 논문 보충자료의 Algorithm 1 순서대로 처리합니다. \
Sec. V-H의 주기에 따라 네 모듈이 비동기로 돕니다: pose 10 Hz, 2D segmentation 1 Hz, dense mapping 3 Hz, scene graph 5 Hz. \
데이터 타임스탬프로 구동되는 스케줄러가 이 주기를 재현하며, 무거운 모듈(검출, segmentation)은 자기 주기의 tick에서만 실행됩니다.

```
RGB C_t, depth D_t, intrinsics K, pose
 |- Geometric layer    pose provider -> Eq.(3) P_t = T_WB * T_BC -> back-projection -> colorized dense map (2 cm voxels)
 |- Instance layer     GroundingDINO(C_t, prompts) -> detections
 |                     neighbor objects O_nbr of the map around P_t
 |                     depth comparison (Eq. 7, 9) -> O_front / O_on / O_behind, O_neg <- O_front
 |                     Kalman prediction, 3D-centroid prior (Eq. 5, 6), ByteTrack association -> tracks
 |                     SAM2(C_t, track boxes) -> masks -> Unproject -> O_add
 |                     semantic consistency -> O_neg;  ObjNegate, ObjMerge, MapMerge
 |                     Eq.(8) point log-odds, object existence log-odds, pruning, Eq.(10) label fusion
 `- Topological layer  clustering -> On / Beside edges, temporal edges -> serialization -> VLM -> <answer> ids -> waypoints
```

### 1.1 기하 층 (Sec. IV-A, Eq. 3)

오도메트리가 body pose T_WB를 주고, 고정 extrinsic T_BC로 카메라 pose P_t = T_WB T_BC를 만듭니다(Eq. 3). \
각 depth 영상은 intrinsics로 역투영(z-depth, 유효 depth > 0)한 뒤 world 좌표계로 변환합니다. \
점들은 2 cm voxel당 한 점만 유지하는 색 입힌 dense map에 누적됩니다. \
이 dense map이 기하 층이며, 인스턴스 층이 이후 새 depth와 비교하는 점들의 출처입니다.

### 1.2 Open-vocabulary 검출 (Sec. IV-B, V-B)

GroundingDINO가 텍스트 prompt 목록으로 객체를 검출하며, prompt 목록이 라벨 집합을 정합니다. \
각 질의 구문의 클래스 점수는 그 구문의 최대 토큰 확률이고, 박스 라벨은 점수가 가장 높은 클래스입니다. \
낮은 box threshold(0.1)는 약한 검출도 tracker의 low-score 단계까지 도달하게 합니다. \
클래스마다 추가 구문(alias)을 둘 수 있으며, alias는 검출기에만 전달되고 박스는 그 alias가 속한 클래스로 라벨링됩니다.

### 1.3 이웃 객체와 depth 비교 (Sec. IV-B3, Eq. 7-9)

이웃 집합 O_nbr은 현재 pose에서 보일 수 있는 지도 객체입니다. \
객체당 최대 256개 점을 샘플해 투영하고, 그중 하나라도 10 m 이내에서 영상 안에 들어오면 이웃으로 봅니다. \
이 객체들의 점을 현재 depth 영상에 투영해, 투영 depth와 측정 depth의 잔차로 각 점을 Observable, Disappeared(측정 표면이 점보다 뒤), Unobservable(측정 표면이 점보다 앞)로 분류합니다. \
임계값은 tau_eps = 0.1 m입니다 (Eq. 9). \
시야 밖이거나 depth가 무효인 점은 갱신하지 않습니다. \
객체 수준 상태는 점 상태의 다수결이며, 객체는 O_front, O_on, O_behind로 나뉩니다. \
O_front 객체는 강화 대신 부정(negate)되는 집합 O_neg에 들어갑니다.

### 1.4 하이브리드 2D-3D 추적 (Sec. IV-B2, Eq. 4-6)

각 track은 6차원 상태 S(박스 중심, 크기, 중심 속도)를 가지며 등속 Kalman 예측을 씁니다(Eq. 4, 6). \
살아 있는 track이 있는 O_on 객체는 3D centroid를 현재 영상에 투영한 값이 예측 중심을 대체하므로(Eq. 5), 2D 운동 모델이 실패해도 track이 카메라 움직임을 따라갑니다. \
현재 프레임에서 Disappeared로 표시된 점은 이 centroid에서 제외합니다. \
살아 있는 track이 없는 O_on 객체는 투영된 centroid와 크기로 후보 track으로 재활성화되고, 매칭되면 자기 id를 유지합니다. \
연관은 ByteTrack을 따릅니다. \
첫 단계는 고점수 검출과 track을 IoU로 매칭하며 비용에 클래스 항을 넣고, 둘째 단계는 남은 track을 저점수 검출과 매칭합니다. track id가 곧 인스턴스 id입니다.

### 1.5 SAM2 마스크와 Unproject (Alg. 1, 19-20행)

SAM2가 track 박스마다 마스크 하나를 만듭니다. depth로 마스크를 들어올리기 전에 마스크를 2 px 침식하고(절반 넘게 사라지면 침식을 취소), depth 불연속 경계의 픽셀을 버리고, 3-sigma median-absolute-deviation 규칙으로 depth 이상치를 제거하며, 두 마스크가 겹친 픽셀은 더 작은 마스크에 줍니다. \
남은 픽셀을 world 좌표 점으로 역투영한 것이 O_add입니다.

### 1.6 ObjMerge와 MapMerge (Sec. IV-B)

논문은 3D 공간 일관성으로 모든 검출에 고유 id를 부여합니다. track id가 이미 있는 검출은 그 인스턴스에 점을 더하며 2 cm voxel로 중복을 제거합니다. \
새 id의 검출은 점의 절반 이상이 어떤 이웃 인스턴스의 voxel 안에 들어 있으면 클래스와 무관하게 그 인스턴스에 흡수되고, 아니면 새 인스턴스가 됩니다. \
병합 후 각 인스턴스는 점의 최대 연결 성분(10 cm voxel, 26-연결)만 남겨 떠도는 조각을 없앱니다. \
MapMerge는 갱신된 이웃 객체를 전역 지도에 되씁니다.

### 1.7 확률적 기하 갱신과 pruning (Sec. IV-B3, Eq. 8)

각 점은 log-odds를 가지며 프레임마다 한 번 Eq. (8)로 갱신됩니다. \
P(o|Q)는 Observable 0.7, Disappeared 0.3, Unobservable 0.5입니다. \
O_neg 객체는 시야 안의 점에만 부정 갱신을 받습니다. log-odds가 -1 아래인 점은 제거합니다. \
점 단위 규칙으로는 객체 전체가 사라졌음을 알 수 없으므로 객체도 존재 log-odds를 가집니다. \
Disappeared이거나 O_neg인 프레임은 logit(0.3)을, Observable 프레임은 logit(0.7)을 더하고, 값은 [-3, 3]으로 clamp하며, -1 아래가 되면 객체를 REMOVED로 표시하고 제거 이벤트를 냅니다. \
제거된 인스턴스는 scene graph가 변화를 표현할 수 있도록 기록으로 지도에 남습니다.

### 1.8 베이지안 라벨 융합 (Sec. IV-B4, Eq. 10)

각 인스턴스는 클래스 분포를 가집니다. \
첫 관측에서 시작해, observable 인스턴스에 새 검출이 들어올 때마다 confusion matrix(대각 0.8, 비대각 균등)를 쓴 Bayes 규칙으로 갱신합니다(Eq. 10). \
최선 클래스 사후확률이 0.4 아래로 떨어진 인스턴스는 제거합니다. \
검출을 시간에 걸쳐 융합하므로 한 프레임에서 틀린 라벨이 이후 프레임에 의해 뒤집힙니다.

### 1.9 4D scene graph와 VLM 질의 (Sec. IV-C, IV-D)

인스턴스는 centroid 거리로 군집화합니다(single linkage, 3 m). \
군집 안에서 공간 술어 On(xy 겹침이 gamma = 0.05 초과이고 수직 접촉이 delta = 0.05 m 이내)과 Beside(0.5 m 이내)를 계산합니다. \
객체가 0.1 m 이상 움직이면 새 시간 노드를 만들어 이전 노드와 시간 엣지로 잇기 때문에, 그래프에는 사물이 어디에 있었고 언제 움직이거나 사라졌는지가 기록됩니다. \
로봇 주변(5 m)의 국소 subgraph를 노드, 엣지, 시간 단서의 텍스트 스키마로 직렬화해 질의와 함께 VLM(Gemini)에 보냅니다. \
답은 `<answer>` 태그에서 읽고, 안의 id를 3D 웨이포인트(객체 위치)로 변환합니다. \
Gemini 키가 없으면 VLM 호출만 건너뛰고 직렬화 결과는 그대로 저장됩니다.

---

## 2. 범위 결정

논문 Fig. 2의 세 층(Geometric / Instance / Topological)을 모두 구현했습니다. 범위 결정은 사람이 내렸고 DECISIONS에 기록했습니다.

| 결정 | 내용 |
|---|---|
| D-001 | SuperOdometry(LiDAR-IMU, ROS)는 통합하지 않는다. `PoseProvider` 인터페이스로 **데이터셋 pose**를 쓰고, Eq. (3)의 extrinsic 합성 P_t = T_WB T_BC는 구현한다(D-009: T_WB = 데이터셋 camera-to-world pose, T_BC = I). 입력은 RGB-D 비디오. |
| D-002 | 개발·스모크는 Replica office0, 주 벤치마크는 ScanNet v2(사람이 ToS 승인 후 다운로드, D-109). 논문의 실로봇 데이터는 비공개라 REQ-008은 N/A(D-014). |
| D-003 | VLM(Gemini)은 플러그형 백엔드. 직렬화, `<answer>` 파서, 웨이포인트 변환은 결정론적으로 검증한다. |

입력 스트림은 RGB `C_t`(H×W×3 uint8), depth `D_t`(H×W float32, m), 내부 파라미터 K, pose입니다. ScanNet은 30 fps 중 stride 3으로 **10 Hz pose 스트림**을
만들고(D-109, D-013: color 1296×968을 depth 640×480으로 리사이즈, depth는 uint16 mm), Replica는 명목 10 Hz입니다(D-006). pose 모듈이 프레임 시계이며
10 Hz보다 빠르게 들어온 프레임은 건너뜁니다(D-088).

---

## 3. 코드 구조

파이프라인은 실험 코드에 의존하지 않습니다. `src/supermap` + `configs/default.yaml`만으로 `python -m supermap.run`이 돕니다. 모듈 의존은 아래 순서(하위에서
상위)로만 흐르고, 무거운 모델 로딩은 객체 생성 시점으로 미룹니다.

### 3.1 패키지 맵

| 경로 | 구현 내용 | 논문 위치 |
|---|---|---|
| `types.py`, `map.py`, `camera.py`, `pose.py`, `config.py` | 공용 자료형(Observation, ObjectInstance, GlobalMap), 카메라 투영, `PoseProvider` | Sec. III |
| `datasets/` (`base.py`, `replica.py`, `scannet.py`) | RGB-D 시퀀스 로더, 명목 시각, 무효 pose 제외 | Sec. V-B |
| `geometry/` (`extrinsics.py`, `backproject.py`, `dense_map.py`) | Eq. (3) 합성, 역투영(OpenCV 축, z-depth), 2 cm 복셀 dense map | Sec. IV-A, Eq. 3 |
| `perception/` (`detector.py`, `prompts.py`, `nms.py`, `segmenter.py`, `observation.py`) | GroundingDINO 검출과 프롬프트·별칭, 클래스 무관 NMS, SAM2 박스 프롬프트, 스트림 원소 S | Sec. IV-B, Alg. 1 line 3 |
| `tracking/state.py`, `bytetrack.py` | 6차원 KF(Eq. 4, 6), ByteTrack 2단계 연관, UpdateFromMap(Eq. 5), 재활성화, 클래스 비용 | Sec. IV-B2, Eq. 4-6 |
| `tracking/neighbors.py` | GetNeighborObjects(표본 점 기준) | Alg. 1 line 6 |
| `tracking/instance_ops.py`, `mask_ops.py` | Unproject, 마스크 침식·MAD·겹침 해소, 흡수 계획(`absorb_targets`), ObjMerge, MapMerge | Alg. 1 lines 19-20, 27-29 |
| `tracking/components.py` | 최대 26-연결 성분만 유지(`prune_components`, D-118) | 논문 외 선택 |
| `tracking/depth_edges.py` | 깊이 가장자리(flying pixel) 마스크(D-116) | 논문 외 선택 |
| `tracking/gate3d.py`, `footprint.py`, `map_match.py`, `contact.py` | 3D 중심 게이트, 발자국 박스, 지도 매칭, 3D 접촉 흡수. 모두 **기본 꺼짐**, 실험용(6절) | 논문 외 선택 |
| `consistency/` (`residual.py`, `classify.py`, `compare_depth.py`, `logodds.py`, `negate.py`, `update.py`, `existence.py`, `params.py`) | Eq. (7) 잔차, Eq. (9) 상태, CompareDepth, Eq. (8) log-odds, ObjNegate, 가지치기, 객체 존재 log-odds(D-113) | Sec. IV-B3, Eq. 7-9 |
| `semantics/` (`confusion.py`, `fusion.py`, `update.py`) | Eq. (10) 베이즈 라벨 융합, 저확신 제거 | Sec. IV-B4, Eq. 10 |
| `graph/` (`cluster.py`, `predicates.py`, `scene_graph.py`, `serialize.py`) | 클러스터링, On/Beside/Under, 시간 노드·엣지, 직렬화 | Sec. IV-C, IV-D |
| `vlm/` (`backend.py`, `parser.py`, `waypoints.py`, `grounding.py`) | Gemini 백엔드와 fallback, `<answer>` 파서, 웨이포인트 | Sec. IV-D, V-F |
| `pipeline.py` | 프레임 처리 순서(Supp. Alg. 1) | Alg. 1 |
| `scheduler.py` | 모듈 주기 10 / 1 / 3 / 5 Hz 시뮬레이션 | Sec. V-H |
| `run.py` | 엔트리(`python -m supermap.run`), 출력 쓰기 | – |

### 3.2 프레임 처리 순서 (`Pipeline._instance_step`, segmentation 프레임)

1. 검출: GroundingDINO가 `B_det`를 만든다(스트림 원소 S = (B_det, I, D_I, Odom), 후크 없음, D-095).
2. 이웃: `O_nbr = GetNeighborObjects(M, Odom)` (D-110).
3. CompareDepth: Eq. (7), (9)로 점 상태를 만들고 객체를 `O_front / O_on / O_behind`로 나눈다. `O_neg ← O_front`.
4. 추적: 트랙 predict(Eq. 6) → UpdateFromMap(O_on 중 살아 있는 트랙, Eq. 5) → 재활성화(D-035) → ByteTrack Associate → `T_trk`.
5. SAM2: `T_trk` 박스를 프롬프트로 마스크를 만든다.
6. Unproject: 겹침 해소(D-107), 침식과 MAD 이상치 제거(D-106), 깊이 가장자리 제외(D-116) 후 역투영해 `O_add`.
7. 흡수 계획: D-111 흡수 대상을 순수 함수 `absorb_targets`로 계산한다(D-120).
8. 의미 검사: 보이는데 `T_trk`에 없고 이번 프레임 관측도 흡수하지 않는 `O_on` 객체를 `O_neg`에 넣는다(Alg. 1 lines 21-25, D-120).
9. ObjNegate + Eq. (8): `O_neg`는 ObjNegate(시야 안 점에 p_dis), 나머지는 점 상태별 inverse sensor 갱신, 점당 프레임당 1회(D-051). 객체 존재 log-odds도 여기서 갱신(D-113).
10. ObjMerge: 같은 id는 점 추가(2 cm 복셀 중복 제거), 새 id는 7번 계획대로 흡수(D-111), Eq. (10) 라벨 융합(D-089). 이어서 갱신된 인스턴스의 최대 연결 성분 가지치기(D-118).
11. MapMerge: `O_nbr`를 지도에 쓰고 이웃 밖 같은 id는 점만 추가.
12. 가지치기 / 존재 / 저확신 제거: L < L_prune 점 제거, L_exist < -1이면 REMOVED, max_c P < p_min이면 REMOVED, REMOVED 객체의 트랙 삭제(D-089).

segmentation이 없는 프레임은 트랙 predict만 한다(D-085). mapping 프레임은 dense map 적분에 더해 `O_nbr`에 대한 기하 일관성(Eq. 8, 가지치기)만 수행한다
("geometric only", D-052, D-090).

### 3.3 스케줄러

논문 Sec. V-H의 주기(pose 10 Hz, segmentation 1 Hz, mapping 3 Hz, scene graph 5 Hz)를 **타임스탬프 기반 시뮬레이션**으로 재현합니다(D-087).
스레드는 없고 한 루프에서 `ModuleScheduler.due(module, stamp)`로 모듈을 켭니다. 슬롯 격자 방식(D-086): slot k = floor((stamp − t0 + 1e-6) / T_m)이고 아직 서비스하지
않은 슬롯에 도달하면 실행합니다. "마지막 실행 후 T_m 경과" 규칙은 SPEC의 10 Hz × 100 프레임 사례에서 mapping을 30이 아니라 25번 돌려서 쓰지 않았습니다. 이 격자에서
1 Hz 슬롯은 3 Hz 슬롯의 부분집합이라 segmentation 프레임은 항상 mapping 슬롯입니다. 실제 지연은 8.5절 벤치마크가 모듈별로 측정합니다.

---

## 4. 파라미터

모든 값은 `configs/default.yaml`에 REQ 번호·논문 위치·결정 번호 주석과 함께 있고 아래 기본값을 직접 대조했습니다. ScanNet 평가는
평가용 설정을 쓰며 default와 다른 것은 프롬프트 19개, 별칭, VLM 키 환경변수 이름뿐입니다. **논문이 수치를 준 것은 모듈 주기뿐**이고
나머지는 모두 우리가 정했습니다.

### 4.1 기하 / 검출 / 분할 / 라벨 융합

| config 키 | 기본값 | 의미 | 근거 |
|---|---|---|---|
| `rates.pose / segmentation / mapping / scene_graph` | 10 / 1 / 3 / 5 Hz | 모듈 주기 | 논문 Sec. V-H, Supp. VII-D (유일한 논문 수치) |
| `extrinsics.T_BC` | I_4 | body-카메라 extrinsic | D-009 |
| `geometry.depth_min / depth_max` | 0.0 (초과) / null | 유효 depth 범위 | D-015 |
| `geometry.dense_map.voxel_size` | 0.02 m | dense map 복셀, 복셀당 첫 점 유지 | D-016 |
| `perception.prompts` | 23개 = ScanNet 19개 + office 4개 (plant, monitor, trash can, lamp) (ScanNet 평가 설정: 19개) | 라벨 집합 c | D-121, D-109 |
| `perception.prompt_aliases` | cabinet, picture, bookshelf, refridgerator 별칭 (ScanNet 평가 설정과 동일) | 검출 문구만 추가, 라벨은 클래스명 | D-115 |
| `perception.detector.model_id` | `IDEA-Research/grounding-dino-base` | 검출기 | D-114 (D-004는 tiny) |
| `perception.detector.box_threshold` | 0.1 | = ByteTrack `low_thresh` | D-097 (D-020은 0.35) |
| `perception.detector.nms_iou` | 0.5 | 클래스 무관 greedy NMS, D_high/D_low 분리 전 | D-116 (D-020 개정) |
| `perception.segmenter.model_id` | `facebook/sam2.1-hiera-small` | SAM2, 박스당 마스크 1개, logit 임계 0.0 | D-005, D-023 |
| `semantics.confusion.diag_prob` | 0.8 | confusion matrix 대각, 비대각 균등 | D-061 |
| `semantics.p_min` | 0.4 | max_c P(L=c) 미만이면 인스턴스 제거 | D-063 |

### 4.2 기하 일관성 (Eq. 7-9)

| config 키 | 기본값 | 의미 | 근거 |
|---|---|---|---|
| `consistency.tau_eps` | 0.1 m | Eq. (9) 잔차 임계 | D-049 |
| `consistency.sigma` | 0.05 m | 가우시안 우도 함수용, 분류에는 미사용 | D-049 |
| `consistency.p_obs / p_dis / p_unobs` | 0.7 / 0.3 / 0.5 | 상태별 inverse sensor 확률 (0.5 = 무갱신) | D-049 |
| `consistency.L_prune` | −1.0 | 점 가지치기 임계 (L < L_prune) | D-049 |
| `consistency.L_init / L_clamp` | null / null | 초기값 = logit(p_obs), 클램프 없음 | D-048 |
| `consistency.exist_clamp` | 3.0 | 객체 존재 log-odds 대칭 클램프 | D-113 |
| `consistency.exist_prune` | −1.0 | L_exist가 이 값 미만이면 객체 REMOVED (null이면 규칙 끔) | D-113 |
| `consistency.exist_semantic` | true | semantic 증거(O_neg)도 L_exist에 반영 | D-113 |
| `consistency.verdict_window_px` | 0 | 객체 판정 창 탐색(꺼짐, I-9-4 변형) | iter 10 |
| `consistency.exist_min_points / exist_min_share` | 0 / 0.0 | 존재 갱신 커버리지 게이트(꺼짐, I-9-4 변형) | iter 10 |

### 4.3 추적, 병합

| config 키 | 기본값 | 의미 | 근거 |
|---|---|---|---|
| `tracking.kalman` | 6차원, dt 1, Q 25 / 0.390625, R 25, P0 100 / 39.0625 | ByteTrack 상수를 기준 높이 100 px에서 고정 | D-030, D-031 |
| `tracking.bytetrack.high_thresh / new_track_thresh` | 0.35 / 0.35 | D_high 기준, 새 트랙 기준 | D-097 |
| `tracking.bytetrack.low_thresh` | 0.1 | D_low 기준 | D-097 |
| `tracking.bytetrack.match_thresh / second_match_thresh` | 0.8 / 0.5 | 1차 / 2차 매칭 (1 − IoU 상한) | D-033 |
| `tracking.bytetrack.track_buffer` | 30 | 미매칭 허용 프레임 | D-034 |
| `tracking.bytetrack.label_cost` | **1.0** | 라벨이 다르면 1차 연관 비용에 더함 (match_thresh 0.8이므로 하드 클래스 게이트) | D-117 (D-108의 0.0 개정) |
| `tracking.bytetrack.gate_3d` | 0.0 (꺼짐) | 3D 중심 거리 게이트 | D-117 시험, D-119 |
| `tracking.bytetrack.footprint_boxes / map_match_iou` | false / 0.0 | 발자국 박스, 지도 매칭 (꺼짐) | iter 12 |
| `tracking.motion_compensation / reactivation` | true / true | Eq. (5) 보상, 만료 트랙 재활성화 | D-030, D-035 |
| `tracking.neighbors.mode / sample_points / max_range` | points / 256 / 10 m | GetNeighborObjects | D-110 |
| `tracking.unproject.min_points` | 1 | 관측 객체 최소 점 수 | D-036 |
| `tracking.unproject.erode_px / erode_min_keep` | 2 px / 0.5 | 마스크 침식, 50 % 미만 남으면 취소 | D-106 |
| `tracking.unproject.depth_outlier_k / floor` | 3.0 σ_MAD / 0.05 m | 깊이 이상치 제거 | D-106 |
| `tracking.unproject.resolve_overlaps` | smaller | 겹친 픽셀은 작은 마스크에 귀속 | D-107 |
| `tracking.unproject.edge_jump / edge_radius / edge_rel` | 0.1 m / 1 / 0.0 | 3×3 창의 깊이 범위(max − min)가 0.1 m 초과이거나 무효 깊이를 포함하면 픽셀 제외 | D-116 |
| `tracking.unproject.component_voxel` | null (꺼짐) | 관측 단위 연결 성분 필터 | D-118 시험 |
| `tracking.merge.voxel_size` | 0.02 m | 병합 점 중복 제거 | D-101 |
| `tracking.merge.absorb_overlap` | 0.5 | 새 id 점의 복셀 겹침 비율 이상이면 기존 객체에 흡수, 라벨 무관 | D-111 |
| `tracking.merge.absorb_same_label` | false | 같은 라벨 이웃에만 흡수 (꺼짐) | D-119 |
| `tracking.merge.absorb_overlap_rev` | null (꺼짐) | 이웃도 이 비율 이상 덮일 때만 흡수 | D-119 |
| `tracking.merge.component_voxel` | **0.10 m** | ObjMerge 후 점을 받은 인스턴스는 최대 26-연결 성분만 유지 | D-118 (D-036 개정) |
| `tracking.merge.contact_dist / contact_frac / contact_same_label / contact_exclude` | 0.0 (꺼짐) / 0.5 / false / [wall, floor] | 3D 접촉 흡수 | iter 13 |

### 4.4 장면 그래프, VLM, 데이터셋

| config 키 | 기본값 | 의미 | 근거 |
|---|---|---|---|
| `scene_graph.d_clu` | 3.0 m | 단일 연결 클러스터링 | D-070 |
| `scene_graph.gamma / delta` | 0.05 / 0.05 m | On의 IoU_xy 임계, z 접촉 허용 오차 | D-071 |
| `scene_graph.d_bes` | 0.5 m | Beside 수평 간격 상한 | D-073 |
| `scene_graph.d_move` | 0.1 m | 이 이상 움직이면 시간 노드 | D-074 |
| `scene_graph.subgraph_radius` | 5.0 m | 로컬 서브그래프 | D-079 |
| `vlm.model / temperature` | gemini-2.0-flash / 0.0 | 논문 Sec. V-F | D-080 |
| `vlm.fallback_models` | gemini-flash-lite-latest, gemini-3.8-flash, gemini-flash-latest, gemini-2.5-flash | 404일 때만 다음 모델 | D-099, D-100 |
| `datasets.scannet.depth_scale / nominal_fps` | 1000.0 / 30.0 | mm 단위, 촬영 30 Hz | D-013 |
| `datasets.replica.nominal_fps` | 10.0 | Replica에는 timestamp가 없음 | D-006 |

---

## 5. 논문이 비워 둔 세부와 선택

각 항목은 결정 ID, 선택, 측정된 효과, 대안 순서입니다.

### 5.1 pose와 extrinsic (D-001, D-009, D-018)
SuperOdometry 없이 데이터셋 camera pose를 T_WB로 두고 T_BC = I로 합성합니다. 합성 코드는 `geometry/extrinsics.py` 한 곳입니다. 효과를 비교할 대안은 없습니다.
ScanNet pose가 float32라 일부 pose가 SE(3) 허용오차 1e-6에 걸려 제외됩니다(2374프레임 중 3개, I-8-4, 평가 영향 없음).

### 5.2 프롬프트와 별칭 (D-025, D-109, D-115, D-121)
논문의 프롬프트 목록은 비공개입니다. ScanNet은 ScanNet20에서 otherfurniture를 뺀 19개(GT 철자 유지, `refridgerator`)를 쓰고 background = wall, floor입니다.
GroundingDINO-base는 "cabinet"(최대 토큰 확률 0.31)과 "picture"(0.19)를 ByteTrack high 0.35 위로 올리지 않았고 "cupboard", "framed picture", "painting"에는 반응했습니다.
그래서 `prompt_aliases`가 별칭을 검출 텍스트에만 넣고 라벨은 클래스명으로 붙입니다(C, Eq. 10 불변). 효과(별칭 전체): mIoU 48.8 → 54.8, mAP50 21.2 → 27.5, cabinet 0 → 33 IoU,
picture 0 → 65. 대안: "office desk" 별칭은 desk 11 → 48이지만 table 39 → 25, bed 60 → 50이라 제외, "curtains"는 shower curtain을 먹어 제외. DualMap 부록 I의 GPT-4o 실내 어휘를 참고했습니다. \
`configs/default.yaml`의 기본 목록은 처음에 office0 스모크용 7개(D-025)였는데, office0 전체 시퀀스에서 배경 단어가 없어 벽·창·러그가 table / plant로 붙었습니다. 그래서 기본값을 ScanNet 19개 + office 4개(plant, monitor, trash can, lamp)의 23개와 D-115 별칭으로 바꿨습니다(D-121). 평가 설정의 목록은 그대로입니다.

### 5.3 체크포인트 (D-114, D-004, D-005)
처음엔 가장 단순한 tiny / small(D-004, D-005)이었고, 사람의 지시(논문 하드웨어 RTX 4090 Laptop 16 GB에 들어가는 가장 큰 모델)로 비교했습니다.
GroundingDINO-base 단독: mIoU 41.1 → 48.8, Acc 46.5 → 57.0, mAP50 18.1 → 21.2, 인스턴스 69 → 47. SAM2.1-large는 단독 41.6(효과 없음), base와 같이 쓰면 −1.9.
채택: 검출기 base, 분할기 small. GPU 피크 4.9 GB, 세그 프레임당 79 + 60 ms.

### 5.4 검출 임계값과 NMS (D-020, D-021, D-097, D-116)
- 라벨 규칙(D-021): 쿼리별 클래스 점수 = 해당 문구 토큰 확률의 최댓값, 라벨 = argmax. 점수는 Eq. (10) 가중치로 쓰지 않음(D-024).
- `box_threshold` 0.1(D-097): ByteTrack이 D_low를 쓸 수 있도록 low_thresh와 같게 둔다. 0.35는 D_high 기준이 됐다.
- 클래스 무관 NMS 0.5(D-116, iter 14): 같은 물체에 두 문구가 켜질 때 박스 하나만 남긴다. 40장면: mIoU 54.8 → 55.9, mAP50 27.5 → 29.0, chair AP50 17.3 → 25.6,
  인스턴스 49.3 → 41.0 /장면(22장면 개선 / 15장면 악화). 대가: desk AP50 → 0, bed AP50 52.5 → 32.5. NMS 단독은 +1.4 mIoU이나 mAP50 이득 없음.
- 임계값 스윕(iter 14b): NMS 0.4는 mIoU −1.6이고 0.6은 chair AP50 −5, 둘 다 desk 미회복이라 0.5 유지.

### 5.5 마스크 정리 (D-036, D-106, D-107, D-116 depth edge)
논문은 정제 규칙을 말하지 않아 처음엔 정제 없음(D-036)이었습니다. Replica office0에서 침식 2 px(50 % 이상 사라지면 취소) + MAD 3σ 깊이 이상치 제거(하한 5 cm)(D-106)와 겹친 픽셀을 작은 마스크에 귀속(D-107)을
넣어 mAP50 9.0 → 28.5(iter 7)가 됐습니다. 깊이 가장자리 제외(D-116)는 단독으로 중립(chair AP50 +6.5, bed −14)이며 NMS와 함께 쓸 때 위 효과에 기여합니다. 임계 0.05 / 0.15는 평균 변화 0.6 이하로 둔감해 0.1 유지.

### 5.6 이웃 기준 (D-032 → D-110)
D-032는 centroid가 이미지 안에 있는 객체만 이웃으로 삼았습니다. 진단(10장면): 중복 새 id 사건의 55 %에서 원본이 이웃 밖(중앙값 5 700점)이었습니다. D-110은 균등 간격 표본 점 최대 256개 중
하나라도 화면 안, 10 m 안이면 이웃으로 봅니다(`neighbors.mode: points`). D-111과 함께 mIoU 36.1 → 43.1, mAP50 14.8 → 18.2, 인스턴스 114 → 85 /장면.

### 5.7 흡수 규칙 (D-111)과 의미 검사 앞의 흡수 계획 (D-120)
- D-111: 새 id의 `O_add` 점이 기존 ACTIVE 이웃의 2 cm 복셀 안에 0.5 이상 들어가면 라벨과 무관하게 흡수하고, 트랙은 흡수한 id로 재지정(`Tracker.remap_id`)하며 쓰지 않은 id는 재사용하지 않습니다.
  진단(10장면, 새 id 1 278개): 62 %가 GT 중복이었고 라벨 무관 겹침 0.5가 54 %를 잡고(같은 라벨만이면 11 %) 진짜 새 객체의 약 9 %를 오흡수. 40장면 흡수 193 /장면.
- D-120(iter 15): D-117 클래스 게이트 때문에 라벨이 바뀐 검출이 새 트랙 id를 열면, 그 객체는 `O_on`인데 `T_trk`에 없어 Alg. 1 의미 검사가 부정하고(L_exist 하락) 그 뒤에야 흡수됩니다. 라벨이 계속 바뀌는 정적 객체가
  5 segmentation 프레임 뒤 제거되고 새 id로 재생성되는 문제(verify_iter_15, I-15-1 P1)였습니다. 해결: 흡수 결정을 Unproject 직후 순수 함수로 계산하고, 의미 검사에서 "흡수하는 객체 = 지지됨"으로 취급하며, ObjMerge는 같은 계획을 적용합니다.
  거부한 대안: label_cost 0으로 복귀(D-117 이득 상실), 소프트 클래스 비용(D-108이 office0에서 0.5가 0보다 나쁨, chair/table 스위치를 못 막음), 검출 라벨을 인스턴스 Eq. (10) 라벨과 비교(지속적 relabel을 못 덮음).
  평균 변화는 i15_lc_cc10 대비 0.4 이하입니다.

### 5.8 클래스 인지 연관 (D-108 → D-117)
D-033은 ByteTrack처럼 클래스 무관 연관이었고 D-108이 `label_cost`를 실험용으로만 두었습니다(tiny 검출기, office0에서 0.5는 mIoU 24.4이나 N_T 109 → 162, mAP 46.4/28.5 → 42.5/25.7).
iter 15 진단: GT chair의 36 %가 인접한 의자 / 다른 클래스 박스 사이의 트랙 ID 스위치로 이웃과 병합됩니다. D-117은 `label_cost` 1.0을 기본으로 합니다(40장면, D-116 기반): mIoU 55.9 → 59.7,
Acc 61.6 → 63.9, mAP50 29.0 → 36.9, chair AP50 25.6 → 36.5, N_active 41.0 → 44.5, 29장면 개선 / 9장면 악화. base 검출기는 tiny보다 덜 relabel해 D-108의 반증이 더는 성립하지 않았습니다.
변화 감지에서는 단독으로 제거 측이 나빠지고(FP strict 0.17 → 0.28) D-118과 함께일 때 회복됩니다.

### 5.9 최대 연결 성분 가지치기 (D-118, D-036 개정)
진단: GT chair의 19 %가 바닥, 벽 점 때문에 상자가 부풀어 있고, 평가 시점 백분위 상자만으로 chair AP50 +4.5. ObjMerge 뒤 이번 프레임에 점을 받은 인스턴스는 `component_voxel` 0.10 m 격자의
최대 26-연결 성분 점만 유지합니다(점, 색, log-odds 함께 삭제). D-117 위 40장면: mAP25 54.7 → 61.1, mAP50 36.9 → 43.3, chair AP50 36.5 → 54.4, sofa AP50 62.9 → 86.4, mIoU 59.7 → 59.4, Acc 63.9 → 63.0
(장면당 11 M 인스턴스 점 중 71 k 삭제). 격자 0.05: mAP50 44.5, chair 57.9이나 Acc 62.1, desk / table 하락이라 0.10 채택. 관측 단위 필터(`unproject.component_voxel`)는 8개 chair 다수 장면에서 mIoU 51.7 → 48.5 / 51.4로 악화(침식 뒤 얇은 부분이 떨어짐)해 꺼 둡니다.
변화 감지: 제거 검출 strict 0.43 → 0.51, REMOVED 8 → 14 /29(옛 위치의 조각을 지움).

### 5.10 점 log-odds와 가지치기 (D-048, D-049, D-050, D-051)
점당 프레임당 한 번 Eq. (8)을 적용합니다. P(o|Q): Observable 0.7, Disappeared 0.3, Unobservable 0.5. 새 점 초기 log-odds = logit(0.7), L_prune = −1, 클램프 없음, τ_ε = 0.1 m. 시야 밖 / 무효 depth 점은
NOT_IN_VIEW로 갱신 없음(D-046). 객체 판정은 시야 안 점의 다수결이며 동률은 Observable > Unobservable > Disappeared(D-047). `O_neg` 객체는 ObjNegate(시야 안 점에 p_dis)만 받습니다. 점이 모두 사라진 객체는 REMOVED로
보존해 id를 재사용하지 않습니다(D-053). D-101: 병합 복셀 중복 제거 0.02 m로 office0 인스턴스 점이 101 M에 이르던 문제(실시간의 24.7배 느림)를 막았습니다.

### 5.11 객체 단위 존재 log-odds (D-113)
논문 초록의 "existence-and-label confidence update"에 공식이 없어 추가했습니다. 점 단위 음의 증거가 시야 안 점에만 가고 점이 전부 사라져야 제거되어, 합성 변화 감지에서 제거된 객체가 옛 점으로 살아남았습니다(완화 FP 0.88, 29개 중 1개 제거, 클램프 효과 미미).
각 인스턴스의 L_exist(초기 logit(p_obs))를 프레임마다 DISAPPEARED 또는 `O_neg`이면 + logit(p_dis), Observable이면 + logit(p_obs), 그 외 무변화로 갱신하고 ±3으로 클램프하며 −1 미만이면 REMOVED입니다.
효과: 제거 FP 0.41 → 0.20, change recall soft 0.43 → 0.66, 제거 도달 1 → 13 /29. 대가: 정적 mIoU 43.1 → 41.1(장면당 약 24개 오제거). `exist_prune: null`이면 규칙이 꺼집니다.
변형 `exist_semantic: false`는 거의 차이가 없고(FP 0.23, 12 /29, mIoU 41.1), `exist_prune −3`은 클램프 때문에 도달할 수 없어 규칙을 끈 것과 같았습니다.

### 5.12 라벨 융합 (D-060, D-061, D-062, D-063)
첫 관측으로 균등 prior에 Eq. (10)을 1회 적용해 초기 분포를 만들고(D-060), 대각 0.8 / 비대각 균등 confusion(D-061), 관측 가능한 점이 하나라도 있는 인스턴스만 갱신(D-062, D-091), max_c P < 0.4면 제거(D-063)합니다.
첫 관측 belief가 정확히 0.8이라 (1/C, 0.8] 범위의 p_min은 사실상 작동하지 않습니다(iter 8 ablation에서 p_min 0.6 / 0.7이 같은 결과).

### 5.13 장면 그래프 (D-070 ~ D-079)
단일 연결 클러스터링 d_clu = 3 m(D-070), On(A,B): |z_min^A − z_max^B| ≤ δ이고 IoU_xy > γ, γ = 0.05, δ = 0.05 m(D-071), Under는 On의 역, Beside는 수직 구간이 겹치고 IoU_xy ≤ γ이며 수평 간격 < 0.5 m(D-073),
센트로이드가 0.1 m 이상 움직이면 시간 노드와 엣지(D-074), 사라진 id는 마지막 위치와 시각을 단서로 남김(D-075, D-078), 로컬 서브그래프 반경 5 m(D-079). 직렬화는 노드, 엣지, 스키마, 단서 순(D-076 ~ D-078).
논문 supplementary에도 "full construction details"가 없어 모두 가장 단순한 결정론적 선택입니다.

### 5.14 VLM 백엔드와 fallback (D-080, D-081, D-082, D-098, D-099, D-100)
모델은 논문대로 `gemini-2.0-flash`, 온도 0이지만 API에서 제공되지 않습니다(2026-09-30 확인). `vlm.fallback_models`가 404일 때만 다음 모델로 넘어가고, 다른 오류(429, 503)는 재시도 없이 올립니다(질의당 호출 1회).
무료 등급이 모델당 하루 20회라 flash-lite 계열을 앞에 두었습니다(D-100). 키는 환경변수, 그다음 루트 `.env`(`KEY=VALUE`)에서 읽고 값은 로그에 남기지 않습니다(D-098).
`<answer>` 파서는 마지막 태그 안의 모든 음이 아닌 정수를 id로 봅니다(D-082). 없거나 사라진 id는 경고 후 건너뜁니다(D-083).

---

## 6. 시도했으나 채택하지 않은 것

코드는 남아 있고 기본 꺼짐입니다. 한 줄에 그것을 기각한 수치를 붙였습니다.

| 시도 | 기각한 수치 | 비고 |
|---|---|---|
| segmentation 속도 3 / 10 Hz (iter 15, D-119) | 8개 chair 다수 장면 mIoU 51.7 → 24.5 / 19.5 | Eq. (8)과 D-113 갱신이 segmentation 이벤트마다 적용돼 지도가 붕괴. 논문값 1 Hz 유지 |
| ByteTrack 임계값 / buffer / p_min / min_points (iter 8, 11종) | 8장면에서 bt 0.4 / 0.5가 +3.4 mIoU, mAP50 11.3 → 16.1, 인스턴스 125 → 52였으나 **40장면에서 −1.0 mIoU, −1.3 mAP50** (17장면만 개선) | 8장면은 대표성이 없어 이후 기본값은 40장면으로만 결정. 임계값 상승은 점수가 낮은 큰 구조물(wall, cabinet, desk)의 트랙을 막을 뿐 |
| SAM2.1-large (D-114) | 단독 41.6, base와 함께 −1.9 | 검출기가 병목 |
| 별칭 변형 (D-115) | "office desk": desk 11 → 48이나 table 39 → 25, bed 60 → 50 | "curtains"도 shower curtain 손실로 제외 |
| I-9-4 완화: 객체 판정 창 탐색 / 커버리지 게이트 / w3gate (iter 10, 14c) | iter-9 기반: 창 3은 거짓 제거 24 → 15 /장면이나 mIoU 40.8, 참 제거 13 → 6 /29. 게이트는 mIoU 42.1 ~ 42.6 회복하나 제거 FP 0.30 ~ 0.33. D-116 기반 w3gate: mIoU 55.9 → 57.1, Acc 61.6 → 63.4이나 mAP50 −0.9, chair AP50 −4.2, 변화 감지 제거 change soft 0.49 → 0.44, REMOVED 8 → 5 /29, 중앙값 25.5 → 603 프레임 | "정적 정확도 ↔ 변화 감지" 선 위를 움직일 뿐. 결정은 사람에게 남음(I-9-4), 오케스트레이터 권고는 미채택 |
| 발자국 박스 재활성화 / 지도 매칭 (iter 12) | mIoU 51.5 ~ 52.8, chair AP50 17 → 11 ~ 13 | GT 대조: 붙인 검출의 25 %가 다른 객체 |
| 3D 접촉 흡수 5 cm (iter 13) | mIoU 52.9, chair AP50 11.6 (같은 라벨만: 54.7 / 13.0) | 나란한 의자가 5 cm 안에서 맞닿아 병합 |
| D-111 끄기 (D-119) | mIoU 56.7 / mAP50 32.3 / chair 35.3이나 인스턴스 41 → 67.8, 변화 감지 제거 change soft 0.49 → 0.43, FP 0.17 → 0.24, REMOVED 8 → 5 | 같은 라벨만 흡수(`absorb_same_label`)는 8장면에서 mIoU 51.7 → 52.4, chair AP50 6.1 → 8.1로 거의 무변화 (chair 병합은 같은 라벨) |
| 역방향 겹침 `absorb_overlap_rev` 0.1 / 0.2 (D-119) | 흡수 1 395 → 365 / 106건으로 사실상 "끄기"와 같음 | 8장면 mAP50 15.1 / 16.9 |
| 3D 중심 게이트 0.5 / 1.0 m (D-117, D-119) | 8장면 mIoU +2.1 / +2.8이나 chair AP50 6.1 → 5.8 / 4.1, 트랙 1.4배 | 거부된 매칭이 새 트랙이 됨. 자기 운동과 깊이 중앙값 centroid가 참 매칭을 거부 |
| 관측 단위 연결 성분 필터 0.05 / 0.10 (D-118) | 8장면 mIoU 51.7 → 48.5 / 51.4 | 얇은 부분이 침식 뒤 떨어져 참 점 손실 |
| AP 순위 점수 대안 (D-119) | 점 수 28.4, 존재 확률 27.2, 관측 기간 27.4, 라벨 × 존재 28.6 (기준 29.0); finalist에서 라벨 × 존재 41.5 vs 43.3 | 평가 시점에만 시험(`--score`). 지도는 별도 점수를 갖지 않음 |
| NMS 0.4 / 0.6 (iter 14b) | mIoU 54.3 / 55.8, mAP50 29.1 / 28.9, chair AP50 26.6 / 20.6, desk AP50 6.7 / 3.3 | 0.5가 최적, desk 미회복 |
| 깊이 가장자리 0.05 / 0.15 (iter 14b) | mIoU 56.2 / 55.7, mAP50 29.6 / 28.1 | 평균 변화 ≤ 0.6, 둔감 |
| NMS 0.5 단독 (iter 13) | mIoU 56.2(+1.4), mAP50 보합, desk IoU 11.5 → 5.1 | 이후 깊이 가장자리와 함께 D-116으로 채택 |
| 깊이 가장자리 필터 단독 (iter 13) | 중립: chair AP50 +6.5, bed −14 | 큰 물체 경계 점 손실 |

---

## 7. 평가 프로토콜

### 7.1 ScanNet GT 처리 (D-109)
- 장면: validation 40개(scene id당 `_00` 스캔 하나: `scene0011_00`, `scene0030_00`, `scene0046_00`, `scene0086_00`, `scene0222_00`, `scene0378_00`, `scene0389_00`, `scene0435_00`, `scene0050_00`, `scene0064_00`, `scene0084_00`, `scene0100_00`, `scene0131_00`, `scene0164_00`, `scene0193_00`, `scene0207_00`, `scene0231_00`, `scene0256_00`, `scene0300_00`, `scene0329_00`, `scene0355_00`, `scene0406_00`, `scene0423_00`, `scene0474_00`, `scene0500_00`, `scene0549_00`, `scene0568_00`, `scene0580_00`, `scene0599_00`, `scene0616_00`, `scene0629_00`, `scene0645_00`, `scene0653_00`, `scene0664_00`, `scene0678_00`, `scene0685_00`, `scene0695_00`, `scene0700_00`, `scene0025_00`, `scene0088_00`), 앞 8개는 열린 어휘 3D 매핑 논문들이 공유하는 장면. 입력은 30 Hz 촬영의 3프레임당 1프레임(10 Hz).
- GT: NYU40 정점 라벨(`_vh_clean_2.labels.ply`), 인스턴스는 aggregation + segs. 19개 클래스, 배경 = wall, floor.
- 라벨 전사: 예측 점 → GT 정점 최근접(반경 r = 0.05 m, 0.10 m도 계산). **GT 정점 전체**가 분모.
- 의미 분할: mIoU, f-mIoU, Acc를 배경 제외 / 포함 두 가지로 계산. 클래스별 평균은 그 클래스가 있는 장면에 대해서만.
- 인스턴스 AP: 예측 점의 3D AABB와 GT 인스턴스의 3D AABB IoU 0.25 / 0.50.
- 40장면 평균의 장면 간 표준편차가 mIoU 10.8이라 8개 공통 장면만으로는 불안정하며 기본값 결정은 40장면으로만 합니다.

### 7.2 프로토콜 변형 (재평가만, 파이프라인 변경 없음)
최종 결과(iter 15)에서 plain 59.4와 비교: 병합(desk>table, shower curtain>curtain) **62.9**, 반경 5 cm 다수결 전사 **60.2**, 둘 다 **63.7**; 병합 시 mAP50 44.6.
iter 14는 59.3 / 58.8 / 62.4(plain 55.9), iter 11은 58.5 / 59.5 / 64.2(plain 54.8)였습니다. 이전 측정(iter 11 기준, 재측정 안 함)으로 예측 상자를 1 ~ 99 백분위로 하면 mAP50 27.5 → 30.1,
5 cm 복셀 IoU는 mAP50 14.2였습니다. 변형은 plain 수치를 대체하지 않고 옆에 보고합니다.

### 7.3 합성 변화 감지 (D-112)
논문 Table IV의 실로봇 데이터가 비공개라 ScanNet 9장면(공통 8 + scene0193_00)의 GT 인스턴스 29개를 제거 / 추가한 58시퀀스(`remove`: 대상을 뒤 절반에서 제거, `add`: 앞 절반에서 제거)로 평가합니다.
대상 클래스는 chair, toilet, sink, refridgerator, picture, sofa, bed, desk, bathtub(정점 300개 이상, 양쪽 절반의 15개 이상 프레임에서 보임; 실제로는 chair 13, desk 6, bed 5, sink 2, refridgerator 2, picture 1).
객체 픽셀은 GT 메시에서 그 객체의 정점을 뺀 뒤 레이캐스팅한 배경(깊이 = camera z, 색 = barycentric 정점 색)으로 치환합니다(메시 구멍은 최근접 픽셀로 채움).
지표는 대상 위치가 보이는 segmentation 프레임 기준입니다. 검출 recall(존재 구간), FP율(부재 구간), change recall soft(검출됐다면 1 − FP율)와 strict(FP 0일 때만 1).
논문 정의의 strict 매칭(3D IoU > 0.1, centroid < 0.3 m, 라벨 일치)과 완화 매칭(IoU > 0.1 + 라벨; 한쪽에서만 본 큰 객체의 점 centroid가 GT 메시 centroid에서 0.3 m 넘게 벗어나므로)을 함께 보고합니다.
로그-오즈 클램프(`L_clamp` 3.0 / 2.0)는 실험 전용입니다.

### 7.4 실시간 벤치마크 프로토콜 (논문 Sec. V-H)
Replica office0 2000프레임(10 Hz = 200 s), VLM 단계 제외. 모듈 호출당 벽시계 지연(CUDA 동기화)을 첫 호출(warm-up)을 빼고 통계화합니다. 판정은 평균, p95, 최대가 모두 모듈 주기 이하여야 충족.
모듈은 한 스레드에서 순차 실행되며 pose는 데이터셋에서 읽으므로 pose 모듈 지연은 Eq. (3) 합성뿐입니다. GPU는 RTX PRO 6000 Blackwell(논문 노트북 RTX 4090보다 강해 낙관적).

---

## 8. 결과 이력

### 8.1 ScanNet v2 40장면 (논문 Table II/III 지표)

| 40장면 평균 | mIoU w/o bg | f-mIoU | Acc | mIoU w bg | f-mIoU | Acc | mAP25 | mAP50 | 인스턴스/장면 (GT 15) |
|---|---|---|---|---|---|---|---|---|---|
| iter 7 (D-101 ~ 108) | 36.1 | 38.1 | 43.3 | 32.5 | 39.4 | 49.3 | 32.7 | 14.8 | 113.8 |
| iter 8 (+ D-110, D-111) | 43.1 | 44.7 | 49.0 | 39.5 | 46.3 | 54.5 | 32.7 | 18.2 | 85.2 |
| iter 9 (+ D-113) | 41.1 | 42.7 | 46.5 | 38.4 | 45.2 | 52.9 | 31.2 | 18.1 | 68.7 |
| iter 11 (+ D-114, D-115) | 54.8 | 56.8 | 61.2 | 46.8 | 50.0 | 58.8 | 47.3 | 27.5 | 49.3 |
| iter 14 (+ D-116) | 55.9 | 57.1 | 61.6 | 47.1 | 50.2 | 58.3 | 48.2 | 29.0 | 41.0 |
| **iter 15 (+ D-117, D-118, D-120, 현재)** | **59.4** | **59.9** | **62.8** | **55.6** | **59.0** | **64.0** | **61.3** | **42.9** | 46.3 |
| 논문 SuperMap (Table II) | 27.4 | 43.5 | 55.5 | 22.6 | 29.1 | 33.0 | – | – | – |

논문은 장면 목록, 프롬프트, GT 대응 규칙을 공개하지 않아 "같은 데이터셋, 같은 지표"의 참고 비교입니다. iter 8은 40장면 중 33장면에서 mIoU가 개선(평균 +7.0)됐고
iter 11은 iter 9 대비 +13.7 mIoU(검출기 base +7.7, 별칭 +6.0)입니다. 논문 Table III의 chair AP50 63.8, window 42.2에 대해 우리의 현재 값은 chair 54.6, window 27.7입니다.

### 8.2 클래스별 (iter 14 → iter 15, 그 클래스가 있는 장면 평균, wall / floor는 배경 포함 프로토콜)

| class | scenes | IoU iter 14 | IoU iter 15 | AP50 iter 14 | AP50 iter 15 |
|---|---|---|---|---|---|
| toilet | 9 | 82.2 | 88.0 | 66.7 | 77.8 |
| floor | 40 | 57.5 | 75.4 | - | - |
| picture | 16 | 66.2 | 75.3 | 41.3 | 67.5 |
| curtain | 9 | 72.1 | 74.8 | 37.3 | 63.9 |
| bookshelf | 9 | 68.9 | 73.3 | 48.0 | 58.5 |
| sofa | 11 | 69.2 | 71.9 | 57.6 | 86.4 |
| bathtub | 7 | 61.5 | 69.9 | 28.6 | 33.3 |
| chair | 32 | 61.3 | 68.8 | 25.6 | 54.6 |
| bed | 10 | 52.4 | 62.7 | 32.5 | 50.0 |
| sink | 11 | 61.9 | 61.4 | 63.6 | 58.3 |
| door | 34 | 59.1 | 60.0 | 20.7 | 29.7 |
| window | 23 | 55.6 | 59.0 | 19.8 | 27.7 |
| table | 22 | 44.3 | 51.4 | 25.4 | 40.3 |
| wall | 39 | 42.4 | 48.0 | - | - |
| refridgerator | 8 | 45.5 | 47.0 | 21.9 | 26.0 |
| counter | 6 | 40.1 | 42.0 | 16.7 | 16.7 |
| shower curtain | 7 | 30.7 | 31.9 | 28.6 | 28.6 |
| cabinet | 16 | 32.0 | 29.2 | 11.7 | 25.0 |
| desk | 15 | 2.7 | 1.9 | 0.0 | 0.0 |

### 8.3 iter 15 chair AP50 레버 연구
시작점은 D-116 헤드라인(chair AP50 25.6). 진단: 1 Hz는 원인이 아니고, GT chair의 7 %만 놓치며 36 %가 이웃과 병합(D-111 흡수 + 트랙 ID 스위치), 19 %가 부푼 박스, 라벨 확률 점수가 1.0으로 포화.
사람이 정한 순서: (1) 1 Hz 유지 (2) 흡수 규칙 (3) ID 스위치 억제 (4) 외곽 점 제거 (5) 점수. 모든 스위치는 기본 꺼짐으로 추가(commit 63cd938) 뒤 선별했습니다.

8개 chair 다수 장면(0011 0030 0088 0131 0231 0355 0500 0599) 선별. head8 = D-116 헤드라인을 이 장면으로 제한(chair AP50 6 vs 40장면 25.6):

| run | override | mIoU | Acc | mAP25 | mAP50 | N_act | chair IoU / AP25 / AP50 | table IoU / AP25 / AP50 |
|---|---|---|---|---|---|---|---|---|
| head8 | (D-116 defaults) | 51.7 | 63.9 | 30.8 | 11.5 | 64.0 | 67.2 / 28.6 / 6.1 | 34.6 / 42.3 / 5.0 |
| chairdiag_noabsorb | absorb_overlap null | 55.5 | 66.5 | 41.7 | 17.3 | 94.6 | 71.9 / 42.5 / 8.4 | 40.9 / 31.5 / 18.0 |
| i15_abs_label | absorb_same_label true | 52.4 | 64.5 | 28.0 | 13.5 | 70.1 | 70.7 / 28.1 / 8.1 | 30.8 / 23.1 / 11.7 |
| i15_abs_rev02 | absorb_overlap_rev 0.2 | 53.2 | 65.9 | 40.7 | 16.9 | 91.9 | 70.6 / 41.1 / 9.1 | 35.3 / 28.7 / 16.2 |
| i15_abs_rev01 | absorb_overlap_rev 0.1 | 53.1 | 64.9 | 36.5 | 15.1 | 87.0 | 68.7 / 32.5 / 7.9 | 33.3 / 33.6 / 13.8 |
| i15_labelcost | label_cost 1.0 | 58.6 | 68.1 | 39.0 | 25.5 | 70.4 | 75.6 / 32.0 / 8.8 | 38.3 / 35.3 / 31.4 |
| i15_lc_noabs | label_cost 1.0 + absorb off | 62.2 | 71.0 | 43.8 | 25.4 | 115.8 | 77.4 / 48.2 / 10.3 | 47.9 / 39.6 / 31.0 |
| i15_gate05 | gate_3d 0.5 | 53.8 | 63.9 | 32.7 | 17.4 | 88.1 | 67.5 / 26.2 / 5.8 | 35.0 / 23.1 / 14.3 |
| i15_gate10 | gate_3d 1.0 | 54.5 | 66.7 | 34.5 | 16.6 | 76.0 | 70.9 / 28.0 / 4.1 | 35.7 / 41.1 / 12.3 |
| i15_cc_unp05 | unproject component_voxel 0.05 | 48.5 | 58.8 | 31.0 | 17.3 | 65.1 | 64.6 / 31.0 / 10.0 | 28.2 / 33.6 / 21.4 |
| i15_cc_unp10 | unproject component_voxel 0.10 | 51.4 | 61.4 | 33.0 | 14.4 | 66.1 | 66.2 / 33.0 / 9.1 | 32.6 / 34.9 / 5.0 |
| i15_cc_merge05 | merge component_voxel 0.05 | 51.3 | 59.1 | 44.9 | 28.9 | 73.0 | 65.8 / 51.5 / 33.8 | 28.3 / 36.4 / 15.0 |
| i15_cc_merge10 | merge component_voxel 0.10 | 54.8 | 62.2 | 46.8 | 25.6 | 67.1 | 66.3 / 49.6 / 21.3 | 30.1 / 38.8 / 20.2 |
| i15_lc_cc05 | label_cost 1.0 + merge cc 0.05 | 56.9 | 62.9 | 45.7 | 31.5 | 79.8 | 69.5 / 63.6 / 39.9 | 36.8 / 30.9 / 22.0 |
| i15_lc_cc10 | label_cost 1.0 + merge cc 0.10 | 58.3 | 64.9 | 48.2 | 29.1 | 76.1 | 72.1 / 59.3 / 31.6 | 39.2 / 38.5 / 26.4 |
| i15_lc_cc05_noabs | + absorb off | 56.7 | 64.3 | 44.1 | 28.5 | 118.9 | 72.6 / 65.6 / 41.3 | 35.4 / 26.9 / 7.6 |
| i15_lc_cc10_noabs | + absorb off | 57.7 | 65.7 | 47.2 | 26.8 | 115.9 | 73.2 / 62.7 / 35.5 | 38.6 / 30.5 / 14.7 |

8장면 계수(헤드라인: 흡수 1 395건, N_T 512): `absorb_same_label` 1 153건, `absorb_overlap_rev` 0.2 / 0.1은 106 / 365건, gate_3d 0.5 / 1.0은 가능한 쌍 8 373 / 2 777개를 거부하고
그 검출이 새 트랙이 되어(N_T 705 / 608) chair AP50이 하락, 병합 component_voxel 0.05 / 0.10은 약 6.6 M 인스턴스 점 중 2.16 M / 1.61 M을 삭제.

40장면:

| run | mIoU | Acc | mAP25 | mAP50 | N_act | chair IoU / AP25 / AP50 | table AP50 | bed AP50 | sofa AP50 | desk AP50 | 개선 / 악화 장면 (mIoU, 0.5) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| headline (D-116) | 55.9 | 61.6 | 48.2 | 29.0 | 41.0 | 61.3 / 45.4 / 25.6 | 25.4 | 32.5 | 57.6 | 0.0 | - |
| i15_noabsorb | 56.7 | 62.5 | 53.2 | 32.3 | 67.8 | 64.9 / 59.6 / 35.3 | 32.9 | 39.6 | 75.3 | 2.2 | 21 / 14 |
| i15_labelcost | 59.7 | 63.9 | 54.7 | 36.9 | 44.5 | 70.3 / 62.4 / 36.5 | 42.4 | 56.7 | 62.9 | 3.3 | 29 / 9 |
| i15_lc_cc05 | 59.1 | 62.1 | 59.8 | 44.5 | 47.5 | 65.2 / 72.6 / 57.9 | 45.3 | 45.0 | 86.4 | 7.8 | 26 / 12 |
| i15_lc_cc10 | 59.4 | 63.0 | 61.1 | 43.3 | 46.3 | 65.8 / 72.7 / 54.4 | 40.3 | 50.0 | 86.4 | 0.0 | 27 / 10 |

해석: (2) 흡수 변형은 chair를 가르지 못하고 "끄기"만 도움이 되며 그 대가로 인스턴스가 1.5배(8장면)에서 1.65배(40장면)입니다. (3) 클래스 인지 1차 연관이 가장 강한 단일 레버이고 3D 게이트는 아닙니다.
(4) 관측 단위 필터는 참 점을 지우고 ObjMerge 뒤 인스턴스 단위 필터가 동작합니다. 0.05 격자는 chair AP50만 보면 57.9 vs 54.4이나 Acc −0.9, table / desk / toilet AP50 하락. 최종 기본값은 D-117 + D-118 + D-120이며,
D-120 적용 후 **최종 헤드라인**은 `i15_d120`(위 8.1절 iter 15 행): mIoU 59.4(55.9), Acc 62.8(61.6), mAP25 61.3(48.2), mAP50 42.9(29.0), chair AP50 54.6(25.6), chair IoU 68.8(61.3), N_active 46.3(41.0).
iter 14 결과는 `_iter14` 접미사로 보관.

### 8.4 변화 감지 이력 (58시퀀스, D-112)

| 설정 | remove: 검출 strict | FP strict | change soft (strict 매칭) | change soft (완화 매칭) | 대상 REMOVED 도달 | add: 검출 strict | add: change soft |
|---|---|---|---|---|---|---|---|
| iter 8 (존재 규칙 없음) | 0.47 | 0.41 | 0.43 | 0.11 | 1 / 29 | 0.38 | 0.55 |
| iter 9 (+ D-113) | 0.47 | 0.20 | 0.66 | 0.31 | 13 / 29 | 0.35 | 0.62 |
| iter 11 (+ D-114, D-115) | 0.52 | 0.22 | 0.62 | 0.47 | 15 / 29 | 0.43 | 0.63 |
| iter 14 (+ D-116) | 0.43 | 0.17 | 0.49 | 0.35 | 8 / 29 | 0.41 | 0.61 |
| iter 14c (D-116 + w3gate, 미채택) | 0.44 | 0.22 | 0.44 | 0.22 | 5 / 29 | 0.38 | 0.59 |
| iter 15: 흡수 끄기 (미채택) | 0.47 | 0.24 | 0.43 | 0.23 | 5 / 29 (중앙값 678프레임) | 0.50 | 0.63 |
| iter 15: label_cost 1.0만 | 0.49 | 0.28 | 0.43 | 0.28 | 11 / 29 (300) | 0.47 | 0.56 |
| iter 15: label_cost 1.0 + cc 0.05 (미채택) | 0.46 | 0.22 | 0.41 | 0.31 | 15 / 29 (411) | 0.40 | 0.62 |
| **iter 15 (D-117 + D-118 + D-120, 현재)** | **0.51** | **0.23** | **0.46** | **0.28** | **14 / 29 (354)** | **0.51** | **0.64** |
| 논문 Table IV (실객체 6개) | 0.26 ~ 1.00 | | 0.62 ~ 1.00 | | | | |

(`i15_d120/`은 `i15_lc_cc10/`과 동일.) 클래스별 remove(검출 strict / change soft): chair(13) d116 0.55 / 0.62 → 현재 0.68 / 0.71, bed(5) 0.47 / 0.70 → 0.40 / 0.69,
desk(6) 0.06 / 0.17 → 0.14 / 0.00(I-14-2: NMS가 억제한 desk 박스). iter 11 → iter 14의 desk는 0.42 / 0.54 → 0.06 / 0.17입니다. 흡수 끄기와 클래스 게이트 단독은 옛 위치에 조각이 남아 제거 측이 나쁘고,
D-118 가지치기가 그 조각을 지워 d116 수준으로 돌아오면서 14 /29가 REMOVED에 도달합니다.
D-113 진단의 결론: 점 log-odds 크기(클램프)가 아니라 "음의 증거는 시야 안 점에만, 제거는 모든 점이 사라져야"라는 구조가 원인이었습니다.

### 8.5 실시간 벤치마크 이력 (Replica office0 2000프레임, 지연 평균 / p95 / 최대 ms)

| 모듈 | 주기 | iter 14 | iter 15 (현재) | 충족 |
|---|---|---|---|---|
| pose | 100 ms | 0.1 / 0.1 / 0.3 | 0.1 / 0.1 / 0.7 | ✓ |
| segmentation | 1000 ms | 187.6 / 248.5 / 386.1 | 189.5 / 259.8 / 330.0 | ✓ |
| mapping | 333 ms | 64.5 / 76.3 / 162.6 | 64.6 / 77.9 / 128.2 | ✓ |
| scene graph | 200 ms | 0.6 / 0.7 / 23.9 | 0.6 / 0.8 / 1.1 | ✓ |

루프 벽시계: iter 11 97.3 s, iter 14 100.8 s, iter 15 100.2 s / 200 s 데이터(0.50×). iter 11 segmentation은 171.0 / 207.6 / 228.5 ms. iter 15 세부(평균 ms, warm-up 제외):
`seg.groundingdino` 94.2, `seg.unproject` 44.3, `seg.sam2` 25.9, `seg.obj_merge` 7.1(D-118 가지치기 포함), `dense_map.integrate` 62.3, `compare_depth` 2.7, `get_neighbor_objects` 0.4.
같은 날 부하가 높은 머신(load average 40 ~ 48)에서 잰 두 번(루프 164 / 206 s)은 무효로 폐기하고 유휴 머신에서 다시 쟀습니다. GPU가 논문의 노트북 4090보다 강해 낙관적인 수치입니다.

### 8.6 Replica office0 (개발용)
iter 7: mIoU 23.6 / 20.1(배경 제외 / 포함), mAP25 / mAP50 46.4 / 28.5. 마스크 정리(D-106 + D-107) 전 mAP50 9.0. iter 8(D-110, D-111)에서 office0 N_T 73 → 31(흡수 132건 / 200 segmentation 프레임).
D-108 시험(tiny, D-106 + D-107 기반): label_cost 0.5는 mIoU 24.4이나 N_T 109 → 162, mAP 46.4 / 28.5 → 42.5 / 25.7. Replica는 논문 벤치마크가 아니라 참고용입니다.

---

## 9. 한계와 열린 이슈

### 9.1 논문과의 격차
- **window / refrigerator AP**: window AP50 27.7(논문 42.2), refrigerator AP50 26.0. 얇은 평면 물체와 다른 클래스 박스가 겹치는 경우가 남아 있습니다. chair AP50은 54.6으로 논문 63.8에 근접했습니다.
- **desk = 0**: D-116의 클래스 무관 NMS가 같은 물체의 "table" 박스 뒤에서 "desk" 박스를 지워 desk가 트랙을 얻지 못합니다(desk IoU 1.9, AP50 0.0, 변화 감지 desk remove change soft 0.17 → 0.00). 임계값(0.4 / 0.6)으로 풀리지 않아 억제된 박스의 라벨 증거를 남는 박스에 합치는 처리가 남은 경로입니다(I-14-2).
- **좁은 재방문에서의 제거 recall**: 논문은 360° LiDAR와 의도적 재방문이지만 ScanNet RGB-D는 옛 위치를 조각으로만 봅니다. 제거 change soft 0.46, 29개 중 14개만 REMOVED에 도달하며 중앙값 354프레임이 걸립니다.
- **비교 불가 영역**: 실로봇 변화 감지(Table IV, V)와 VLM 내비게이션 성공률(Sec. V-F)은 데이터가 비공개이고 Gemini 무료 등급 제한이 있어 수치가 없습니다. 논문의 gemini-2.0-flash는 서비스되지 않아 flash-lite 계열로 대체했습니다.
- **데이터셋 모호성**: desk / table, curtain / shower curtain은 ScanNet GT 경계 자체가 흐려 어느 쪽을 택해도 한쪽이 손해입니다(병합 평가 +3.5 mIoU).

### 9.2 열린 이슈 (모두 P2)
- **I-9-4 w3gate 결정**: D-113이 정적 장면에서 작은 조각의 DISAPPEARED 판정으로 장면당 17 ~ 24개를 오제거합니다(mIoU −2.0). 완화 변형(window / gate / w3gate)은 정적 정확도와 변화 감지 사이를 이동할 뿐입니다. D-116 기반 w3gate는 +1.2 mIoU이나 제거 감지를 늦춥니다(중앙값 25.5 → 603 프레임). 기본값은 유지, 채택은 사람의 결정입니다.
- **I-15-2**: D-118이 인스턴스 전체의 최대 연결 성분만 유지하므로, 기존 점과 떨어진 더 큰 관측(예: SAM2가 큰 면으로 번진 경우)이 기존 점을 log-odds와 무관하게 지웁니다. 동률 규칙도 없습니다. 후보 개선: 삭제를 이번 프레임 점으로 제한하거나 병합 전 다수 점을 포함한 성분만 유지.
- **I-15-4**: (a) `absorb_targets` docstring 문구, (b) D-117로 트랙 라벨이 마지막 매칭 검출 라벨이라 계속 relabel되는 객체는 프레임마다 새 트랙 id를 열고 버려 트랙 id를 소모합니다(맵 id, 의미 검사, Eq. 10은 정상).
- **I-14-2**: desk 손실(위 9.1).
- 그 외 충실도 메모: I-6-1(Eq. 5 X_O에서 Disappeared 점 제외, 중앙값 0.24 px 이동), I-7-1(D-106 / D-107이 최소 선택이 아님), I-8-4(float32 pose 3프레임 제외), I-8-7(D-111 세부: O_add 순서 의존, 동률 시 최소 id),
  I-9-1(존재 log-odds 기본값이 map.py 직접 생성 경로에서 0.0), I-9-2(Supp. VII-A2 ablation은 `exist_prune: null`로 재현), I-11-1(`tolerant_states`가 구간 hull 기준, 꺼짐), I-11-3(남은 조각), I-12-1 / I-12-2(꺼진 지도 매칭 경로),
  I-13-2 / I-13-3(꺼진 깊이 가장자리와 접촉 흡수의 충실도 메모).
- **인스턴스 과분할**: 장면당 46.3개(GT 15)이며 3D로 겹치지 않는 같은 물체의 다른 부분과 같은 프레임의 다른 라벨 박스가 남은 원인입니다. 근접 병합(2D, 3D)은 이웃 가구를 합쳐 손해였습니다.

---

## 10. 개발 하네스

구현은 하네스 엔지니어링 방식으로 만들었습니다. 논문에서 62개 요구사항(REQ-001 ~ 062)을 뽑아 명세로 삼고, 역할이 나뉜 LLM 에이전트
(논문 분석, 데이터 준비, 구현, 검증, 보완)가 build → verify → refine 반복을 돌렸습니다. 검증 에이전트는 요구사항마다 논문 기반 테스트를
쓰고 PASS / PARTIAL / N/A 판정을 기록했으며(최종 61 PASS / 1 N/A), 완료는 사람이나 에이전트의 판단이 아니라 자동 게이트(모든 요구사항의
코드 주석·테스트 존재, 스텁 금지, 전체 테스트와 스모크 실행 통과, 최신 검증 보고서가 현재 코드를 대상으로 했는지, 미해결 P0/P1 이슈 없음)가
판정했습니다. 논문이 비워 둔 세부에 대한 선택은 번호가 붙은 결정 기록(D-xxx)으로, 발견된 문제는 이슈 기록(I-xx-x)으로 남겼고 이 문서의
번호들은 그 기록을 가리킵니다. 명세, 결정·이슈 기록, 검증 보고서, 실험 스크립트와 출력은 로컬에만 보관하며 저장소에는 올리지 않았습니다.
