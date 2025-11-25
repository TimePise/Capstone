# 실시간 낙상 감지 관제 시스템 (Django)

환자 낙상을 실시간으로 감지하고, 병동 담당자가 웹 대시보드에서 알림·이력을 모니터링할 수 있도록 만든 Django 백엔드입니다. MediaPipe Pose로 수집한 3차원 관절 좌표를 정규화한 뒤, GRU 기반 딥러닝 모델(`FallTemporalHybridNet`)로 낙상 여부와 충격 부위를 추정합니다. 감지된 이벤트는 Django Channels를 통해 대시보드에 스트리밍되며, 병동별 계정/기록 관리를 함께 제공합니다.

## 핵심 기능
- **실시간 낙상 감지**: OpenCV로 카메라 스트림을 가져와 MediaPipe Pose → Feature 인코딩 → PyTorch 추론 파이프라인을 상시 실행합니다.
- **부위별 위험 단계 분류**: 머리/골반/손목 등 충격 부위에 따라 고·중·저위험을 구분하고 다른 알림 복수 채널(SSE, 알림 목록)에 반영합니다.
- **낙상 알림 스트리밍**: 최신 알림, 미확인 수, 위험도 분포를 `/member/fall/alert/list/`와 SSE(`/fall/sse/fall_alert/`)로 실시간 노출합니다.
- **낙상 이력 관리**: 환자 정보, 발생 일시, 단계 등을 CRUD/필터/CSV 내보내기로 관리합니다.
- **병동 계정/사용자 로그**: 인증번호 기반 회원가입, 로그인/로그아웃 로그 적재, 마이페이지, 계정 삭제를 지원합니다.
- **프라이버시 모드**: 스트리밍 프레임을 즉시 마스킹하면서 관절 스켈레톤만 보여주어 영상 노출을 최소화합니다.


## 기술 스택
| 레이어 | 사용 기술 |
| --- | --- |
| Web & API | Django 4, Django Channels, StreamingHttpResponse, SSE |
| 실시간 처리 | OpenCV, MediaPipe Pose, PyTorch, NumPy, pygame (알람 사운드) |
| 데이터베이스 | SQLite (기본), Django ORM |
| 인프라 | Dockerfile (python:3.10-slim 기반), requirements.txt |

## 시스템 흐름
1. Django 서버가 기동되면 `fall.apps.FallConfig.ready()`에서 카메라 쓰레드를 1회만 시작합니다.
2. OpenCV가 프레임을 읽고 MediaPipe Pose가 선택 관절(코, 손목, 엉덩 등) 좌표와 가시성 정보를 생성합니다.
3. 좌표/속도 피처를 통계값(`feature_stats.json`)으로 정규화해 `FallTemporalHybridNet`에 전달합니다.
4. 모델 출력(낙상 여부, 충격 부위)에 임계치·과반 체크와 엉덩이 높이 검증을 적용해 오검출을 줄입니다.
5. 낙상으로 확정되면 `FallAlert` 레코드를 만들고, mp3 경보와 SSE/페이지에 즉시 전파합니다.
6. 보호자/간호사는 웹의 낙상 감지, 알림 목록, 이력 페이지에서 상황을 확인하고 필요 시 기록을 업데이트합니다.

## 디렉터리 개요
```
.
|-- config/          Django 프로젝트 설정, ASGI/WSGI, URL
|-- fall/            낙상 감지 뷰, 모델(FallAlert), PyTorch 네트워크
|-- member/          계정, 낙상 기록, 사용자 로그 도메인 로직
|-- templates/       member용 화면(대시보드, 로그인 등)
|-- static/          정적 자원
|-- fall_temporal_hybrid_best.pth / feature_stats.json
|-- pose_data_combined.csv         (학습/분석용 포즈 데이터)
|-- Dockerfile, requirements.txt
```

## 설치 및 실행
### 1) 로컬 환경
```bash
python3 -m venv venv
source venv/bin/activate  # Windows는 venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

python manage.py migrate
python manage.py createsuperuser  # 관리자 계정이 필요하면 실행
python manage.py runserver
```
- 기본 카메라 인덱스는 `config/settings.py`의 `CAMERA_INDEX`이며 현재 `0`으로 설정되어 있습니다. 외부 카메라를 쓸 경우 `fall/views.py`의 `cv2.VideoCapture(0)` 값을 변경하세요.
- 서버가 뜨면 `http://127.0.0.1:8000/member/login/`으로 접속해 로그인 후 낙상 감지 화면(`/member/fall_prevention/`)으로 이동합니다.
- Django 개발 서버는 MediaPipe와 PyTorch를 CPU 모드로 실행합니다. 성능 이슈가 있으면 Raspberry Pi/Jetson 등 엣지 디바이스에 맞게 추론 워커를 분리할 수 있습니다.

### 2) Docker
```bash
docker build -t fall-monitor .
docker run --rm -it --net=host --device=/dev/video0 fall-monitor
```
- `--device=/dev/video0` 옵션으로 호스트 카메라를 컨테이너에 전달해야 합니다.
- slim 이미지는 libgl1, libglib2.0-0만 포함하므로 GPU 가속이 필요한 경우 Dockerfile을 확장하세요.

## 주요 페이지·엔드포인트
| 경로 | 설명 |
| --- | --- |
| `/member/login/`, `/member/reg/` | 인증번호 기반 회원 가입/로그인 |
| `/member/fall_prevention/` | 실시간 카메라 스트림, 프라이버시 모드, 감지 결과 위젯 |
| `/fall/pose_feed/` | MJPEG 스트림 엔드포인트 (대시보드에서 `<img>`로 구독) |
| `/fall/toggle_privacy/` | 프라이버시 모드 on/off API |
| `/fall/fall_status/` | 현재 추론 라벨/낙상 여부 JSON |
| `/fall/sse/fall_alert/` | 최신 미확인 알림 SSE 스트림 |
| `/member/fall/list/` | 낙상 이력 필터/차트/CSV 내보내기 |
| `/member/fall/add/` | 수동 낙상 기록 등록 |
| `/member/fall/alert/list/` | 최근 알림, 위험도 통계, 읽음 처리 |
| `/member/mypage/` | 병동 계정 정보와 탈퇴 |

## 데이터 모델 요약
| 모델 | 주요 필드 | 설명 |
| --- | --- | --- |
| `Member` | `member_id`, `name`, `ward_name`, `phone`, `usage_flag` | 병동별 계정 정보와 사용 여부를 저장합니다. |
| `UserLog` | `member`, `action`, `reg_date` | 회원가입/로그인/로그아웃 이력을 남깁니다. |
| `FallRecord` | `member`, `name`, `fall_date`, `fall_level`, `note` | 병동 담당자가 입력한 낙상 이력, 필터·CSV로 활용됩니다. |
| `FallAlert` | `timestamp`, `message`, `part`, `fall_level`, `is_read` | 실시간 모델이 생성한 알림과 읽음 상태를 저장합니다. |

## 낙상 감지 모델 세부 정보
- `fall/model_gru.py`의 `FallTemporalHybridNet`은 **Temporal Conv1d + Bi-GRU + 어텐션** 구조로 시퀀스 단위 특징을 추출합니다.
- 입력은 선택 관절 6개(코, 눈, 손목, 엉덩) 좌표 + XY 속도를 펼친 30차원 벡터이며, `feature_stats.json`을 이용해 정규화합니다.
- 추론 시 최소 30프레임 이상의 시퀀스를 확보하고, 슬라이딩 윈도우 다수결(`FALL_CONFIRMATION_WINDOW=4`, `FALL_CONFIRMATION_THRESHOLD=2`)로 안정성을 확보합니다.
- 골반 높이 하락(`HIP_DROP_THRESHOLD=0.12`) 검증으로 단순 자세 변화에 대한 오탐을 추가로 차단합니다.
- 위험도는 충격 부위에 따라 고(`머리`)·중(`골반`)·저(`손목/기타`)로 나뉘며, 낮은 신뢰도일 경우 Z축 깊이로 재판단합니다.

## 프라이버시·알림 전략
- `toggle_privacy_mode` API 호출 시 `shared_frame`을 즉시 블라인드 처리하고, 스켈레톤만 표시합니다.
- `FallAlert` 생성 시 `pygame`으로 mp3 경보를 재생하며, Django Channels의 InMemory 레이어와 SSE가 병행되어 웹에 전파됩니다.
- `/fall/fall_status/`는 프론트엔드가 주기적으로 폴링하여 UI 상태(정상/위험)를 표시합니다.

## 개발 이력
- 2025.01.08 ~ 2025.01.20: 프로젝트 구조 설계, Django 기본 세팅, `Member`/`UserLog` 모델 정의
- 2025.01.21 ~ 2025.02.05: 인증번호 회원가입·로그인, 마이페이지·계정 삭제, 사용자 로그 수집 완성
- 2025.02.06 ~ 2025.02.20: MediaPipe Pose + OpenCV 스트림, PyTorch `FallTemporalHybridNet` 감지 파이프라인 구축
- 2025.02.21 ~ 2025.03.05: 낙상 이력 CRUD, 필터/통계 카드, CSV 내보내기 UI 구현
- 2025.03.06 ~ 2025.03.20: Django Channels + SSE 실시간 알림, 읽음 처리, 위험도 통계 대시보드 개발
- 2025.03.21 ~ 2025.04.05: 프라이버시 모드, hip-drop 오탐 방지, MP3 알람, Docker 배포 환경 정비

## 향후 개선 아이디어
- 멀티 카메라 환경을 위한 카메라 풀 관리 및 GPU 추론 서버 분리
- FCM/문자로 확장 가능한 알림 어댑터
- 모델 재학습 자동화 파이프라인과 MLOps(데이터 증강, AutoLabel) 연동
- 환자별 Risk Score 대시보드, 프론트엔드(별도 repo)와의 완전 통합
