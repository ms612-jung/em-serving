# 환경 분리: 개발(CPU) / 운영(GPU)

## 환경 비교

| 항목 | 개발 환경 (현재) | 운영 환경 |
|------|-----------------|----------|
| 디바이스 | CPU (GPU 없음) | NVIDIA H200 |
| PyTorch | CPU 전용 (`--index-url .../cpu`) | CUDA 버전 (Docker 이미지 내 설치) |
| 정밀도 | `float32` | `float32` (H200 + 220M 모델이라 최적화 불필요) |
| 배포 방식 | 로컬 실행 | Docker 이미지 |
| OS | - | Ubuntu 22.04 (GPU 장비) |

## 디바이스 관리

- `torch.cuda.is_available()` 기반 자동 감지
- `DEVICE` 환경 변수로 오버라이드 가능

## 모델 캐시

- `HF_HOME` 환경 변수로 HuggingFace 모델 다운로드 경로 지정
- 기본값: `~/.cache/huggingface`

## Docker 배포 (추후 구성)

- 베이스 이미지: `nvidia/cuda:12.1-runtime-ubuntu22.04` 또는 `pytorch/pytorch` 공식 이미지
- 호스트 GPU 드라이버 공유, CUDA 런타임 + PyTorch는 이미지 내 설치
- 실행: `docker run --gpus all`
- 호스트의 Python 패키지는 컨테이너에서 사용 불가 (파일시스템 격리)
