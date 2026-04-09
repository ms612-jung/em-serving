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

### 다중 GPU 환경 (H200 4장)

운영 환경은 H200 4장이므로 인스턴스마다 사용할 GPU를 명시적으로 지정한다.

```bash
# GPU 0번에 서버 기동 (포트 8000)
DEVICE=cuda:0 uvicorn app.main:app --host 0.0.0.0 --port 8000

# GPU 1번에 서버 기동 (포트 8001)
DEVICE=cuda:1 uvicorn app.main:app --host 0.0.0.0 --port 8001
```

`DEVICE`를 지정하지 않으면 `cuda:0`이 기본으로 선택된다.
같은 GPU에 여러 인스턴스를 올리는 것도 가능하지만, CodeT5p-220m은 모델 자체가 ~880MB라 GPU당 1개 인스턴스로도 메모리 여유가 충분하다.

## 모델 캐시

- `HF_HOME` 환경 변수로 HuggingFace 모델 다운로드 경로 지정
- 기본값: `~/.cache/huggingface`

## Docker 배포 (추후 구성)

- 베이스 이미지: `nvidia/cuda:12.1-runtime-ubuntu22.04` 또는 `pytorch/pytorch` 공식 이미지
- 호스트 GPU 드라이버 공유, CUDA 런타임 + PyTorch는 이미지 내 설치
- 실행: `docker run --gpus all`
- 호스트의 Python 패키지는 컨테이너에서 사용 불가 (파일시스템 격리)
