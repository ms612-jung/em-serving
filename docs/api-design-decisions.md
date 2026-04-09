# API 설계 결정 기록

## chunking 옵션 도입 배경

### 문제

임베딩 모델은 최대 입력 길이(max_token_size=512)가 있다. 이를 초과하는 긴 코드 파일을 처리하는 방식이 필요했다.

### 검토한 방식

#### 방식 1: 호출자가 슬라이싱

호출자가 코드를 512 토큰 이하로 잘라서 API를 여러 번 호출한다.

```
호출자 → slice → [청크1, 청크2, ..., 청크N]
호출자 → POST /v1/embeddings {input: 청크1}
호출자 → POST /v1/embeddings {input: 청크2}
...
호출자 → POST /v1/embeddings {input: 청크N}
```

단점:
- 호출 횟수가 청크 수만큼 늘어남 (네트워크 왕복 오버헤드)
- 호출자가 슬라이싱 로직(stride, 겹침 크기)을 알아야 함
- 슬라이싱 방식이 바뀌면 호출자도 수정 필요

#### 방식 2: 별도 엔드포인트 분리

`/v1/embeddings/direct`와 `/v1/embeddings/chunked`로 엔드포인트를 나눈다.

단점:
- 엔드포인트 네이밍이 어색함 (chunked는 "이미 잘린 입력"처럼 읽힘)
- 동일한 요청 스키마에 대해 엔드포인트가 2개로 분산

#### 방식 3: 단일 엔드포인트 + chunking 파라미터 (채택)

OpenAI, Jina AI 등 주요 임베딩 API의 관례를 따라, 하나의 엔드포인트에서 옵션으로 동작을 선택한다.

```json
POST /v1/embeddings
{"input": "긴 코드...", "chunking": true}
```

장점:
- API 호출 1회로 슬라이싱 + 임베딩 완료 (네트워크 오버헤드 최소화)
- 호출자는 슬라이싱 로직을 몰라도 됨
- GPU 환경에서 서버가 배치 처리를 효율적으로 수행
- 슬라이싱 파라미터(stride, max_token_size) 변경이 서버에 캡슐화

### 동작 정리

| chunking | 입력 길이 | 동작 |
|----------|----------|------|
| false (기본값) | 512 이하 | 그대로 임베딩 |
| false (기본값) | 512 초과 | truncation (뒷부분 유실) |
| true | 512 이하 | 그대로 임베딩 (분할 불필요) |
| true | 512 초과 | sliding window 분할 후 청크별 임베딩 |

### Sliding window 파라미터

- `max_token_size=512`: 각 청크의 최대 토큰 수
- `stride=256`: 윈도우 이동 간격 (겹침 = max_token_size - stride = 256 토큰)
- `batch_size=16`: 한 번의 forward pass에 처리하는 청크 수 (OOM 방지)

모두 환경 변수로 설정 가능하다.

### batch_size 튜닝 가이드

`batch_size`는 한 번의 forward pass에 모델에 넣는 텍스트 개수의 상한이다.
API 요청 하나에 여러 텍스트가 들어오거나, chunking으로 슬라이스가 많이 생겼을 때 이를 나눠서 처리한다.

배치가 클수록 GPU 활용률이 높아져 처리량(throughput)이 증가하지만, GPU 메모리를 더 많이 소비한다.
배치가 작으면 forward pass 횟수가 늘어나 오버헤드가 생긴다.

#### 메모리 사용량 추정

한 배치의 GPU 메모리 사용량은 대략 다음과 같다:

```
배치 메모리 ≈ batch_size × max_token_size × hidden_size × bytes_per_param
```

CodeT5p-220m (hidden_size=768, float32) 기준:
- batch_size=16: 16 × 512 × 768 × 4 ≈ **24MB** (중간 텐서 포함 시 ~100MB)
- batch_size=128: 128 × 512 × 768 × 4 ≈ **192MB** (중간 텐서 포함 시 ~800MB)
- batch_size=256: ≈ **384MB** (중간 텐서 포함 시 ~1.6GB)

여기에 모델 파라미터 자체가 ~880MB (220M × 4 bytes) 를 차지한다.

#### 환경별 권장값

| 환경 | GPU 메모리 | 권장 batch_size | 비고 |
|------|-----------|----------------|------|
| CPU (개발) | - | 8~16 | 시스템 RAM 기준, 너무 크면 느려짐 |
| T4 (16GB) | 16GB | 32~64 | 추론 전용 저가 GPU |
| A100 (80GB) | 80GB | 128~256 | 대규모 배치 처리 가능 |
| H200 (141GB HBM3e) | 141GB | 256~512 | CodeT5p-220m은 작은 모델이라 메모리 여유 충분 |

H200 같은 대용량 GPU에서는 CodeT5p-220m(220M 파라미터) 정도의 모델은 메모리 병목이 거의 없다.
batch_size=512로 설정해도 모델+배치 메모리가 ~5GB 수준이므로, 남는 메모리로 동시 요청 처리에 여유를 둘 수 있다.

#### 설정 방법

```bash
# 환경 변수로 설정
export BATCH_SIZE=128

# 또는 서버 실행 시 인라인
BATCH_SIZE=128 uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## warm_up()

### 목적

모델 로딩 후 첫 번째 forward pass는 CUDA 커널 초기화 등으로 이후 요청보다 수초 느리다.
서버 시작 시 더미 입력으로 먼저 추론을 실행해 이 지연을 서버 시작 시점에 미리 처리한다.
클라이언트의 첫 요청에서는 이미 준비된 상태로 응답한다.

```python
def warm_up(self) -> None:
    self._embed_sync(["def hello(): pass"])
```

`main.py` lifespan에서 백엔드 생성 직후, `app.state.backend` 할당 전에 호출한다.

### 백엔드별 구현 여부

`warm_up()`은 `EmbeddingBackend` 프로토콜에 포함되어 있지만, 구현이 필요 없는 백엔드는 생략해도 된다 (Protocol의 `...`가 no-op으로 동작).

| 백엔드 | 구현 필요 여부 | 이유 |
|--------|-------------|------|
| CodeT5pBackend | 필요 | CUDA 커널 초기화 지연 발생 |
| SentenceTransformerBackend | 선택적 | 라이브러리가 내부 처리하는 경우 있음 |
| ExternalAPIBackend | 선택적 | 연결 테스트 용도로 구현 가능 |

> 백엔드별 상세: [docs/model-migration.md](model-migration.md)

## fp16 vs fp32

### 트레이드오프

| | fp32 (기본값) | fp16 |
|---|---|---|
| 정밀도 | 높음 | 낮음 (수치 오차 가능) |
| 메모리 | 2배 | 1배 |
| 연산 속도 | 기준 | H100/H200 기준 ~2x 빠름 |

### embedding 추론에서 fp16 정밀도 손실이 문제가 되는가?

실용적으로 문제가 되는 경우는 드물다.

- 추론 단계라 gradient 누적 오차가 없음
- embedding 벡터의 코사인 유사도 계산 수준에서는 fp16 정밀도로 충분

### 기본값을 fp32로 유지하는 이유

H200(141GB HBM3e) 환경에서는 메모리 여유가 충분해 fp32를 써도 병목이 없다.
처리량이 병목이 될 때 fp16으로 전환하면 된다.

### 설정 방법

```bash
# fp16 활성화 (CUDA 환경에서만 적용, CPU에서는 무시됨)
USE_FP16=true uvicorn app.main:app --host 0.0.0.0 --port 8000
```

fp16은 CUDA가 없으면 자동으로 무시된다 (`use_fp16=True`로 설정해도 CPU에서는 fp32로 동작).
