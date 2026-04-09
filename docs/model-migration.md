# 모델 교체 가이드

현재 CodeT5p-220m을 사용하고 있으며, 향후 embedding 전용 모델로 교체할 가능성을 고려해 백엔드를 추상화했다.
이 문서는 모델 교체 시 고려해야 할 사항들을 정리한다.

> 모델 후보 분석: [docs/pre-analysis.md](pre-analysis.md)

---

## EmbeddingBackend 프로토콜

모든 백엔드는 `app/backends/base.py`의 `EmbeddingBackend` 프로토콜을 구현해야 한다.

```python
class EmbeddingBackend(Protocol):
    model_name: str

    def warm_up(self) -> None: ...
    async def embed(self, texts: list[str]) -> list[list[float]]: ...
    async def slice_code(self, code: str) -> list[str]: ...
    async def count_tokens(self, text: str) -> int: ...
```

`/v1/embeddings`의 요청/응답 스키마는 백엔드와 무관하게 고정된다.
라우터는 어떤 백엔드가 주입되든 동일하게 동작한다.

---

## 메서드별 교체 시 고려사항

### warm_up()

서버 시작 시 첫 요청 지연을 방지하기 위한 더미 추론.

- **기본 구현**: Protocol의 `...` (no-op) — 구현하지 않으면 아무것도 하지 않음
- **CodeT5pBackend**: `_embed_sync(["def hello(): pass"])` 호출로 CUDA 커널 초기화
- **SentenceTransformerBackend**: 라이브러리가 내부적으로 warm-up을 처리하는 경우 구현 불필요
- **ExternalAPIBackend**: HTTP 호출 기반이라 warm-up 개념이 다름 — 연결 테스트 용도로 구현 가능

### embed()

- **CodeT5p**: `T5EncoderModel` (encoder-only) + mean pooling 필요
- **SentenceTransformer 계열**: 라이브러리가 pooling까지 내장, `model.encode(texts)` 호출로 충분
- **ExternalAPI**: HTTP POST 호출, 배치 크기 제한 확인 필요

### slice_code()

긴 코드를 sliding window로 분할하는 로직. max_token_size는 모델마다 다르다.

| 모델 | max_token_size |
|------|---------------|
| CodeT5p-220m | 512 |
| jina-embeddings-v2-base-code | 8192 |
| text-embedding-3-small (OpenAI) | 8191 |

max_token_size가 커지면 chunking이 필요한 상황 자체가 줄어든다.
단, stride 기반 분할 로직 자체는 재사용 가능하므로 백엔드 내부에서 max_token_size만 변경하면 된다.

### count_tokens()

토크나이저가 바뀌면 동일 텍스트의 토큰 수가 달라진다.
외부 API 백엔드(OpenAI 등)는 토크나이저를 직접 로딩하거나 tiktoken 등 별도 라이브러리를 사용해야 한다.

---

## CodeT5p 특화 설정 vs 범용 설정

모델 교체 시 어떤 설정이 유효한지 파악하기 위한 분류.

### 범용 설정 (모든 백엔드 공통)

| 설정 | 환경 변수 | 설명 |
|------|----------|------|
| `max_token_size` | `MAX_TOKEN_SIZE` | 모델의 최대 토큰 길이 |
| `stride` | `STRIDE` | sliding window 이동 간격 |
| `batch_size` | `BATCH_SIZE` | 한 번에 처리할 텍스트 수 |

### CodeT5p 특화 설정

| 설정 | 환경 변수 | 설명 | 교체 시 |
|------|----------|------|--------|
| `use_fp16` | `USE_FP16` | fp16 추론 활성화 | SentenceTransformer도 지원, ExternalAPI는 불필요 |
| `device` | `DEVICE` | CPU/CUDA 디바이스 지정 | ExternalAPI 백엔드는 불필요 |

> fp16 관련 트레이드오프: [docs/api-design-decisions.md](api-design-decisions.md#fp16-vs-fp32)

---

## 새 백엔드 추가 절차

1. `app/backends/` 아래에 새 파일 생성 (예: `sentence_transformer.py`)
2. `EmbeddingBackend` 프로토콜의 메서드 구현
   - `warm_up()`: 필요 없으면 구현 생략 (no-op으로 동작)
   - `embed()`: `asyncio.to_thread`로 동기 추론 로직을 감쌀 것
   - `slice_code()`, `count_tokens()`: 새 모델의 토크나이저 기준으로 구현
3. `app/config.py`에 백엔드 선택 설정 추가 (현재 미구현 — 아래 참고)
4. `app/main.py` lifespan에서 설정에 따라 백엔드 분기

### 백엔드 선택 설정 (미구현)

현재 `main.py`는 `CodeT5pBackend`를 하드코딩으로 사용한다.
환경 변수로 백엔드를 선택할 수 있도록 구현이 필요하다:

```python
# config.py에 추가 예정
backend_type: str = "codet5p"  # "codet5p" | "sentence_transformer" | "external_api"
```

```python
# main.py lifespan 분기 예정
if settings.backend_type == "codet5p":
    backend = CodeT5pBackend(...)
elif settings.backend_type == "sentence_transformer":
    backend = SentenceTransformerBackend(...)
```
