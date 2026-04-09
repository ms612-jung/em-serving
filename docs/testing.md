# 테스트 명세

## 프로젝트 구조

```
em-serving/
├── app/
│   ├── main.py                  # FastAPI 앱 + lifespan (모델 로딩, warm_up)
│   ├── config.py                # 환경 변수 기반 설정
│   ├── routers/
│   │   └── embeddings.py        # POST /v1/embeddings 라우터
│   ├── backends/
│   │   ├── base.py              # EmbeddingBackend 프로토콜
│   │   └── codet5p.py           # CodeT5p 백엔드 구현
│   └── schemas/
│       └── embedding.py         # 요청/응답 Pydantic 스키마
└── tests/
    └── test_embeddings.py       # 엔드포인트 통합 테스트
```

---

## 테스트 전략

### 실제 모델을 쓰지 않는 이유

`CodeT5pBackend`는 초기화 시 HuggingFace에서 모델을 다운로드하고 수백MB를 메모리에 로딩한다.
테스트마다 이 과정이 발생하면 실행 시간이 수분 이상 걸리고 GPU/CPU 자원을 낭비한다.

대신 `FakeBackend`를 주입해 라우터 로직과 응답 스키마만 검증한다.

```
실제 요청 흐름:         테스트 흐름:
  HTTP 요청               HTTP 요청 (httpx AsyncClient)
    ↓                       ↓
  라우터                  라우터          ← 검증 대상
    ↓                       ↓
  CodeT5pBackend          FakeBackend     ← 대체
    ↓
  T5EncoderModel (생략)
```

### FakeBackend 동작

| 메서드 | 실제 동작 | FakeBackend 동작 |
|--------|----------|----------------|
| `embed(texts)` | T5 모델 추론 + mean pooling | 인덱스 기반 더미 벡터 반환 (`[[0.0, ...], [1.0, ...], ...]`) |
| `slice_code(code)` | tokenizer 기반 sliding window | 100자 단위 문자 분할 |
| `count_tokens(text)` | tokenizer.encode 호출 | 공백 기준 단어 수 반환 |
| `warm_up()` | 더미 추론 실행 | no-op |

FakeBackend의 분할 기준(100자)과 토큰 계산(단어 수)은 실제 모델과 다르다.
라우터의 분기 로직과 응답 구조가 목적이므로 정확한 수치보다 **동작 흐름**이 중요하다.

---

## 구성요소별 테스트 범위

### 1. 라우터 (`app/routers/embeddings.py`)

라우터가 담당하는 로직:
- `input`이 문자열이면 리스트로 변환
- `chunking` 값에 따라 분기
- 백엔드 호출 결과를 응답 스키마로 조립
- `usage.total_tokens` 집계

| 테스트 | 검증 내용 |
|--------|----------|
| `test_single_input` | 문자열 입력 → 리스트 변환 후 embed 호출, 응답 구조 전체 |
| `test_batch_input` | 리스트 입력 → 각 항목 index 부여, total_tokens 합산 |
| `test_chunking_false_is_default` | chunking 생략 시 false 분기로 동작 |
| `test_chunking_short_input` | chunking=true, 분할 불필요 → chunk_count=1 |
| `test_chunking_long_input` | chunking=true, 분할 발생 → 청크 수·index 순서 |
| `test_chunking_batch_input` | chunking=true + 배치 → input_index 부여 |
| `test_chunking_total_tokens` | usage.total_tokens = 전체 청크 token_count 합산 |

### 2. 응답 스키마 (`app/schemas/embedding.py`)

Pydantic 스키마는 별도 단위 테스트 없이 라우터 테스트에서 암묵적으로 검증된다.
응답 JSON의 필드 존재 여부와 타입을 각 테스트에서 확인한다.

| 스키마 | 검증 방법 |
|--------|----------|
| `EmbeddingData` | `index`, `embedding`, `token_count` 필드 확인 |
| `ChunkedEmbeddingData` | `input_index`, `chunk_count`, `embeddings` 필드 확인 |
| `EmbeddingResponse` | `data`, `model`, `usage` 필드 확인 |
| chunking 분기 구분 | `"index" in data` / `"embeddings" not in data` 로 응답 타입 구분 |

### 3. 헬스체크 (`app/main.py`)

| 테스트 | 검증 내용 |
|--------|----------|
| `test_health` | GET /health → 200, `{"status": "ok"}` |

### 4. 현재 미검증 영역

| 영역 | 이유 | 향후 방향 |
|------|------|----------|
| `CodeT5pBackend` 내부 로직 | 모델 로딩 필요 | 통합 테스트 또는 모델 fixture로 별도 구성 |
| `slice_code()` sliding window 경계 | FakeBackend는 문자 단위로 단순화 | 실제 tokenizer 기반 단위 테스트 필요 |
| fp16 동작 | CUDA 환경 필요 | GPU 환경에서 별도 검증 |
| 오류 케이스 (빈 입력, 잘못된 타입 등) | 미구현 | 추가 예정 |

---

## 실행 방법

```bash
# 전체 테스트
pytest tests/

# 상세 출력
pytest tests/ -v

# 특정 테스트만
pytest tests/test_embeddings.py::test_chunking_long_input
```

---

## 의존성

| 패키지 | 용도 |
|--------|------|
| `pytest` | 테스트 프레임워크 |
| `pytest-asyncio` | async 테스트 함수 지원 |
| `httpx` | `AsyncClient`로 FastAPI 앱에 HTTP 요청 |
