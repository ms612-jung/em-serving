# em-serving

Embedding Model 서빙 서버 (FastAPI 기반)

> 사전 분석 문서: [docs/pre-analysis.md](docs/pre-analysis.md)

## 프로젝트 개요

텍스트/코드 입력을 받아 embedding 벡터를 반환하는 API 서버.
현재 CodeT5p-220m을 사용하며, 향후 embedding 전용 모델로 교체 가능한 구조.

## 기술 스택

- **Framework**: FastAPI + Uvicorn
- **ML**: PyTorch, HuggingFace Transformers (`>=4.46,<5` — 5.x에서 CodeT5p 토크나이저 호환성 문제)
- **Python**: 3.12+
- **현재 모델**: `Salesforce/codet5p-220m` (encoder-decoder, pooling 필요)

## 핵심 설계 원칙

### 모델 백엔드 추상화

모델 교체를 코드 수정 최소화로 가능하게 하기 위해 백엔드를 추상화한다.

```
EmbeddingBackend (Protocol)
├── embed(texts: list[str]) -> list[list[float]]
│
├── CodeT5pBackend        ← 현재: T5EncoderModel + Mean Pooling
├── SentenceTransformerBackend  ← 향후: sentence-transformers 모델
└── ExternalAPIBackend    ← 향후: 외부 API 호출
```

- **Pooling은 백엔드 내부에 캡슐화**: CodeT5p 백엔드는 pooling을 내부 처리, embedding 전용 모델 백엔드는 pooling 불필요
- **설정 기반 백엔드 선택**: 환경 변수로 백엔드 지정, 코드 수정 없이 모델 교체
- **API 계약 고정**: 어떤 백엔드든 동일한 요청/응답 스키마 유지

## API

| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/v1/embeddings` | 텍스트를 embedding 벡터로 변환 |
| GET | `/health` | 서버 상태 확인 |

### POST /v1/embeddings

요청:
```json
{
  "input": "string 또는 string[]"
}
```

응답:
```json
{
  "data": [
    {
      "index": 0,
      "embedding": [0.1, 0.2, ...],
      "token_count": 15
    }
  ],
  "model": "codet5p-220m",
  "usage": {
    "total_tokens": 15
  }
}
```

### 사용 예시

```bash
# 단일 입력
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "hello world"}'

# 배치 입력
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": ["hello world", "def foo(): pass"]}'
```

## 프로젝트 구조

```
em-serving/
├── CLAUDE.md
├── pyproject.toml
├── docs/
│   └── pre-analysis.md         # 사전 분석 (서빙 방식, pooling, 모델 비교)
├── app/
│   ├── main.py                 # FastAPI 앱 + lifespan (모델 로딩) + /health
│   ├── config.py               # 설정 (모델명, 백엔드 타입, 디바이스 등)
│   ├── routers/
│   │   └── embeddings.py       # /v1/embeddings 엔드포인트
│   ├── backends/
│   │   ├── base.py             # EmbeddingBackend 프로토콜
│   │   ├── codet5p.py          # CodeT5p 백엔드 (인코더 + pooling)
│   │   └── sentence_transformer.py  # (향후) ST 백엔드
│   └── schemas/
│       └── embedding.py        # Pydantic 요청/응답 스키마
└── tests/
    └── test_embeddings.py
```

## 개발 규칙

- Python 코드 포매팅: ruff
- 타입 힌트 필수
- 모델은 앱 시작 시 1회 로딩 (lifespan event)
- 환경 변수로 설정 관리 (pydantic-settings)

## 환경

- **개발**: CPU (현재 장비, GPU 없음)
- **운영**: GPU 장비에 Docker 이미지로 배포 (상세: [docs/environment.md](docs/environment.md))
- **프록시 환경**: HuggingFace 모델 다운로드 시 SSL 인증서 설정 필요
  ```bash
  export SSL_CERT_FILE=~/proxy.crt
  export REQUESTS_CA_BUNDLE=~/proxy.crt
  ```

## 핵심 구현 포인트

1. **백엔드 프로토콜**: `embed(texts) -> vectors` 인터페이스 통일
2. **CodeT5p 백엔드**: `T5EncoderModel` (인코더만 로딩) + attention mask 기반 mean pooling
3. **배치 처리**: 여러 입력을 한 번에 받아 배치 인퍼런스
4. **torch.no_grad()**: 인퍼런스 시 그래디언트 비활성화
5. **디바이스 관리**: `torch.cuda.is_available()` 기반 자동 감지

## 명령어

```bash
# venv 활성화
source .venv/bin/activate

# 의존성 설치 (최초 1회: CPU 전용 PyTorch 먼저)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"

# 서버 실행
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 테스트
pytest tests/

# 린트
ruff check . && ruff format --check .
```
