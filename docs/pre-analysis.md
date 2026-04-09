# 사전 분석: Embedding Model 서빙 방안

## 1. 모델 분석: CodeT5p-220m

| 항목 | 내용 |
|------|------|
| 모델명 | `Salesforce/codet5p-220m` |
| 아키텍처 | Encoder-Decoder (T5 기반) |
| 파라미터 | 220M |
| 용도 | 코드 이해/생성 (범용) |
| Embedding 전용 여부 | **아님** — 인코더 출력에 pooling 적용 필요 |
| Hidden Size | 768 |
| Max Sequence Length | 512 토큰 |

## 2. 서빙 프레임워크 비교

| 기준 | FastAPI 직접 구현 | TEI (Text Embeddings Inference) | vLLM | Triton Inference Server |
|------|---|---|---|---|
| **CodeT5p 지원** | O (직접 로딩) | X (sentence-transformer 구조 전제) | X (생성 모델 전용) | O (커스텀 백엔드 필요) |
| **Embedding 전용 모델 지원** | O | O (네이티브) | X | O |
| **Pooling 커스텀** | O (완전 제어) | 제한적 (지원 모델만) | 해당 없음 | O (전처리 파이프라인) |
| **배치 최적화** | 직접 구현 | 내장 | 내장 | 내장 |
| **동적 배칭** | 직접 구현 | 내장 | 내장 | 내장 |
| **ONNX/TensorRT 최적화** | 별도 통합 필요 | 내장 | 내장 | 내장 |
| **구현 난이도** | 낮음 | 낮음 (지원 모델일 때) | 해당 없음 | 높음 |
| **인프라 복잡도** | 낮음 (단일 프로세스) | 중간 (Docker 기반) | 중간 | 높음 (모델 리포지토리) |
| **모델 교체 유연성** | 높음 (코드 수정) | 높음 (모델명 변경) | 해당 없음 | 중간 (설정 재배포) |
| **적합 규모** | 소~중 | 중~대 | 대 | 대 |

### 결론: FastAPI 직접 구현 채택

| 근거 | 설명 |
|------|------|
| CodeT5p 호환성 | Encoder-Decoder 모델이라 TEI의 sentence-transformer 파이프라인에 맞지 않음 |
| Pooling 제어 | 모델이 embedding 전용이 아니므로 pooling 전략을 직접 구현/선택해야 함 |
| 모델 크기 | 220M은 별도 서빙 프레임워크 없이도 충분히 서빙 가능 |
| 유연성 | 향후 embedding 전용 모델 전환 시에도 추상화 레이어를 통해 대응 가능 |
| 비동기 처리 | FastAPI의 async 지원으로 동시 요청 처리 |

## 3. Pooling 전략 비교

CodeT5p-220m은 embedding 전용 모델이 아니므로 인코더 출력을 고정 길이 벡터로 변환하는 pooling이 필수.

| Pooling 방식 | 설명 | T5 계열 적합도 | 특징 |
|---|---|---|---|
| **Mean Pooling** | 모든 토큰 hidden state를 attention mask 기반 가중 평균 | **높음 (권장)** | 범용적으로 가장 안정적, 문맥 전체를 반영 |
| CLS Pooling | 첫 번째 토큰의 hidden state 사용 | 낮음 | T5는 BERT와 달리 CLS 토큰에 문장 표현이 집중되지 않음 |
| Max Pooling | 각 차원별 최대값 선택 | 중간 | 특정 피처가 강조됨, 노이즈에 민감 |
| Weighted Mean | 레이어 위치에 따른 가중 평균 | 중간 | 후반 레이어에 더 높은 가중치, 구현 복잡 |

> **기본 전략**: Mean Pooling 채택. API 파라미터로 전략 선택 가능하게 구현.
> **참고**: 향후 embedding 전용 모델 전환 시 pooling이 불필요할 수 있으므로, pooling을 모델 백엔드 내부에 캡슐화.

## 4. 모델 전환 시나리오

> 교체 시 구체적인 고려사항 및 절차: [docs/model-migration.md](model-migration.md)


현재 CodeT5p를 사용하지만, 향후 embedding 전용 모델로 전환할 가능성을 고려한 분석.

| 시나리오 | 예시 모델 | Pooling 필요 | 로딩 방식 변경 | API 변경 |
|---|---|---|---|---|
| 현재 (CodeT5p) | `Salesforce/codet5p-220m` | O (Mean Pooling) | `T5EncoderModel` | 없음 |
| Sentence Transformer 전환 | `all-MiniLM-L6-v2` 등 | X (내장) | `SentenceTransformer` | 없음 |
| 코드 특화 Embedding 전환 | `jinaai/jina-embeddings-v2-base-code` 등 | X (내장) | `AutoModel` | 없음 |
| OpenAI 호환 외부 API 전환 | OpenAI, Cohere 등 | X | HTTP 클라이언트 | 없음 |

### 전환 대응 설계 원칙

| 원칙 | 구현 방법 |
|------|----------|
| 모델 백엔드 추상화 | `EmbeddingBackend` 프로토콜 정의 — `embed(texts) -> vectors` 인터페이스 통일 |
| Pooling 캡슐화 | Pooling 로직을 백엔드 내부에 포함, 외부(라우터)에서는 pooling 존재를 모름 |
| 설정 기반 백엔드 선택 | 환경 변수로 사용할 백엔드 지정, 코드 수정 없이 모델 교체 |
| API 계약 유지 | 어떤 백엔드든 동일한 요청/응답 스키마 유지 |

## 5. 성능 고려사항

| 항목 | 방안 | 우선순위 |
|------|------|----------|
| 배치 인퍼런스 | 여러 입력을 한 번에 처리 | 높음 |
| `torch.no_grad()` | 인퍼런스 시 그래디언트 비활성화로 메모리 절약 | 높음 |
| GPU 자동 감지 | CUDA 사용 가능 시 자동 할당 | 높음 |
| 인코더만 로딩 | `T5EncoderModel`로 디코더 제외, 메모리 절약 | 높음 |
| 토큰 길이 제한 | max_length 설정으로 OOM 방지 | 중간 |
| ONNX 변환 | 인퍼런스 속도 향상 (향후) | 낮음 |
| 동적 배칭 | 요청 큐잉 후 일괄 처리 (향후) | 낮음 |
