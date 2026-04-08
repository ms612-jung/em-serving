from fastapi import APIRouter, Request

from app.schemas.embedding import EmbeddingData, EmbeddingRequest, EmbeddingResponse, EmbeddingUsage

router = APIRouter()


@router.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: Request, body: EmbeddingRequest):
    backend = request.app.state.backend

    texts = [body.input] if isinstance(body.input, str) else body.input
    vectors = backend.embed(texts)

    # 토큰 수 계산
    encoded = backend.tokenizer(texts, padding=False, truncation=True, max_length=512)
    token_counts = [len(ids) for ids in encoded["input_ids"]]

    data = [
        EmbeddingData(index=i, embedding=vec, token_count=tc)
        for i, (vec, tc) in enumerate(zip(vectors, token_counts))
    ]

    return EmbeddingResponse(
        data=data,
        model=backend.model_name,
        usage=EmbeddingUsage(total_tokens=sum(token_counts)),
    )
