from fastapi import APIRouter, Request

from app.schemas.embedding import (
    ChunkedEmbeddingData,
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
)

router = APIRouter()


@router.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: Request, body: EmbeddingRequest):
    """텍스트를 embedding 벡터로 변환한다.

    chunking=false(기본값): 입력을 그대로 embedding. max_token_size 초과 시 truncation.
    chunking=true: 긴 입력을 sliding window로 분할 후 청크별 embedding.
    """
    backend = request.app.state.backend

    texts = [body.input] if isinstance(body.input, str) else body.input

    if not body.chunking:
        vectors = await backend.embed(texts)
        token_counts = [await backend.count_tokens(text) for text in texts]

        return EmbeddingResponse(
            data=[
                EmbeddingData(index=i, embedding=vec, token_count=tc)
                for i, (vec, tc) in enumerate(zip(vectors, token_counts))
            ],
            model=backend.model_name,
            usage=EmbeddingUsage(total_tokens=sum(token_counts)),
        )

    # chunking=true
    data: list[ChunkedEmbeddingData] = []
    total_tokens = 0
    for input_index, text in enumerate(texts):
        snippets = await backend.slice_code(text)
        vectors = await backend.embed(snippets)
        token_counts = [await backend.count_tokens(s) for s in snippets]

        data.append(ChunkedEmbeddingData(
            input_index=input_index,
            chunk_count=len(snippets),
            embeddings=[
                EmbeddingData(index=j, embedding=vec, token_count=tc)
                for j, (vec, tc) in enumerate(zip(vectors, token_counts))
            ],
        ))
        total_tokens += sum(token_counts)

    return EmbeddingResponse(
        data=data,
        model=backend.model_name,
        usage=EmbeddingUsage(total_tokens=total_tokens),
    )
