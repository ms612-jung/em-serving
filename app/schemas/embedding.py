from pydantic import BaseModel


class EmbeddingRequest(BaseModel):
    input: str | list[str]
    chunking: bool = False


class EmbeddingData(BaseModel):
    index: int
    embedding: list[float]
    token_count: int


class ChunkedEmbeddingData(BaseModel):
    input_index: int
    chunk_count: int
    embeddings: list[EmbeddingData]


class EmbeddingUsage(BaseModel):
    total_tokens: int


class EmbeddingResponse(BaseModel):
    data: list[EmbeddingData] | list[ChunkedEmbeddingData]
    model: str
    usage: EmbeddingUsage
