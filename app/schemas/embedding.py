from pydantic import BaseModel


class EmbeddingRequest(BaseModel):
    input: str | list[str]


class EmbeddingData(BaseModel):
    index: int
    embedding: list[float]
    token_count: int


class EmbeddingUsage(BaseModel):
    total_tokens: int


class EmbeddingResponse(BaseModel):
    data: list[EmbeddingData]
    model: str
    usage: EmbeddingUsage
