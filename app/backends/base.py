from typing import Protocol


class EmbeddingBackend(Protocol):
    """Embedding 백엔드 인터페이스. 모든 백엔드는 이 프로토콜을 구현한다."""

    model_name: str

    def embed(self, texts: list[str]) -> list[list[float]]:
        """텍스트 리스트를 받아 embedding 벡터 리스트를 반환한다."""
        ...
