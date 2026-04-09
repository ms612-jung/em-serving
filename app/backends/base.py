from typing import Protocol


class EmbeddingBackend(Protocol):
    """Embedding 백엔드 인터페이스. 모든 백엔드는 이 프로토콜을 구현한다."""

    model_name: str

    def warm_up(self) -> None:
        """서버 시작 시 첫 요청 지연을 방지하기 위한 준비 작업을 수행한다."""
        ...

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """텍스트 리스트를 받아 embedding 벡터 리스트를 반환한다."""
        ...

    async def slice_code(self, code: str) -> list[str]:
        """긴 코드를 max_token_size 이하의 스니펫으로 분할한다."""
        ...

    async def count_tokens(self, text: str) -> int:
        """텍스트의 토큰 수를 반환한다."""
        ...
