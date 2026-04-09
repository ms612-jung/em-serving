"""POST /v1/embeddings 엔드포인트 테스트.

실제 모델을 로딩하지 않고 FakeBackend로 대체한다.
"""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.main import app


FAKE_DIM = 4  # 테스트용 embedding 차원


class FakeBackend:
    """실제 모델 없이 동작하는 테스트용 백엔드."""

    model_name = "fake-model"

    def warm_up(self) -> None:
        pass

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(i)] * FAKE_DIM for i in range(len(texts))]

    async def slice_code(self, code: str) -> list[str]:
        # 100자 단위로 분할 (토큰 대신 문자 수 기준, 테스트용 단순화)
        if len(code) <= 100:
            return [code]
        return [code[i : i + 100] for i in range(0, len(code), 100)]

    async def count_tokens(self, text: str) -> int:
        # 공백 기준 단어 수로 단순화
        return len(text.split())


@pytest_asyncio.fixture
async def client():
    app.state.backend = FakeBackend()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


# ── chunking=false ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_single_input(client):
    resp = await client.post("/v1/embeddings", json={"input": "hello world"})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["data"]) == 1
    assert body["data"][0]["index"] == 0
    assert len(body["data"][0]["embedding"]) == FAKE_DIM
    assert body["data"][0]["token_count"] == 2  # "hello world" → 2 단어
    assert body["model"] == "fake-model"
    assert body["usage"]["total_tokens"] == 2


@pytest.mark.asyncio
async def test_batch_input(client):
    resp = await client.post("/v1/embeddings", json={"input": ["hello world", "def foo(): pass"]})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["data"]) == 2
    assert body["data"][0]["index"] == 0
    assert body["data"][1]["index"] == 1
    assert body["usage"]["total_tokens"] == 2 + 3  # 2 + 3 단어


@pytest.mark.asyncio
async def test_chunking_false_is_default(client):
    """chunking 파라미터를 생략하면 false로 동작한다."""
    resp = await client.post("/v1/embeddings", json={"input": "hello"})
    assert resp.status_code == 200
    data = resp.json()["data"][0]
    # chunking=false 응답은 index 필드를 가짐
    assert "index" in data
    assert "embeddings" not in data


# ── chunking=true ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_chunking_short_input(client):
    """100자 이하 입력은 분할 없이 청크 1개로 반환된다."""
    resp = await client.post("/v1/embeddings", json={"input": "short code", "chunking": True})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["data"]) == 1
    item = body["data"][0]
    assert item["input_index"] == 0
    assert item["chunk_count"] == 1
    assert len(item["embeddings"]) == 1


@pytest.mark.asyncio
async def test_chunking_long_input(client):
    """100자 초과 입력은 여러 청크로 분할된다."""
    long_code = "x" * 250  # FakeBackend는 100자 단위 분할 → 3청크
    resp = await client.post("/v1/embeddings", json={"input": long_code, "chunking": True})
    assert resp.status_code == 200
    body = resp.json()
    item = body["data"][0]
    assert item["chunk_count"] == 3
    assert len(item["embeddings"]) == 3
    # 각 청크의 index가 순서대로 부여됐는지 확인
    for j, emb in enumerate(item["embeddings"]):
        assert emb["index"] == j


@pytest.mark.asyncio
async def test_chunking_batch_input(client):
    """chunking=true + 배치 입력: 입력마다 input_index가 올바르게 부여된다."""
    resp = await client.post(
        "/v1/embeddings",
        json={"input": ["short", "x" * 250], "chunking": True},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["data"]) == 2
    assert body["data"][0]["input_index"] == 0
    assert body["data"][0]["chunk_count"] == 1
    assert body["data"][1]["input_index"] == 1
    assert body["data"][1]["chunk_count"] == 3


@pytest.mark.asyncio
async def test_chunking_total_tokens(client):
    """usage.total_tokens는 모든 청크의 토큰 수 합산이다."""
    long_code = "a b " * 100  # 400자, 3청크로 분할
    resp = await client.post("/v1/embeddings", json={"input": long_code, "chunking": True})
    assert resp.status_code == 200
    body = resp.json()
    embeddings = body["data"][0]["embeddings"]
    expected_total = sum(e["token_count"] for e in embeddings)
    assert body["usage"]["total_tokens"] == expected_total


# ── 기타 ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
