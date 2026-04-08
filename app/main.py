from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.backends.codet5p import CodeT5pBackend
from app.config import settings
from app.routers import embeddings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작: 모델 로딩
    app.state.backend = CodeT5pBackend(
        model_name=settings.model_name,
        device=settings.device,
    )
    yield
    # 종료: 정리
    del app.state.backend


app = FastAPI(title="em-serving", lifespan=lifespan)
app.include_router(embeddings.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
