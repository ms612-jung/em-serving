from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.backends.codet5p import CodeT5pBackend
from app.config import settings
from app.routers import embeddings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작: 모델 로딩
    backend = CodeT5pBackend(
        model_name=settings.model_name,
        device=settings.device,
        max_token_size=settings.max_token_size,
        stride=settings.stride,
        batch_size=settings.batch_size,
        use_fp16=settings.use_fp16,
    )
    backend.warm_up()
    app.state.backend = backend
    yield
    # 종료: 정리
    del app.state.backend


app = FastAPI(title="em-serving", lifespan=lifespan)
app.include_router(embeddings.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
