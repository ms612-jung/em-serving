"""로컬 테스트용 CodeT5p-220m 임베딩 모델.

T5EncoderModel로 encoder hidden state를 추출하고,
mean pooling으로 고정 길이(768차원) 벡터를 생성한다.
(220m은 seq2seq 모델이라 직접 임베딩 출력이 없으므로 encoder + pooling 방식 사용)

운영 환경에서는 별도의 임베딩 서비스 구현체를 사용한다.
"""

import asyncio
import logging

import torch
from transformers import AutoTokenizer, T5EncoderModel

from app.embedding.base import EmbeddingModel

logger = logging.getLogger(__name__)


class LocalCodeT5pEmbeddingModel(EmbeddingModel):

    def __init__(
        self,
        model_dir: str,
        max_token_size: int = 512,
        stride: int = 256,
        batch_size: int = 16,
        use_torch_compile: bool = True,
        use_fp16: bool = False,
        cuda_device_id: int = 0,
    ):
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{cuda_device_id}")
        else:
            self.device = torch.device("cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.max_token_size = max_token_size
        self.stride = stride
        self.batch_size = batch_size
        self._use_fp16 = use_fp16 and self.device.type == "cuda"

        self.model = T5EncoderModel.from_pretrained(model_dir).to(self.device)
        if self._use_fp16:
            self.model = self.model.half()
        self.model.eval()

        # torch.compile로 추론 그래프 최적화 (PyTorch 2.0+)
        # 실제 컴파일은 첫 forward pass(warm_up) 시점에 발생하므로 거기서 검증
        self._use_torch_compile = use_torch_compile and hasattr(torch, "compile")
        if self._use_torch_compile:
            self.model = torch.compile(self.model, mode="reduce-overhead")
            logger.info("torch.compile 등록 완료 (실제 컴파일은 warm-up 시 수행)")

        logger.info(
            "CodeT5p-220m 모델 로드 완료 (device=%s, max_token=%d, stride=%d, batch_size=%d, fp16=%s)",
            self.device, max_token_size, stride, batch_size, self._use_fp16,
        )

    def warm_up(self) -> None:
        """torch.compile 첫 컴파일 비용을 해소하기 위한 더미 추론.

        torch.compile은 첫 forward pass에서 실제 그래프 컴파일이 발생한다.
        Windows CPU 등 C++ 컴파일러가 없는 환경에서는 여기서 실패하므로,
        실패 시 torch.compile을 해제하고 기본 모드로 fallback한다.
        """
        logger.info("모델 warm-up 시작")
        try:
            self._embed_sync("def hello(): pass")
        except Exception:
            if self._use_torch_compile:
                logger.warning(
                    "torch.compile warm-up 실패 — 기본 모드로 fallback "
                    "(Windows CPU 환경에서는 C++ 컴파일러가 필요합니다)",
                    exc_info=True,
                )
                # compile된 모델을 원본으로 되돌림
                self.model = self.model._orig_mod
                self._use_torch_compile = False
                # fallback 모드로 재시도
                self._embed_sync("def hello(): pass")
            else:
                raise
        logger.info("모델 warm-up 완료")

    def _embed_sync(self, code_snippet: str) -> list[float]:
        """동기 임베딩 생성 — asyncio.to_thread에서 호출된다."""
        inputs = self.tokenizer(
            code_snippet,
            return_tensors="pt",
            max_length=self.max_token_size,
            truncation=True,
            padding="longest",
        ).to(self.device)

        with torch.no_grad(), torch.autocast(
            device_type=self.device.type, enabled=self._use_fp16
        ):
            outputs = self.model(**inputs)

        # mean pooling: attention_mask를 고려하여 평균
        hidden_states = outputs.last_hidden_state.float()
        attention_mask = inputs["attention_mask"].unsqueeze(-1).float()
        masked_hidden = hidden_states * attention_mask
        emb = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1)

        return emb.squeeze(0).cpu().tolist()

    async def get_embedding(self, code_snippet: str) -> list[float]:
        return await asyncio.to_thread(self._embed_sync, code_snippet)

    def _embed_batch_sync(self, code_snippets: list[str]) -> list[list[float]]:
        """배치 임베딩 생성 — 여러 스니펫을 한 번의 forward pass로 처리한다."""
        if not code_snippets:
            return []

        all_embeddings: list[list[float]] = []
        for i in range(0, len(code_snippets), self.batch_size):
            batch = code_snippets[i : i + self.batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                max_length=self.max_token_size,
                truncation=True,
                padding="longest",
            ).to(self.device)

            with torch.no_grad(), torch.autocast(
                device_type=self.device.type, enabled=self._use_fp16
            ):
                outputs = self.model(**inputs)

            # 배치 mean pooling: (B, seq_len, 768) → (B, 768)
            hidden_states = outputs.last_hidden_state.float()
            attention_mask = inputs["attention_mask"].unsqueeze(-1).float()
            masked_hidden = hidden_states * attention_mask
            embs = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1)

            all_embeddings.extend(embs.cpu().tolist())
        return all_embeddings

    async def get_embeddings_batch(self, code_snippets: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._embed_batch_sync, code_snippets)

    def _slice_sync(self, code: str) -> list[str]:
        """동기 코드 슬라이싱 — sliding window 방식."""
        token_ids = self.tokenizer.encode(
            code, truncation=False, add_special_tokens=False
        )

        # 토큰 수가 max_token_size 이하이면 그대로 반환
        if len(token_ids) <= self.max_token_size:
            return [code]

        # sliding window로 스니펫 분할
        snippets = []
        for start in range(0, len(token_ids), self.stride):
            end = min(start + self.max_token_size, len(token_ids))
            snippet_tokens = token_ids[start:end]
            snippets.append(snippet_tokens)
            if end >= len(token_ids):
                break

        # 토큰 ID를 텍스트로 디코딩
        return [
            self.tokenizer.decode(snip, skip_special_tokens=True)
            for snip in snippets
        ]

    async def slice_code(self, code: str) -> list[str]:
        return await asyncio.to_thread(self._slice_sync, code)

    def _count_tokens_sync(self, code_snippet: str) -> int:
        return len(
            self.tokenizer.encode(
                code_snippet, truncation=False, add_special_tokens=False
            )
        )

    async def get_token_count(self, code_snippet: str) -> int:
        return await asyncio.to_thread(self._count_tokens_sync, code_snippet)
