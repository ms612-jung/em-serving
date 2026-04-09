import asyncio

import torch
from transformers import AutoTokenizer, T5EncoderModel


class CodeT5pBackend:
    """CodeT5p 인코더 + Mean Pooling 기반 embedding 백엔드."""

    def __init__(
        self,
        model_name: str = "Salesforce/codet5p-220m",
        device: str | None = None,
        max_token_size: int = 512,
        stride: int = 256,
        batch_size: int = 16,
        use_fp16: bool = False,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_token_size = max_token_size
        self.stride = stride
        self.batch_size = batch_size
        self._use_fp16 = use_fp16 and self.device == "cuda"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = T5EncoderModel.from_pretrained(model_name).to(self.device)
        if self._use_fp16:
            self.model = self.model.half()
        self.model.eval()

    def warm_up(self) -> None:
        """서버 시작 시 더미 추론으로 첫 요청 지연을 방지한다."""
        self._embed_sync(["def hello(): pass"])

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """텍스트 리스트를 받아 mean pooling된 embedding 벡터를 반환한다."""
        return await asyncio.to_thread(self._embed_sync, texts)

    async def slice_code(self, code: str) -> list[str]:
        """긴 코드를 sliding window 방식으로 분할한다."""
        return await asyncio.to_thread(self._slice_code_sync, code)

    async def count_tokens(self, text: str) -> int:
        """텍스트의 토큰 수를 반환한다."""
        return await asyncio.to_thread(self._count_tokens_sync, text)

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_token_size,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad(), torch.autocast(device_type=self.device, enabled=self._use_fp16):
                outputs = self.model(**encoded)

            embeddings = self._mean_pooling(outputs.last_hidden_state, encoded["attention_mask"])
            all_embeddings.extend(embeddings.cpu().tolist())
        return all_embeddings

    def _slice_code_sync(self, code: str) -> list[str]:
        token_ids = self.tokenizer.encode(
            code, truncation=False, add_special_tokens=False,
        )

        if len(token_ids) <= self.max_token_size:
            return [code]

        snippets: list[list[int]] = []
        for start in range(0, len(token_ids), self.stride):
            end = min(start + self.max_token_size, len(token_ids))
            snippets.append(token_ids[start:end])
            if end >= len(token_ids):
                break

        return [
            self.tokenizer.decode(snip, skip_special_tokens=True)
            for snip in snippets
        ]

    def _count_tokens_sync(self, text: str) -> int:
        return len(
            self.tokenizer.encode(text, truncation=False, add_special_tokens=False)
        )

    @staticmethod
    def _mean_pooling(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Attention mask를 반영한 mean pooling."""
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = (hidden_state * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / sum_mask
