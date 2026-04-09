"""ONNX Runtime 기반 CodeT5p-220m 임베딩 모델.

CPU 환경에서 PyTorch 대비 ~1.5-2x 추론 속도 향상을 제공한다.
사전에 scripts/export_onnx.py로 ONNX 모델을 변환해야 한다.
"""

import asyncio
import logging
import os

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from app.embedding.base import EmbeddingModel

logger = logging.getLogger(__name__)


class OnnxCodeT5pEmbeddingModel(EmbeddingModel):

    def __init__(
        self,
        model_dir: str,
        onnx_model_path: str,
        max_token_size: int = 512,
        stride: int = 256,
        batch_size: int = 16,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.max_token_size = max_token_size
        self.stride = stride
        self.batch_size = batch_size

        # ONNX 모델 경로 결정
        if not onnx_model_path:
            onnx_model_path = os.path.join(model_dir, "model.onnx")

        # ONNX Runtime 세션 생성 (CPU 최적화)
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # CPU 스레드 수 자동 설정
        sess_options.intra_op_num_threads = 0
        sess_options.inter_op_num_threads = 0

        providers = ["CPUExecutionProvider"]
        # CUDA 사용 가능한 경우 GPU 프로바이더 우선
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.session = ort.InferenceSession(
            onnx_model_path, sess_options=sess_options, providers=providers,
        )
        logger.info(
            "ONNX CodeT5p-220m 모델 로드 완료 (path=%s, providers=%s, max_token=%d, stride=%d, batch_size=%d)",
            onnx_model_path, self.session.get_providers(), max_token_size, stride, batch_size,
        )

    def _embed_sync(self, code_snippet: str) -> list[float]:
        """동기 임베딩 생성 (단일 스니펫)."""
        inputs = self.tokenizer(
            code_snippet,
            return_tensors="np",
            max_length=self.max_token_size,
            truncation=True,
            padding="longest",
        )

        ort_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }
        outputs = self.session.run(None, ort_inputs)

        # mean pooling: attention_mask를 고려하여 평균
        hidden_states = outputs[0]  # (1, seq_len, 768)
        attention_mask = inputs["attention_mask"][..., np.newaxis].astype(np.float32)
        masked_hidden = hidden_states * attention_mask
        emb = masked_hidden.sum(axis=1) / attention_mask.sum(axis=1)

        return emb.squeeze(0).tolist()

    async def get_embedding(self, code_snippet: str) -> list[float]:
        return await asyncio.to_thread(self._embed_sync, code_snippet)

    def _embed_batch_sync(self, code_snippets: list[str]) -> list[list[float]]:
        """배치 임베딩 생성."""
        if not code_snippets:
            return []

        all_embeddings: list[list[float]] = []
        for i in range(0, len(code_snippets), self.batch_size):
            batch = code_snippets[i : i + self.batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="np",
                max_length=self.max_token_size,
                truncation=True,
                padding="longest",
            )

            ort_inputs = {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64),
            }
            outputs = self.session.run(None, ort_inputs)

            # 배치 mean pooling: (B, seq_len, 768) → (B, 768)
            hidden_states = outputs[0]
            attention_mask = inputs["attention_mask"][..., np.newaxis].astype(np.float32)
            masked_hidden = hidden_states * attention_mask
            embs = masked_hidden.sum(axis=1) / attention_mask.sum(axis=1)

            all_embeddings.extend(embs.tolist())
        return all_embeddings

    async def get_embeddings_batch(self, code_snippets: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._embed_batch_sync, code_snippets)

    def _slice_sync(self, code: str) -> list[str]:
        """동기 코드 슬라이싱 — sliding window 방식."""
        token_ids = self.tokenizer.encode(
            code, truncation=False, add_special_tokens=False
        )

        if len(token_ids) <= self.max_token_size:
            return [code]

        snippets = []
        for start in range(0, len(token_ids), self.stride):
            end = min(start + self.max_token_size, len(token_ids))
            snippet_tokens = token_ids[start:end]
            snippets.append(snippet_tokens)
            if end >= len(token_ids):
                break

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
