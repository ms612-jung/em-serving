import torch
from transformers import AutoTokenizer, T5EncoderModel


class CodeT5pBackend:
    """CodeT5p 인코더 + Mean Pooling 기반 embedding 백엔드."""

    def __init__(self, model_name: str = "Salesforce/codet5p-220m", device: str | None = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = T5EncoderModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """텍스트 리스트를 받아 mean pooling된 embedding 벡터를 반환한다."""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**encoded)

        embeddings = self._mean_pooling(outputs.last_hidden_state, encoded["attention_mask"])
        return embeddings.cpu().tolist()

    @staticmethod
    def _mean_pooling(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Attention mask를 반영한 mean pooling."""
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = (hidden_state * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / sum_mask
