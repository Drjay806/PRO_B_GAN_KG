import logging
from typing import List

import torch
from torch import nn

logger = logging.getLogger(__name__)


class TextEncoder(nn.Module):
    def __init__(self, model_name: str = "allenai/scibert_scivocab_uncased", output_dim: int = 256) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.tokenizer = None
        self.model = None
        self.hidden_dim = 768
        self.proj = None

        try:
            from transformers import AutoTokenizer, AutoModel

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.hidden_dim = self.model.config.hidden_size
            logger.info(f"Loaded {model_name} with hidden_dim={self.hidden_dim}")

            if output_dim != self.hidden_dim:
                self.proj = nn.Linear(self.hidden_dim, output_dim)
                logger.info(f"Added projection head: {self.hidden_dim} -> {output_dim}")
        except ImportError:
            logger.warning("transformers not installed, using dummy encoder")

    def forward(self, texts: List[str]) -> torch.Tensor:
        if self.model is None:
            logger.warning("Model not loaded, returning random embeddings")
            return torch.randn(len(texts), self.output_dim)

        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]

        if self.proj is not None:
            embeddings = self.proj(embeddings)

        return embeddings


def encode_texts(encoder: TextEncoder, texts: List[str], batch_size: int = 32) -> torch.Tensor:
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_emb = encoder(batch)
        all_embeddings.append(batch_emb)
    return torch.cat(all_embeddings, dim=0)
