import logging
from pathlib import Path
from typing import Dict, Optional

import torch

logger = logging.getLogger(__name__)


class SemanticEmbeddingCache:
    def __init__(self, embeddings_dir: Path) -> None:
        self.embeddings_dir = Path(embeddings_dir)
        self.cache: Dict[str, torch.Tensor] = {}
        self.metadata = {}

    def load_embeddings(self, entity_type: str) -> Optional[torch.Tensor]:
        if entity_type in self.cache:
            return self.cache[entity_type]

        embedding_path = self.embeddings_dir / f"{entity_type}_embeddings.pt"
        if embedding_path.exists():
            try:
                emb = torch.load(embedding_path, map_location="cpu")
                self.cache[entity_type] = emb
                logger.info(f"Loaded {entity_type} embeddings: {emb.shape}")
                return emb
            except Exception as e:
                logger.warning(f"Failed to load {entity_type} embeddings: {e}")
        else:
            logger.warning(f"Embeddings file not found: {embedding_path}")

        return None

    def get_entity_embedding(self, entity_type: str, entity_idx: int) -> Optional[torch.Tensor]:
        embeddings = self.load_embeddings(entity_type)
        if embeddings is None or entity_idx >= len(embeddings):
            return None
        return embeddings[entity_idx]

    def fill_entity_table(
        self,
        entity_type: str,
        entity_table: torch.Tensor,
        entity_id_mapping: Dict[int, int],
    ) -> torch.Tensor:
        embeddings = self.load_embeddings(entity_type)
        if embeddings is None:
            logger.warning(f"No precomputed embeddings for {entity_type}, keeping random init")
            return entity_table

        for entity_id, semantic_idx in entity_id_mapping.items():
            if semantic_idx < len(embeddings):
                entity_table[entity_id] = embeddings[semantic_idx]

        logger.info(f"Filled {len(entity_id_mapping)} entities of type {entity_type}")
        return entity_table
