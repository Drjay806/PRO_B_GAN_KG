from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np
import torch


class FaissRetriever:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.index = None

    def build(self, embeddings: np.ndarray) -> None:
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)

    def search(self, queries: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
        scores, ids = self.index.search(queries, topk)
        return scores, ids

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))

    @staticmethod
    def load(path: Path, dim: int) -> "FaissRetriever":
        retriever = FaissRetriever(dim)
        retriever.index = faiss.read_index(str(path))
        return retriever


def filter_candidates(
    candidates: np.ndarray,
    true_tails: List[int],
    filtered: Optional[List[int]] = None,
) -> np.ndarray:
    filtered_set = set(filtered or [])
    out = []
    for row, true_t in zip(candidates, true_tails):
        row_filtered = [c for c in row if c not in filtered_set or c == true_t]
        out.append(row_filtered)
    return np.array(out, dtype=np.int64)
