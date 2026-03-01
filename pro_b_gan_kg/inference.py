from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .attention import ContextAttention
from .data import NeighborCache
from .gan import Generator
from .retrieval import FaissRetriever
from .rl_evidence import EvidencePolicy, run_evidence_rollout


def predict(
    h_id: int,
    r_id: int,
    entity_emb: torch.Tensor,
    relation_emb: torch.Tensor,
    attention: ContextAttention,
    generator: Generator,
    neighbor_cache: NeighborCache,
    retriever: FaissRetriever,
    topk: int = 10,
    num_samples: int = 5,
    evidence_policy: Optional[EvidencePolicy] = None,
    evidence_budget: int = 3,
) -> Dict[str, List[Tuple[int, float]]]:
    device = entity_emb.device

    h = torch.tensor([h_id], device=device)
    r = relation_emb[torch.tensor([r_id], device=device)]
    neighbors = neighbor_cache.get(h_id, r_id)
    if neighbors:
        neighbor_emb = entity_emb[torch.tensor(neighbors, device=device)].unsqueeze(0)
        context, alpha = attention(entity_emb[h], r, neighbor_emb)
        attn = list(zip(neighbors, alpha.squeeze(0).detach().cpu().tolist()))
    else:
        context = torch.zeros(1, entity_emb.shape[1], device=device)
        attn = []

    candidates = []
    for _ in range(num_samples):
        noise = torch.randn(1, generator.noise_dim, device=device)
        t_hat = generator(entity_emb[h], r, context, noise)
        scores, ids = retriever.search(t_hat.detach().cpu().numpy().astype(np.float32), topk)
        for cand, score in zip(ids[0].tolist(), scores[0].tolist()):
            candidates.append((cand, score))

    candidates.sort(key=lambda x: -x[1])
    output = {"candidates": candidates[:topk], "attention": attn}
    if evidence_policy is not None:
        evidence = run_evidence_rollout(
            policy=evidence_policy,
            entity_emb=entity_emb,
            neighbors=neighbor_cache.pairs,
            query=(h_id, r_id),
            target_tail=candidates[0][0] if candidates else -1,
            budget=evidence_budget,
        )
        output["evidence"] = [(e.head, e.rel, e.tail, e.reward) for e in evidence]
    return output


def load_inference_artifacts(
    checkpoint_dir: Path, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, FaissRetriever, NeighborCache]:
    entity_emb = torch.load(checkpoint_dir / "entity_emb_final.pt", map_location=device)
    retriever = FaissRetriever.load(checkpoint_dir / "faiss.index", entity_emb.shape[1])
    neighbor_cache = NeighborCache.load(checkpoint_dir / "neighbors_index.npy")
    
    from .utils import load_json
    rel2id = load_json(checkpoint_dir / "rel2id.json")
    relation_emb = torch.zeros(len(rel2id), entity_emb.shape[1], device=device)
    
    return entity_emb, relation_emb, retriever, neighbor_cache
