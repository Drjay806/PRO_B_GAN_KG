from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from .attention import ContextAttention, batch_neighbors
from .data import NeighborCache
from .embeddings import distmult_score
from .gan import Discriminator, Generator
from .retrieval import FaissRetriever
from .rl_evidence import EvidencePolicy, run_evidence_rollout


@dataclass
class NodeExplanation:
    entity_id: int
    entity_name: str
    prediction_score: float
    distmult_score: float
    discriminator_score: float
    attention_weights: List[Tuple[int, str, float]]  # True explanation: what influenced the GAN
    evidence_path: Optional[List[Tuple[int, str, int, str, int, str]]]  # Post-hoc verification only
    node_degree: int
    generation_samples: List[float]
    rank: int


@dataclass
class ExplanationBundle:
    query_head: str
    query_relation: str
    candidates: List[NodeExplanation]
    query_context_summary: Dict[str, float]


def compute_node_degree(entity_id: int, neighbor_cache: NeighborCache) -> int:
    degree = 0
    for (h, r), tails in neighbor_cache.pairs.items():
        if h == entity_id:
            degree += len(tails)
        if entity_id in tails:
            degree += 1
    return degree


def explain_prediction(
    h_id: int,
    r_id: int,
    entity_emb: torch.Tensor,
    relation_emb: torch.Tensor,
    attention: ContextAttention,
    generator: Generator,
    discriminator: Discriminator,
    neighbor_cache: NeighborCache,
    retriever: FaissRetriever,
    id2entity: Dict[int, str],
    id2rel: Dict[int, str],
    topk: int = 10,
    num_samples: int = 10,
    evidence_policy: Optional[EvidencePolicy] = None,
    evidence_budget: int = 3,
) -> ExplanationBundle:
    device = entity_emb.device
    h = torch.tensor([h_id], device=device)
    r = relation_emb[torch.tensor([r_id], device=device)]

    neighbors = neighbor_cache.get(h_id, r_id)
    if neighbors:
        neighbor_emb = entity_emb[torch.tensor(neighbors, device=device)].unsqueeze(0)
        neighbor_mask = torch.ones(1, len(neighbors), dtype=torch.bool, device=device)
        context, alpha = attention(
            entity_emb[h], r, neighbor_emb, mask=neighbor_mask
        )
        attn_weights = [(n, id2entity.get(n, f"entity_{n}"), float(w)) for n, w in zip(neighbors, alpha.squeeze(0).tolist())]
    else:
        context = torch.zeros(1, entity_emb.shape[1], device=device)
        alpha = torch.zeros(1, 1, device=device)
        attn_weights = []

    all_candidates_scored = {}
    sample_diversity = {}

    for sample_idx in range(num_samples):
        noise = torch.randn(1, generator.noise_dim, device=device)
        t_hat = generator(entity_emb[h], r, context, noise)
        t_hat_norm = t_hat / (torch.norm(t_hat, dim=-1, keepdim=True) + 1e-9)

        scores, ids = retriever.search(t_hat_norm.detach().cpu().numpy(), topk * 2)
        
        for cand_id, score in zip(ids[0].tolist(), scores[0].tolist()):
            if cand_id not in all_candidates_scored:
                all_candidates_scored[cand_id] = []
                sample_diversity[cand_id] = []
            all_candidates_scored[cand_id].append(float(score))
            sample_diversity[cand_id].append(t_hat_norm.squeeze(0).detach().cpu().numpy())

    candidates_list = []
    for rank, (cand_id, scores) in enumerate(
        sorted(all_candidates_scored.items(), key=lambda x: -sum(x[1]) / len(x[1]))[:topk], start=1
    ):
        avg_score = sum(scores) / len(scores)
        
        distmult = distmult_score(
            entity_emb[h],
            r,
            entity_emb[torch.tensor([cand_id], device=device)]
        ).item()

        disc_score = discriminator(
            entity_emb[h],
            r,
            entity_emb[torch.tensor([cand_id], device=device)],
            context
        ).item()

        degree = compute_node_degree(cand_id, neighbor_cache)

        evidence_path = None
        if evidence_policy is not None:
            try:
                evidence_steps = run_evidence_rollout(
                    policy=evidence_policy,
                    entity_emb=entity_emb,
                    neighbors=neighbor_cache.pairs,
                    query=(h_id, r_id),
                    target_tail=cand_id,
                    budget=evidence_budget,
                )
                if evidence_steps:
                    evidence_path = [
                        (
                            e.head,
                            id2entity.get(e.head, f"entity_{e.head}"),
                            e.rel,
                            id2rel.get(e.rel, f"rel_{e.rel}"),
                            e.tail,
                            id2entity.get(e.tail, f"entity_{e.tail}"),
                        )
                        for e in evidence_steps
                    ]
            except Exception:
                evidence_path = None

        candidates_list.append(
            NodeExplanation(
                entity_id=cand_id,
                entity_name=id2entity.get(cand_id, f"entity_{cand_id}"),
                prediction_score=avg_score,
                distmult_score=distmult,
                discriminator_score=disc_score,
                attention_weights=attn_weights,
                evidence_path=evidence_path,
                node_degree=degree,
                generation_samples=scores,
                rank=rank,
            )
        )

    context_summary = {
        "num_neighbors": len(neighbors),
        "avg_attention": float(alpha.mean().item()) if len(neighbors) > 0 else 0.0,
        "max_attention": float(alpha.max().item()) if len(neighbors) > 0 else 0.0,
        "context_norm": float(torch.norm(context).item()),
    }

    return ExplanationBundle(
        query_head=id2entity.get(h_id, f"entity_{h_id}"),
        query_relation=id2rel.get(r_id, f"rel_{r_id}"),
        candidates=candidates_list,
        query_context_summary=context_summary,
    )


def format_explanation_text(explanation: ExplanationBundle) -> str:
    lines = []
    lines.append(f"Query: ({explanation.query_head}, {explanation.query_relation}, ?)")
    lines.append(f"\nContext: {explanation.query_context_summary['num_neighbors']} neighbors")
    lines.append("")
    
    for cand in explanation.candidates:
        lines.append(f"Rank {cand.rank}: {cand.entity_name}")
        lines.append(f"  Prediction score: {cand.prediction_score:.4f}")
        lines.append(f"  DistMult score: {cand.distmult_score:.4f}")
        confidence_pct = torch.sigmoid(torch.tensor(cand.discriminator_score)).item() * 100
        lines.append(f"  Confidence: {confidence_pct:.1f}%")
        lines.append(f"  Node degree: {cand.node_degree}")
        lines.append(f"  Generation diversity: {len(cand.generation_samples)} samples")
        
        if cand.attention_weights:
            lines.append(f"  Top attention neighbors:")
            for n_id, n_name, weight in sorted(cand.attention_weights, key=lambda x: -x[2])[:5]:
                lines.append(f"    - {n_name}: {weight:.4f}")
        
        if cand.evidence_path:
            lines.append(f"  Evidence path:")
            for h_id, h_name, r_id, r_name, t_id, t_name in cand.evidence_path:
                lines.append(f"    {h_name} --{r_name}-> {t_name}")
        else:
            lines.append(f"  Evidence path: None found (may be novel prediction)")
        
        lines.append("")
    
    return "\n".join(lines)
