import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .attention import ContextAttention, batch_neighbors
from .compgcn import CompGCN
from .config import RunConfig
from .data import NeighborCache, build_neighbor_cache, load_and_prepare
from .embeddings import EntityEmbedding, RelationEmbedding, distmult_score
from .fusion import FusionConcat, FusionGate
from .metrics import ranking_metrics
from .retrieval import FaissRetriever
from .rgcn import RGCN
from .sampler import NegativeSampler
from .utils import assert_finite, get_device, save_json, set_seed, setup_logging


class TripleDataset(Dataset):
    def __init__(self, triples: List[Tuple[int, int, int]]) -> None:
        self.triples = triples

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int) -> Tuple[int, int, int]:
        return self.triples[idx]


def build_edge_index(triples: List[Tuple[int, int, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    src = torch.tensor([h for h, _, _ in triples], dtype=torch.long)
    dst = torch.tensor([t for _, _, t in triples], dtype=torch.long)
    rel = torch.tensor([r for _, r, _ in triples], dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)
    return edge_index, rel


def pretrain_distmult(
    entity_emb: EntityEmbedding,
    relation_emb: RelationEmbedding,
    triples: List[Tuple[int, int, int]],
    num_entities: int,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    batch_size: int,
    logger,
) -> None:
    dataset = TripleDataset(triples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        losses = []
        for h, r, t in loader:
            h = h.to(device)
            r = r.to(device)
            t = t.to(device)

            neg = torch.randint(0, num_entities, t.shape, device=device)

            pos_score = distmult_score(entity_emb(h), relation_emb(r), entity_emb(t))
            neg_score = distmult_score(entity_emb(h), relation_emb(r), entity_emb(neg))

            loss = torch.relu(1.0 - pos_score + neg_score).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        logger.info("DistMult pretrain epoch %d loss %.4f", epoch + 1, float(np.mean(losses)))


def build_context(
    attention: ContextAttention,
    h: torch.Tensor,
    r_id: torch.Tensor,
    r_emb: torch.Tensor,
    neighbor_cache: NeighborCache,
    entity_emb: torch.Tensor,
    neighbor_dropout: float,
    leave_one_out: bool,
    true_t: torch.Tensor,
    max_neighbors: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # max_neighbors is now a parameter (config: training.max_neighbors)
    batch_pairs = []
    # Extract scalar values using numpy to handle autocast state issues
    h_ids = h.cpu().detach().numpy().astype(int)
    r_ids = r_id.cpu().detach().numpy().astype(int)
    t_ids = true_t.cpu().detach().numpy().astype(int)
    
    for h_id, r_id, t_id in zip(h_ids, r_ids, t_ids):
        # Ensure IDs are Python scalars
        h_id = int(h_id)
        r_id = int(r_id)
        t_id = int(t_id)
        neighbors = list(neighbor_cache.get(h_id, r_id))
        if leave_one_out and t_id in neighbors:
            neighbors = [n for n in neighbors if n != t_id]
        if neighbor_dropout > 0:
            keep = max(1, int(len(neighbors) * (1 - neighbor_dropout)))
            if neighbors:
                neighbors = list(np.random.choice(neighbors, size=keep, replace=False))
        if len(neighbors) > max_neighbors:
            neighbors = list(np.random.choice(neighbors, size=max_neighbors, replace=False))
        batch_pairs.append(neighbors)

    neighbor_emb, mask = batch_neighbors(batch_pairs, entity_emb, entity_emb.device)
    if mask.sum() == 0:
        context = torch.zeros(h.shape[0], entity_emb.shape[1], device=entity_emb.device)
        alpha = torch.zeros(h.shape[0], neighbor_emb.shape[1], device=entity_emb.device)
        return context, alpha

    context, alpha = attention(
        h_emb=entity_emb[h],
        r_emb=r_emb,
        neighbor_emb=neighbor_emb,
        mask=mask,
    )
    empty = mask.sum(dim=1) == 0
    if empty.any():
        # Use torch.where instead of in-place indexing to avoid gradient version mismatch
        context = torch.where(empty.unsqueeze(-1), torch.zeros_like(context), context)
        alpha = torch.where(empty.unsqueeze(-1), torch.zeros_like(alpha), alpha)
    return context, alpha


def _initialize_from_semantic_files(
    logger,
    entity_weight: torch.Tensor,
    entity2id: Dict[str, int],
    embeddings_dir: Path,
) -> int:
    total_initialized = 0
    embedding_dim = entity_weight.shape[1]
    semantic_files = sorted(embeddings_dir.glob("*_embeddings*.pt"))

    if not semantic_files:
        logger.warning(f"No semantic embedding files found in {embeddings_dir}")
        return 0

    for emb_path in semantic_files:
        try:
            payload = torch.load(emb_path, map_location="cpu")
        except Exception as exc:
            logger.warning(f"Failed to load semantic file {emb_path.name}: {exc}")
            continue

        if not isinstance(payload, dict):
            logger.warning(
                f"Skipping {emb_path.name}: expected dict[entity_id, vector], got {type(payload).__name__}"
            )
            continue

        initialized_this_file = 0
        skipped_dim = 0

        for entity_name, vector in payload.items():
            target_idx = entity2id.get(entity_name)
            if target_idx is None:
                continue

            vector_tensor = torch.as_tensor(vector, dtype=entity_weight.dtype)
            if vector_tensor.ndim != 1:
                skipped_dim += 1
                continue
            if vector_tensor.shape[0] != embedding_dim:
                skipped_dim += 1
                continue

            entity_weight[target_idx] = vector_tensor
            initialized_this_file += 1

        total_initialized += initialized_this_file
        logger.info(
            "Initialized %d entities from %s%s",
            initialized_this_file,
            emb_path.name,
            f" (skipped_dim={skipped_dim})" if skipped_dim else "",
        )

    return total_initialized


def run_training(config: Dict, output_dir: Path) -> None:
    run_cfg = RunConfig.from_dict(config)
    logger = setup_logging(output_dir)
    set_seed(run_cfg.training.seed)

    logger.info("Starting data loading and preparation...")
    mappings, id_triples = load_and_prepare(
        train_path=Path(run_cfg.data.train_path),
        val_path=Path(run_cfg.data.val_path),
        test_path=Path(run_cfg.data.test_path),
        delimiter=run_cfg.data.delimiter,
        has_header=run_cfg.data.has_header,
        output_dir=output_dir,
    )
    logger.info(f"Data loading complete: {len(mappings.entity2id)} entities, {len(mappings.rel2id)} relations")

    num_entities = len(mappings.entity2id)
    num_relations = len(mappings.rel2id)
    device = get_device()

    entity_emb = EntityEmbedding(num_entities, run_cfg.model.embedding_dim).to(device)
    relation_emb = RelationEmbedding(num_relations, run_cfg.model.embedding_dim).to(device)

    if run_cfg.semantic and run_cfg.semantic.embeddings_dir:
        embeddings_dir = Path(run_cfg.semantic.embeddings_dir)
        logger.info(f"Loading semantic embeddings from {embeddings_dir}")
        with torch.no_grad():
            initialized = _initialize_from_semantic_files(
                logger=logger,
                entity_weight=entity_emb.embedding.weight,
                entity2id=mappings.entity2id,
                embeddings_dir=embeddings_dir,
            )
        logger.info(f"Semantic initialization complete: {initialized} entities initialized")
    else:
        logger.info("No semantic embeddings configured, using random initialization")

    optimizer = torch.optim.Adam(
        list(entity_emb.parameters()) + list(relation_emb.parameters()),
        lr=run_cfg.training.lr,
        weight_decay=run_cfg.training.weight_decay,
    )

    if run_cfg.training.max_epochs_pretrain > 0:
        logger.info(f"Starting pretrain phase ({run_cfg.training.max_epochs_pretrain} epochs)...")
        pretrain_distmult(
            entity_emb=entity_emb,
            relation_emb=relation_emb,
            triples=id_triples.train,
            num_entities=num_entities,
            optimizer=optimizer,
            device=device,
            epochs=run_cfg.training.max_epochs_pretrain,
            batch_size=run_cfg.training.batch_size,
            logger=logger,
        )
        logger.info("Pretrain phase complete")

    logger.info("Building edge index and neighbor cache...")

    edge_index, edge_type = build_edge_index(id_triples.train)
    
    # Sample edges to reduce memory for CompGCN (controlled by config)
    num_edges = edge_index.shape[1]
    sample_ratio = run_cfg.training.compgcn_edge_sample_ratio
    num_sample = max(int(num_edges * sample_ratio), 10000)
    perm = torch.randperm(num_edges)[:num_sample]
    edge_index_sample = edge_index[:, perm]
    edge_type_sample = edge_type[perm]
    
    edge_index = edge_index_sample.to(device)
    edge_type = edge_type_sample.to(device)
    logger.info(f"Edge index sampled: {edge_index.shape} (from {num_edges} edges)")

    logger.info("Initializing encoder and GAN models...")

    if run_cfg.model.use_rgcn:
        encoder = RGCN(
            dim=run_cfg.model.embedding_dim,
            num_relations=num_relations,
            layers=run_cfg.model.compgcn_layers,
            dropout=run_cfg.model.dropout,
        ).to(device)
    else:
        encoder = CompGCN(
            dim=run_cfg.model.embedding_dim,
            layers=run_cfg.model.compgcn_layers,
            dropout=run_cfg.model.dropout,
            op=run_cfg.model.comp_op,
        ).to(device)

    if run_cfg.model.fusion == "concat":
        fusion = FusionConcat(run_cfg.model.embedding_dim).to(device)
    else:
        fusion = FusionGate(run_cfg.model.embedding_dim).to(device)

    attention = ContextAttention(
        dim=run_cfg.model.embedding_dim,
        hidden=run_cfg.model.attention_hidden,
        dropout=run_cfg.model.dropout,
    ).to(device)

    from .gan import Generator, Discriminator

    generator = Generator(
        dim=run_cfg.model.embedding_dim,
        hidden=run_cfg.model.generator_hidden,
        noise_dim=run_cfg.model.noise_dim,
    ).to(device)
    discriminator = Discriminator(
        dim=run_cfg.model.embedding_dim,
        hidden=run_cfg.model.discriminator_hidden,
    ).to(device)

    all_params = (
        list(entity_emb.parameters())
        + list(relation_emb.parameters())
        + list(fusion.parameters())
        + list(attention.parameters())
        + list(generator.parameters())
        + list(discriminator.parameters())
    )
    # Encoder has its own optimizer — updated only during periodic refresh passes,
    # not per-batch (its output is cached/detached so main optimizer grads never reach it)
    optimizer = torch.optim.Adam(all_params, lr=run_cfg.training.lr, weight_decay=run_cfg.training.weight_decay)
    encoder_optimizer = torch.optim.Adam(
        list(encoder.parameters()),
        lr=run_cfg.training.lr,
        weight_decay=run_cfg.training.weight_decay,
    )

    neighbor_cache = build_neighbor_cache(id_triples.train)
    logger.info(f"Neighbor cache built: {len(neighbor_cache.pairs)} (h, r) pairs")
    neighbor_cache.save(output_dir / "neighbors_index.npy")

    logger.info("Initializing negative sampler...")
    sampler = NegativeSampler(
        num_entities=num_entities,
        train_pairs=neighbor_cache.pairs,
        easy_ratio=run_cfg.sampling.easy_ratio,
        medium_ratio=run_cfg.sampling.medium_ratio,
        hard_ratio=run_cfg.sampling.hard_ratio,
    )
    logger.info("Negative sampler ready")

    # Build filtered evaluation lookup — maps (h, r) → set of ALL known true tails
    # across train + val + test. When ranking, known true tails other than the current
    # query's target are removed so the model isn't penalized for predicting valid answers.
    # This is the standard filtered protocol used in all KG benchmark papers.
    logger.info("Building filtered evaluation lookup...")
    known_tails: Dict[Tuple[int, int], set] = {}
    for h_id, r_id, t_id in id_triples.train + id_triples.val + id_triples.test:
        key = (h_id, r_id)
        if key not in known_tails:
            known_tails[key] = set()
        known_tails[key].add(t_id)
    logger.info("Filtered lookup built: %d (h,r) pairs", len(known_tails))

    def evaluate(triples: List[Tuple[int, int, int]]) -> Dict[str, float]:
        # Cap evaluation size for speed (controlled by max_eval_samples config)
        if run_cfg.training.max_eval_samples > 0 and len(triples) > run_cfg.training.max_eval_samples:
            import random
            triples = random.sample(triples, run_cfg.training.max_eval_samples)
        logger.info("  Evaluating on %d triples...", len(triples))
        entity_emb.eval()
        relation_emb.eval()
        encoder.eval()
        fusion.eval()
        attention.eval()
        generator.eval()
        entity_sem = entity_emb.embedding.weight
        with torch.no_grad():
            if run_cfg.model.use_rgcn:
                entity_struct = encoder(entity_sem, edge_index, edge_type)
            else:
                entity_struct = encoder(entity_sem, relation_emb.embedding.weight, edge_index, edge_type)

            entity_final = fusion(entity_sem, entity_struct)
            assert_finite(entity_final, "entity_final")

            retriever = FaissRetriever(run_cfg.model.embedding_dim)
            retriever.build(entity_final.detach().cpu().numpy().astype(np.float32))

            ranks = []
            for h_id, r_id, t_id in triples:
                h = torch.tensor([h_id], device=device)
                r_ids = torch.tensor([r_id], device=device)
                r = relation_emb(r_ids)
                context, _ = build_context(
                    attention=attention,
                    h=h,
                    r_id=r_ids,
                    r_emb=r,
                    neighbor_cache=neighbor_cache,
                    entity_emb=entity_final,
                    neighbor_dropout=0.0,
                    leave_one_out=False,
                    true_t=torch.tensor([t_id], device=device),
                    max_neighbors=run_cfg.training.max_neighbors,
                )
                noise = torch.randn(1, run_cfg.model.noise_dim, device=device)
                t_hat = generator(entity_final[h], r, context, noise)
                t_hat = t_hat / (torch.norm(t_hat, dim=-1, keepdim=True) + 1e-9)

                scores, ids = retriever.search(
                    t_hat.detach().cpu().numpy().astype(np.float32),
                    # Retrieve extra candidates to account for filtered-out known tails
                    min(run_cfg.training.eval_topk + len(known_tails.get((h_id, r_id), set())), num_entities),
                )
                ids = ids[0].tolist()

                # Filtered ranking: remove all known true tails except the current target
                other_known = known_tails.get((h_id, r_id), set()) - {t_id}
                ids_filtered = [i for i in ids if i not in other_known]

                if t_id in ids_filtered:
                    rank = ids_filtered.index(t_id) + 1
                else:
                    rank = run_cfg.training.eval_topk + 1
                ranks.append(rank)
            metrics = ranking_metrics(ranks)
            entity_emb.train()
            relation_emb.train()
            encoder.train()
            fusion.train()
            attention.train()
            generator.train()
            return metrics

    logger.info("Preparing training dataset...")
    dataset = TripleDataset(id_triples.train)
    logger.info(f"Dataset created with {len(dataset)} triples")

    loader = DataLoader(dataset, batch_size=run_cfg.training.batch_size, shuffle=True)
    scaler = torch.cuda.amp.GradScaler()
    max_steps = run_cfg.training.max_steps_per_epoch

    # ── Resume logic ──────────────────────────────────────────────────────────
    # training_state.json is written after every epoch so a Colab disconnect
    # can be recovered by simply re-running the same command.
    state_path = output_dir / "training_state.json"
    resume_phase = "warmup"   # which phase to start from
    resume_epoch = 0          # which epoch within that phase to start from
    best_mrr = -1.0

    if run_cfg.training.resume and state_path.exists():
        saved = json.loads(state_path.read_text())
        resume_phase = saved["phase"]
        resume_epoch = saved["epoch"] + 1   # next epoch after the last completed one
        best_mrr = saved["best_mrr"]
        ckpt_path = output_dir / ("best_model.pt" if resume_phase == "gan" else "warmup_best.pt")
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            entity_emb.load_state_dict(ckpt["entity_emb"])
            relation_emb.load_state_dict(ckpt["relation_emb"])
            encoder.load_state_dict(ckpt["encoder"])
            fusion.load_state_dict(ckpt["fusion"])
            attention.load_state_dict(ckpt["attention"])
            generator.load_state_dict(ckpt["generator"])
            discriminator.load_state_dict(ckpt["discriminator"])
            logger.info(
                "Resumed from %s — phase=%s epoch=%d best_mrr=%.4f",
                ckpt_path.name, resume_phase, resume_epoch, best_mrr,
            )
        else:
            logger.warning("training_state.json found but checkpoint file missing — starting fresh")
            resume_phase = "warmup"
            resume_epoch = 0
            best_mrr = -1.0
    else:
        logger.info("Starting warm-up")
    # ─────────────────────────────────────────────────────────────────────────

    best_state = None

    # CSV file that logs every epoch's metrics — survives Colab disconnects and enables plotting
    metrics_csv = output_dir / "metrics_log.csv"
    if not metrics_csv.exists():
        metrics_csv.write_text("phase,epoch,loss,mrr,hits@1,hits@3,hits@10\n")

    def _log_metrics_csv(phase: str, epoch: int, loss: float, m: Dict) -> None:
        with metrics_csv.open("a") as f:
            f.write(f"{phase},{epoch},{loss:.6f},{m['mrr']:.6f},{m['hits@1']:.6f},{m['hits@3']:.6f},{m['hits@10']:.6f}\n")

    def _save_training_state(phase: str, epoch: int, mrr: float) -> None:
        """Persist current training position so a restart can resume here."""
        state_path.write_text(json.dumps({"phase": phase, "epoch": epoch, "best_mrr": mrr}))

    def _refresh_encoder(epoch: int) -> torch.Tensor:
        """Compute and cache CompGCN structural embeddings for one epoch.

        On epochs divisible by encoder_refresh_interval: run CompGCN in train mode
        on a sampled edge subset, compute a link-prediction loss, update encoder weights
        via encoder_optimizer, then recompute the full cache for the batch loop.

        All other epochs: eval/no_grad pass on the full (sampled-at-setup) edge set.
        Returns a detached tensor of shape [num_entities, embedding_dim].
        """
        interval = run_cfg.training.encoder_refresh_interval
        is_refresh = (interval > 0) and (epoch % interval == 0)

        if is_refresh:
            n = edge_index.shape[1]
            k = max(int(n * run_cfg.training.encoder_refresh_sample_ratio), 10000)
            perm = torch.randperm(n, device=device)[:k]
            ei_s = edge_index[:, perm]
            et_s = edge_type[perm]

            encoder.train()
            encoder_optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                e_sem = entity_emb.embedding.weight.detach()
                r_w = relation_emb.embedding.weight.detach()
                if run_cfg.model.use_rgcn:
                    e_struct = encoder(e_sem, ei_s, et_s)
                else:
                    e_struct = encoder(e_sem, r_w, ei_s, et_s)
                e_final = fusion(e_sem, e_struct)
                # Link-prediction margin loss on a fresh mini-batch of triples
                ref_batch = next(iter(DataLoader(dataset, batch_size=min(2048, len(dataset)), shuffle=True)))
                ref_h, ref_r, ref_t = [x.to(device) for x in ref_batch]
                ref_neg = sampler.sample(
                    list(zip(
                        ref_h.cpu().numpy().astype(int),
                        ref_r.cpu().numpy().astype(int),
                        ref_t.cpu().numpy().astype(int),
                    )),
                    num_negatives=5,
                ).to(device)
                pos = torch.sum(e_final[ref_h] * e_final[ref_t], dim=-1)
                neg = torch.sum(e_final[ref_h].unsqueeze(1) * e_final[ref_neg], dim=-1).mean(dim=1)
                enc_loss = torch.relu(1.0 - pos + neg).mean()
            scaler.scale(enc_loss).backward()
            scaler.unscale_(encoder_optimizer)
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()), run_cfg.training.grad_clip)
            scaler.step(encoder_optimizer)
            scaler.update()
            logger.info(
                "Epoch %d: encoder refresh (train mode, %.0f%% edges, enc_loss=%.4f)",
                epoch + 1, run_cfg.training.encoder_refresh_sample_ratio * 100, enc_loss.item(),
            )

        # Always rebuild the full epoch cache (eval/no_grad) after possible weight update
        encoder.eval()
        with torch.no_grad():
            e_sem_full = entity_emb.embedding.weight.detach()
            r_w_full = relation_emb.embedding.weight.detach()
            if run_cfg.model.use_rgcn:
                cached = encoder(e_sem_full, edge_index, edge_type).detach()
            else:
                cached = encoder(e_sem_full, r_w_full, edge_index, edge_type).detach()
        encoder.train()
        if not is_refresh:
            logger.info("Epoch %d: encoder cache rebuilt (eval mode, no grad)", epoch + 1)
        return cached

    logger.info(f"Starting warmup phase ({run_cfg.training.max_epochs_warmup} epochs)...")
    if resume_phase == "warmup":
        warmup_start = resume_epoch
        if warmup_start > 0:
            logger.info("Resuming warmup from epoch %d", warmup_start + 1)
    else:
        warmup_start = run_cfg.training.max_epochs_warmup  # skip entirely
        logger.info("Warmup already complete, skipping to GAN phase")

    for epoch in range(warmup_start, run_cfg.training.max_epochs_warmup):
        entity_struct_cached = _refresh_encoder(epoch)
        logger.info("Warmup epoch %d: GNN pass complete, starting batch loop", epoch + 1)

        losses = []
        for step, batch in enumerate(loader):
            if max_steps > 0 and step >= max_steps:
                break
            h, r, t = [x.to(device) for x in batch]

            # Pre-compute neighbor IDs on CPU (numpy ops can't run inside autocast)
            h_ids = h.cpu().detach().numpy().astype(int)
            r_ids = r.cpu().detach().numpy().astype(int)
            t_ids = t.cpu().detach().numpy().astype(int)
            negatives = sampler.sample(list(zip(h_ids, r_ids, t_ids)), num_negatives=10).to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                entity_sem = entity_emb.embedding.weight
                entity_final = fusion(entity_sem, entity_struct_cached)

                context, _ = build_context(
                    attention=attention,
                    h=h,
                    r_id=r,
                    r_emb=relation_emb(r),
                    neighbor_cache=neighbor_cache,
                    entity_emb=entity_final,
                    neighbor_dropout=run_cfg.training.neighbor_dropout,
                    leave_one_out=run_cfg.training.leave_one_out,
                    true_t=t,
                    max_neighbors=run_cfg.training.max_neighbors,
                )

                noise = torch.randn(h.shape[0], run_cfg.model.noise_dim, device=device)
                t_hat = generator(entity_final[h], relation_emb(r), context, noise)

                pos_score = torch.sum(t_hat * entity_final[t], dim=-1)
                neg_score = torch.sum(t_hat.unsqueeze(1) * entity_final[negatives], dim=-1).mean(dim=1)
                loss = torch.relu(1.0 - pos_score + neg_score).mean()

            torch.cuda.empty_cache()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(all_params, run_cfg.training.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
            if (step + 1) % 50 == 0:
                logger.info(
                    "  Warmup epoch %d  step %d/%s  loss %.4f",
                    epoch + 1, step + 1,
                    str(max_steps) if max_steps > 0 else "?",
                    float(np.mean(losses[-50:])),
                )

        metrics = evaluate(id_triples.val)
        logger.info(
            "[Warmup %d/%d] loss=%.4f | val MRR=%.4f  hits@1=%.4f  hits@3=%.4f  hits@10=%.4f",
            epoch + 1, run_cfg.training.max_epochs_warmup,
            float(np.mean(losses)),
            metrics["mrr"], metrics["hits@1"], metrics["hits@3"], metrics["hits@10"],
        )
        _save_training_state("warmup", epoch, metrics["mrr"])
        _log_metrics_csv("warmup", epoch + 1, float(np.mean(losses)), metrics)
        if metrics["mrr"] > best_mrr:
            best_mrr = metrics["mrr"]
            best_state = {
                "entity_emb": entity_emb.state_dict(),
                "relation_emb": relation_emb.state_dict(),
                "encoder": encoder.state_dict(),
                "fusion": fusion.state_dict(),
                "attention": attention.state_dict(),
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
            }

    if best_state:
        torch.save(best_state, output_dir / "warmup_best.pt")
        # Restore best warmup weights before GAN phase starts —
        # GAN should begin from the best learned representations, not last epoch
        entity_emb.load_state_dict(best_state["entity_emb"])
        relation_emb.load_state_dict(best_state["relation_emb"])
        encoder.load_state_dict(best_state["encoder"])
        fusion.load_state_dict(best_state["fusion"])
        attention.load_state_dict(best_state["attention"])
        generator.load_state_dict(best_state["generator"])
        discriminator.load_state_dict(best_state["discriminator"])
        logger.info("Restored best warmup checkpoint (mrr %.4f) before GAN phase", best_mrr)

    best_mrr = -1.0  # reset for GAN phase
    best_state = None
    logger.info("Starting adversarial training")
    gan_start = resume_epoch if resume_phase == "gan" else 0
    if gan_start > 0:
        logger.info("Resuming GAN from epoch %d", gan_start + 1)
    for epoch in range(gan_start, run_cfg.training.max_epochs_gan):
        entity_struct_cached = _refresh_encoder(epoch)
        logger.info("GAN epoch %d: GNN pass complete, starting batch loop", epoch + 1)

        losses = []
        for step, batch in enumerate(loader):
            if max_steps > 0 and step >= max_steps:
                break
            h, r, t = [x.to(device) for x in batch]

            # Pre-compute CPU/numpy operations outside autocast
            h_ids = h.cpu().detach().numpy().astype(int)
            r_ids = r.cpu().detach().numpy().astype(int)
            t_ids = t.cpu().detach().numpy().astype(int)
            negatives = sampler.sample(list(zip(h_ids, r_ids, t_ids)), num_negatives=10).to(device)

            # --- Discriminator update (gan_k steps) ---
            for _ in range(run_cfg.training.gan_k):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    entity_sem = entity_emb.embedding.weight
                    entity_final = fusion(entity_sem, entity_struct_cached)
                    context, _ = build_context(
                        attention=attention, h=h, r_id=r, r_emb=relation_emb(r),
                        neighbor_cache=neighbor_cache, entity_emb=entity_final,
                        neighbor_dropout=run_cfg.training.neighbor_dropout,
                        leave_one_out=run_cfg.training.leave_one_out, true_t=t,
                        max_neighbors=run_cfg.training.max_neighbors,
                    )
                    noise = torch.randn(h.shape[0], run_cfg.model.noise_dim, device=device)
                    t_hat = generator(entity_final[h], relation_emb(r), context, noise)
                    real_logits = discriminator(entity_final[h], relation_emb(r), entity_final[t], context)
                    fake_logits = discriminator(entity_final[h], relation_emb(r), t_hat.detach(), context)
                    d_loss = (
                        torch.nn.functional.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits) * 0.9)
                        + torch.nn.functional.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
                    )
                scaler.scale(d_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(all_params, run_cfg.training.grad_clip)
                scaler.step(optimizer)
                scaler.update()

            # --- Generator update ---
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                entity_sem = entity_emb.embedding.weight
                entity_final = fusion(entity_sem, entity_struct_cached)
                context, _ = build_context(
                    attention=attention, h=h, r_id=r, r_emb=relation_emb(r),
                    neighbor_cache=neighbor_cache, entity_emb=entity_final,
                    neighbor_dropout=run_cfg.training.neighbor_dropout,
                    leave_one_out=run_cfg.training.leave_one_out, true_t=t,
                    max_neighbors=run_cfg.training.max_neighbors,
                )
                noise = torch.randn(h.shape[0], run_cfg.model.noise_dim, device=device)
                t_hat = generator(entity_final[h], relation_emb(r), context, noise)
                fake_logits = discriminator(entity_final[h], relation_emb(r), t_hat, context)
                adv_loss = torch.nn.functional.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
                pos_score = torch.sum(t_hat * entity_final[t], dim=-1)
                neg_score = torch.sum(t_hat.unsqueeze(1) * entity_final[negatives], dim=-1).mean(dim=1)
                retrieval_loss = torch.relu(1.0 - pos_score + neg_score).mean()
                loss = retrieval_loss + run_cfg.training.lambda_adv * adv_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(all_params, run_cfg.training.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
            if (step + 1) % 50 == 0:
                logger.info(
                    "  GAN epoch %d  step %d/%s  loss %.4f",
                    epoch + 1, step + 1,
                    str(max_steps) if max_steps > 0 else "?",
                    float(np.mean(losses[-50:])),
                )

        metrics = evaluate(id_triples.val)
        logger.info(
            "[GAN %d/%d] loss=%.4f | val MRR=%.4f  hits@1=%.4f  hits@3=%.4f  hits@10=%.4f",
            epoch + 1, run_cfg.training.max_epochs_gan,
            float(np.mean(losses)),
            metrics["mrr"], metrics["hits@1"], metrics["hits@3"], metrics["hits@10"],
        )
        _save_training_state("gan", epoch, metrics["mrr"])
        _log_metrics_csv("gan", epoch + 1, float(np.mean(losses)), metrics)
        if metrics["mrr"] > best_mrr:
            best_mrr = metrics["mrr"]
            best_state = {
                "entity_emb": entity_emb.state_dict(),
                "relation_emb": relation_emb.state_dict(),
                "encoder": encoder.state_dict(),
                "fusion": fusion.state_dict(),
                "attention": attention.state_dict(),
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
            }
        else:
            run_cfg.training.lambda_adv = min(
                run_cfg.training.lambda_adv + run_cfg.training.lambda_adv_step,
                run_cfg.training.lambda_adv_max,
            )

    if best_state:
        torch.save(best_state, output_dir / "best_model.pt")
        # Restore best weights so FAISS index and final embeddings reflect peak performance
        entity_emb.load_state_dict(best_state["entity_emb"])
        relation_emb.load_state_dict(best_state["relation_emb"])
        encoder.load_state_dict(best_state["encoder"])
        fusion.load_state_dict(best_state["fusion"])
        logger.info("Restored best GAN checkpoint (mrr %.4f) for final artifact generation", best_mrr)

    logger.info("Building and saving FAISS index")
    entity_sem = entity_emb.embedding.weight
    with torch.no_grad():
        if run_cfg.model.use_rgcn:
            entity_struct = encoder(entity_sem, edge_index, edge_type)
        else:
            entity_struct = encoder(entity_sem, relation_emb.embedding.weight, edge_index, edge_type)
        entity_final = fusion(entity_sem, entity_struct)
    
    retriever = FaissRetriever(run_cfg.model.embedding_dim)
    retriever.build(entity_final.detach().cpu().numpy().astype(np.float32))
    retriever.save(output_dir / "faiss.index")
    
    torch.save(entity_final, output_dir / "entity_emb_final.pt")
    logger.info("Saved FAISS index and final embeddings")

    final_metrics = evaluate(id_triples.test)
    save_json({"metrics": final_metrics, "config": asdict(run_cfg)}, output_dir / "metrics.json")
    logger.info("Final test metrics: %s", final_metrics)
