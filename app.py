import json
import os
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="PRO-B GAN KG", page_icon="🧬", layout="wide")

ARTIFACT_DIR = Path("artifacts")
ARTIFACT_REPO_ID = os.getenv("ARTIFACT_REPO_ID", "drjay806/PRO-B-GAN-KG-artifacts")
REQUIRED_ARTIFACTS = [
    "best_model.pt",
    "entity_emb_final.pt",
    "faiss.index",
    "entity2id.json",
    "rel2id.json",
    "neighbors_index.npy",
    "metrics.json",
]


@st.cache_resource
def ensure_artifacts() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    missing = [name for name in REQUIRED_ARTIFACTS if not (ARTIFACT_DIR / name).exists()]
    if not missing:
        return

    with st.spinner("Downloading model artifacts from Hugging Face (first boot may take a few minutes)..."):
        snapshot_download(
            repo_id=ARTIFACT_REPO_ID,
            repo_type="model",
            local_dir=ARTIFACT_DIR,
            local_dir_use_symlinks=False,
            allow_patterns=REQUIRED_ARTIFACTS + ["rl_policy.pt"],
        )

    still_missing = [name for name in REQUIRED_ARTIFACTS if not (ARTIFACT_DIR / name).exists()]
    if still_missing:
        raise FileNotFoundError(
            f"Missing required artifacts after download: {still_missing}. "
            f"Check model repo {ARTIFACT_REPO_ID}."
        )


@st.cache_resource
def load_model():
    from pro_b_gan_kg.attention import ContextAttention
    from pro_b_gan_kg.data import NeighborCache
    from pro_b_gan_kg.fusion import FusionConcat
    from pro_b_gan_kg.gan import Discriminator, Generator
    from pro_b_gan_kg.retrieval import FaissRetriever
    from pro_b_gan_kg.rl_evidence import EvidencePolicy

    device = torch.device("cpu")

    config = json.loads((ARTIFACT_DIR / "metrics.json").read_text())["config"]
    dim = config["model"]["embedding_dim"]
    attn_hidden = config["model"]["attention_hidden"]
    gen_hidden = config["model"]["generator_hidden"]
    disc_hidden = config["model"]["discriminator_hidden"]
    noise_dim = config["model"]["noise_dim"]
    dropout = config["model"]["dropout"]

    entity_emb = torch.load(ARTIFACT_DIR / "entity_emb_final.pt", map_location=device)
    checkpoint = torch.load(ARTIFACT_DIR / "best_model.pt", map_location=device)

    entity2id = json.loads((ARTIFACT_DIR / "entity2id.json").read_text())
    rel2id = json.loads((ARTIFACT_DIR / "rel2id.json").read_text())
    id2entity = {v: k for k, v in entity2id.items()}
    id2rel = {v: k for k, v in rel2id.items()}

    neighbor_cache = NeighborCache.load(ARTIFACT_DIR / "neighbors_index.npy")

    from pro_b_gan_kg.embeddings import RelationEmbedding
    relation_emb_mod = RelationEmbedding(len(rel2id), dim).to(device)
    relation_emb_mod.load_state_dict(checkpoint["relation_emb"])
    relation_emb_mod.eval()

    attention = ContextAttention(dim=dim, hidden=attn_hidden, dropout=dropout).to(device)
    attention.load_state_dict(checkpoint["attention"])
    attention.eval()

    generator = Generator(dim=dim, hidden=gen_hidden, noise_dim=noise_dim).to(device)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()

    discriminator = Discriminator(dim=dim, hidden=disc_hidden).to(device)
    discriminator.load_state_dict(checkpoint["discriminator"])
    discriminator.eval()

    fusion = FusionConcat(dim).to(device)
    fusion.load_state_dict(checkpoint["fusion"])
    fusion.eval()

    retriever = FaissRetriever.load(ARTIFACT_DIR / "faiss.index", dim)

    rl_policy = None
    rl_path = ARTIFACT_DIR / "rl_policy.pt"
    if rl_path.exists():
        rl_cfg = config.get("rl", {})
        policy_hidden = rl_cfg.get("policy_hidden", 0) or dim
        rl_policy = EvidencePolicy(dim=dim, hidden=policy_hidden).to(device)
        rl_policy.load_state_dict(torch.load(rl_path, map_location=device))
        rl_policy.eval()

    return {
        "entity_emb": entity_emb,
        "relation_emb": relation_emb_mod,
        "generator": generator,
        "discriminator": discriminator,
        "attention": attention,
        "fusion": fusion,
        "retriever": retriever,
        "neighbor_cache": neighbor_cache,
        "entity2id": entity2id,
        "rel2id": rel2id,
        "id2entity": id2entity,
        "id2rel": id2rel,
        "rl_policy": rl_policy,
        "device": device,
        "noise_dim": noise_dim,
    }


def predict_and_explain(artifacts, head_name, relation_name, topk=10, num_samples=10):
    from pro_b_gan_kg.attention import batch_neighbors
    from pro_b_gan_kg.rl_evidence import run_evidence_rollout

    entity2id = artifacts["entity2id"]
    rel2id = artifacts["rel2id"]
    id2entity = artifacts["id2entity"]
    id2rel = artifacts["id2rel"]
    entity_emb = artifacts["entity_emb"]
    relation_emb = artifacts["relation_emb"]
    device = artifacts["device"]
    noise_dim = artifacts["noise_dim"]

    h_id = entity2id[head_name]
    r_id = rel2id[relation_name]

    h = torch.tensor([h_id], device=device)
    r_ids = torch.tensor([r_id], device=device)
    r_emb = relation_emb(r_ids)

    neighbors = artifacts["neighbor_cache"].get(h_id, r_id)
    if neighbors:
        neighbor_emb, mask = batch_neighbors([neighbors[:64]], entity_emb, device)
        context, alpha = artifacts["attention"](
            h_emb=entity_emb[h],
            r_emb=r_emb,
            neighbor_emb=neighbor_emb,
            mask=mask,
        )
        attn_weights = alpha.squeeze(0).detach().cpu().numpy()
        top_attn_idx = np.argsort(attn_weights)[::-1][:5]
        attn_neighbors = []
        for i in top_attn_idx:
            if i < len(neighbors[:64]):
                nid = neighbors[:64][i]
                attn_neighbors.append({
                    "entity": id2entity.get(nid, str(nid)),
                    "weight": round(float(attn_weights[i]), 4),
                })
    else:
        context = torch.zeros(1, entity_emb.shape[1], device=device)
        attn_neighbors = []

    all_scores = {}
    for _ in range(num_samples):
        noise = torch.randn(1, noise_dim, device=device)
        t_hat = artifacts["generator"](entity_emb[h], r_emb, context, noise)
        t_hat_norm = t_hat / (torch.norm(t_hat, dim=-1, keepdim=True) + 1e-9)
        scores, ids = artifacts["retriever"].search(
            t_hat_norm.detach().cpu().numpy().astype(np.float32), topk * 2
        )
        for cand_id, score in zip(ids[0].tolist(), scores[0].tolist()):
            if cand_id not in all_scores:
                all_scores[cand_id] = []
            all_scores[cand_id].append(float(score))

    ranked = sorted(all_scores.items(), key=lambda x: -sum(x[1]) / len(x[1]))[:topk]

    results = []
    for rank, (cand_id, scores) in enumerate(ranked, start=1):
        avg_score = sum(scores) / len(scores)
        cand_emb = entity_emb[torch.tensor([cand_id], device=device)]

        disc_raw = artifacts["discriminator"](entity_emb[h], r_emb, cand_emb, context).item()
        disc_pct = round(torch.sigmoid(torch.tensor(disc_raw)).item() * 100, 1)

        evidence = []
        if artifacts["rl_policy"] is not None:
            try:
                steps = run_evidence_rollout(
                    policy=artifacts["rl_policy"],
                    entity_emb=entity_emb,
                    neighbors=artifacts["neighbor_cache"].pairs,
                    query=(h_id, r_id),
                    target_tail=cand_id,
                    budget=3,
                )
                for s in steps:
                    h_name = id2entity.get(s.head, str(s.head))
                    r_name = id2rel.get(s.rel, str(s.rel))
                    t_name = id2entity.get(s.tail, str(s.tail))
                    evidence.append(f"{h_name} --[{r_name}]--> {t_name}")
            except Exception:
                pass

        results.append({
            "rank": rank,
            "entity": id2entity.get(cand_id, str(cand_id)),
            "prediction_score": round(avg_score, 4),
            "confidence": disc_pct,
            "evidence": evidence if evidence else ["No path found (may be novel prediction)"],
            "attention_neighbors": attn_neighbors,
        })

    return results


st.markdown(
    """
    <div style="text-align:center; padding:20px 0 10px;">
        <h1>🧬 PRO-B GAN KG</h1>
        <p style="font-size:18px; color:#475569;">
            Biomedical Knowledge Graph Link Prediction<br/>
            <em>CompGCN + GAN + RL Evidence Paths</em>
        </p>
    </div>
    <hr/>
    """,
    unsafe_allow_html=True,
)

# PROOF OF CONCEPT WARNING
st.markdown(
    """
    <div style="background-color:#ff4444; color:white; padding:15px; border-radius:8px; margin-bottom:20px; font-weight:bold;">
        ⚠️ <strong>PROOF OF CONCEPT</strong><br/>
        This model has <strong>NOT been fully trained</strong>. Results are for demonstration purposes only and should not be used for actual biomedical or scientific decision-making. This is a prototype to showcase the architecture and methodology.
    </div>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_metadata():
    """Load entity and relation metadata from artifact JSON files."""
    entity2id = {}
    rel2id = {}
    id2entity = {}
    id2rel = {}
    
    try:
        # Download artifacts if missing
        ensure_artifacts()
        
        if (ARTIFACT_DIR / "entity2id.json").exists():
            entity2id = json.loads((ARTIFACT_DIR / "entity2id.json").read_text())
            id2entity = {v: k for k, v in entity2id.items()}
        
        if (ARTIFACT_DIR / "rel2id.json").exists():
            rel2id = json.loads((ARTIFACT_DIR / "rel2id.json").read_text())
            id2rel = {v: k for k, v in rel2id.items()}
    except Exception as e:
        st.sidebar.error(f"Could not load metadata: {e}")
    
    return entity2id, rel2id, id2entity, id2rel


st.sidebar.markdown("### Query Settings")

@st.cache_resource
def load_and_prepare_metadata():
    """Load metadata and determine available relations per entity."""
    entity2id = {}
    rel2id = {}
    id2entity = {}
    id2rel = {}
    available_relations_per_entity = {}
    
    try:
        ensure_artifacts()
        
        if (ARTIFACT_DIR / "entity2id.json").exists():
            entity2id = json.loads((ARTIFACT_DIR / "entity2id.json").read_text())
            id2entity = {v: k for k, v in entity2id.items()}
        
        if (ARTIFACT_DIR / "rel2id.json").exists():
            rel2id = json.loads((ARTIFACT_DIR / "rel2id.json").read_text())
            id2rel = {v: k for k, v in rel2id.items()}
        
        # Load neighbor cache to find which relations each entity has
        if (ARTIFACT_DIR / "neighbors_index.npy").exists():
            try:
                from pro_b_gan_kg.data import NeighborCache
                neighbor_cache = NeighborCache.load(ARTIFACT_DIR / "neighbors_index.npy")
                
                # Build map of entity -> available relations
                for (h_id, r_id) in neighbor_cache.pairs.keys():
                    if h_id not in available_relations_per_entity:
                        available_relations_per_entity[h_id] = set()
                    available_relations_per_entity[h_id].add(r_id)
            except Exception:
                pass
    
    except Exception as e:
        st.sidebar.error(f"Error loading metadata: {e}")
    
    return entity2id, rel2id, id2entity, id2rel, available_relations_per_entity


st.sidebar.markdown("### Query Settings")

# Load all metadata
entity2id, rel2id, id2entity, id2rel, available_relations_per_entity = load_and_prepare_metadata()

# Get top 5 entities with most relations available
entity_relation_counts = {}
for h_id, relations in available_relations_per_entity.items():
    if h_id in id2entity:
        entity_relation_counts[h_id] = len(relations)

top_5_entity_ids = sorted(entity_relation_counts.items(), key=lambda x: -x[1])[:5]
top_5_entities = [(id2entity[eid], eid) for eid, _ in top_5_entity_ids]

# Display names and internal IDs
entity_display_names = [name for name, _ in top_5_entities]

# Entity dropdown
selected_entity_name = st.sidebar.selectbox(
    "Head Entity",
    options=entity_display_names,
    index=0 if entity_display_names else None,
    key="entity_select"
)

# Get the ID of selected entity
selected_entity_id = None
if selected_entity_name:
    selected_entity_id = entity2id.get(selected_entity_name, None)

# Get available relations for this entity
available_rels = []
if selected_entity_id is not None and selected_entity_id in available_relations_per_entity:
    available_rels = sorted([id2rel[r_id] for r_id in available_relations_per_entity[selected_entity_id] if r_id in id2rel])

# Relation dropdown (filtered to available relations)
relation = st.sidebar.selectbox(
    "Relation Type",
    options=available_rels if available_rels else ["(none available)"],
    index=0 if available_rels else None,
    key="relation_select",
    help="Only shows relations valid for the selected entity"
)

topk = st.sidebar.slider("Top-K Results", 1, 20, 10)
num_samples = st.sidebar.slider("Generator Samples", 1, 20, 10)

if st.sidebar.button("🔍 Predict", use_container_width=True):
    if not head or head == "(none)" or not relation or relation == "(none available)":
        st.error("Please select both a head entity and relation.")
        st.stop()

    with st.spinner("Loading models and artifacts (first run may take 1-2 minutes)..."):
        try:
            ensure_artifacts()
            artifacts = load_model()
        except Exception as e:
            st.error(f"Failed to load model: {str(e)[:500]}")
            st.stop()

    if head not in artifacts["entity2id"]:
        st.error(f"Entity '{head}' not found in knowledge graph.")
        st.stop()
    if relation not in artifacts["rel2id"]:
        st.error(f"Relation '{relation}' not found.")
        st.stop()

    with st.spinner("Running inference + evidence rollout..."):
        try:
            results = predict_and_explain(artifacts, head, relation, topk, num_samples)
        except Exception as e:
            st.error(f"Prediction failed: {str(e)[:500]}")
            st.stop()

    st.markdown(f"### Query: ({head}, {relation}, ?)")
    st.markdown(f"*{len(results)} predictions returned*")

    for r in results:
        conf_color = "#22c55e" if r["confidence"] > 90 else "#eab308" if r["confidence"] > 70 else "#ef4444"
        with st.expander(
            f"**Rank {r['rank']}:** {r['entity']}  —  Confidence: {r['confidence']}%",
            expanded=(r["rank"] <= 3),
        ):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Prediction Score", r["prediction_score"], 
                         help="Similarity between generated candidate and context (0-1 scale)")
            with col2:
                st.markdown(
                    f"<div style='font-size:28px; font-weight:bold; color:{conf_color}; text-align:center;'>"
                    f"{r['confidence']}%</div>"
                    f"<div style='text-align:center; font-size:12px; color:{conf_color};'>Confidence</div>",
                    unsafe_allow_html=True,
                )

            st.markdown("**Evidence Path:**")
            if r["evidence"]:
                for step in r["evidence"]:
                    st.code(step, language=None)
            else:
                st.info("No evidence path found")

            if r["attention_neighbors"]:
                st.markdown("**Top Neighbors:**")
                for n in r["attention_neighbors"]:
                    st.markdown(f"- `{n['entity']}`: {n['weight']}")


    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#94a3b8; font-size:13px;'>"
        "PRO-B GAN KG — Vanderbilt University"
        "</div>",
        unsafe_allow_html=True,
    )
else:
    st.info(
        """
        **Welcome to PRO-B GAN KG!**
        
        Select an entity and relation, then click **Predict** to find likely linked entities.
        """
    )
