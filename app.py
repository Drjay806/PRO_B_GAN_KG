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

# EXPLANATION SECTION
with st.expander("📖 How This Works (Click to expand)", expanded=False):
    st.markdown("""
    ### System Overview
    
    **PRO-B GAN KG** predicts missing links in biomedical knowledge graphs using three complementary techniques:
    
    1. **CompGCN (Composition-based Graph Convolutional Network)**
       - Encodes relational structure in the knowledge graph
       - Learns semantic embeddings for entities and relations
       - Captures multi-hop neighborhood context
    
    2. **GAN (Generative Adversarial Network)**
       - **Generator**: Creates candidate tail entities given (head, relation) pair
       - **Discriminator**: Scores predictions with confidence estimates
       - Learns to generate realistic knowledge graph completions
    
    3. **RL (Reinforcement Learning) Evidence Paths** *(optional)*
       - Discovers interpretable multi-hop reasoning paths
       - Explains *why* a prediction is made
       - Provides transparency through intermediate steps
    
    ### Input Parameters
    
    - **Head Entity**: The source/subject in the knowledge graph (e.g., a protein)
    - **Relation**: The type of relationship to query (e.g., "interacts_with", "regulates")
    - **Top-K Results**: Number of top predictions to return (1-20)
    - **Generator Samples**: Number of times to run the generator for robustness (1-20)
    
    ### Output Metrics Explained
    
    For each predicted tail entity:
    
    - **Prediction Score** (0-1): Raw similarity score between generated and retrieved candidate. Higher = more similar to generated prototype.
    
    - **Confidence %** (0-100%): Discriminator's confidence that this edge should exist in the knowledge graph. 
      - 🟢 90%+: High confidence
      - 🟡 70-90%: Medium confidence  
      - 🔴 <70%: Lower confidence
    
    - **Evidence Path**: Multi-hop reasoning chain from head entity to tail through intermediate entities.
      - Format: `Entity1 --[relation]--> Entity2 --[relation]--> Entity3`
      - Shows *how* the model reasons about the connection
    
    - **Attention Neighbors**: The most relevant neighboring entities the model attended to when making the prediction.
      - Sorted by attention weight (importance)
    """)

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

with st.spinner("Loading entity and relation data..."):
    entity2id, rel2id, id2entity, id2rel = load_metadata()

# Build entity/relation lists
all_entities = sorted(id2entity.values()) if id2entity else []
all_relations = sorted(id2rel.values()) if id2rel else []

if not all_entities:
    st.sidebar.warning("No entities loaded. This may be the first visit—please wait a moment and refresh.")
if not all_relations:
    st.sidebar.warning("No relations loaded. This may be the first visit—please wait a moment and refresh.")

# Entity dropdown (always show, even if empty)
head = st.sidebar.selectbox(
    "Head Entity",
    options=all_entities if all_entities else ["No entities available"],
    key="entity_select"
)

# Relation dropdown (show all for now, filtered if we can load neighbor cache)
relation = st.sidebar.selectbox(
    "Relation Type",
    options=all_relations if all_relations else ["No relations available"],
    key="relation_select"
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
    
    st.markdown("""
    ---
    #### Understanding the Results Below
    
    Each prediction includes:
    - **Prediction Score**: How similar the generated candidate is to the query context
    - **Confidence**: The discriminator's estimate of edge validity (0-100%)
    - **Evidence Path**: The reasoning chain (if RL path discovery succeeded)
    - **Attention Neighbors**: Entities the model focused on during reasoning
    
    Remember: These are *candidate* predictions from the model. Validation against real biomedical databases is essential.
    ---
    """)

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
                    f"<div style='text-align:center; font-size:12px; color:{conf_color};'>Discriminator Confidence</div>",
                    unsafe_allow_html=True,
                )

            st.markdown("**📊 Metric Explanations:**")
            st.markdown(f"""
            - **Prediction Score ({r['prediction_score']})**: How closely this entity matches the generated prototype from the model. Generated via transformer-based generation + FAISS vector search.
            
            - **Confidence ({r['confidence']}%)**: The discriminator network's estimate of whether this link actually belongs in the knowledge graph. Based on adversarial training against real KG patterns.
            """)

            st.markdown("**🔗 Evidence Path:**")
            if r["evidence"]:
                for i, step in enumerate(r["evidence"], 1):
                    st.code(step, language=None)
                st.markdown("*Multi-hop reasoning chain discovered via RL policy*")
            else:
                st.info("No evidence path found (model may predict novel/unseen patterns)")

            if r["attention_neighbors"]:
                st.markdown("**👁️ Top Attention Neighbors** (entities the model focused on):")
                for n in r["attention_neighbors"]:
                    st.markdown(f"- `{n['entity']}` — attention weight: {n['weight']}")
            
            st.markdown("""
            > **Interpretation**: This prediction means the model believes this entity is a likely completion of the (head, relation, ?) triple based on patterns learned during training.
            """)

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
        
        This is a biomedical knowledge graph link prediction system using CompGCN, GANs, and RL evidence paths.
        
        **How to use:**
        1. Select a **Head Entity** from the dropdown (⭐ marks popular entities)
        2. Select a **Relation** (automatically filtered to show valid relations for your entity)
        3. Click **Predict** to find likely tail entities
        
        **What you'll see:**
        - **Prediction Score**: How well the entity matches the model's generated prototype
        - **Confidence %**: The discriminator's estimate of edge validity
        - **Evidence Paths**: Multi-hop reasoning explaining the prediction
        - **Attention Neighbors**: Key entities the model focused on
        
        **⏱️ Timing**: First prediction may take 1-2 minutes as models download and load. Subsequent predictions are instant.
        
        **📖 Need details?** Click "How This Works" above to learn about the system architecture and metric meanings.
        """
    )
