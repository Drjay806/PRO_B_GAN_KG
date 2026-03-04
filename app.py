import json
import os
import sys
import csv
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(
    page_title="PRO-B GAN KG",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    [data-testid="collapsedControl"] button svg {
        display: none;
    }
    [data-testid="collapsedControl"] button::before {
        content: "☰";
        font-size: 22px;
        font-weight: 700;
        line-height: 1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Human-readable entity mappings (name -> ID)
ENTITY_NAME_MAPPING = {
    "TP53 (Tumor Suppressor)": "P04637",
    "EGFR (Growth Factor Receptor)": "P00533",
    "BRCA1 (DNA Repair)": "P38398",
    "MYC (Transcription Factor)": "P01106",
    "PTEN (Phosphatase)": "P60484",
    "AKT1 (Protein Kinase)": "P31749",
    "RAF1 (Serine Kinase)": "P04049",
    "KRAS (GTPase)": "P01116",
    "HIF1A (Hypoxia Factor)": "Q16665",
    "JAK2 (Tyrosine Kinase)": "O60674",
}

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


def get_entity_type(entity_id: str) -> str:
    """Detect entity type from ID prefix/pattern."""
    if not entity_id:
        return "unknown"
    # Protein: UniProt ID format (P/Q/A + 5 digits)
    if entity_id[0] in ['P', 'Q', 'A', 'O'] and len(entity_id) >= 6:
        return "Protein"
    # Disease: starts with 'D' or known disease prefixes
    if entity_id.startswith('D') or entity_id.startswith('MESH'):
        return "Disease"
    # Drug/Compound: typically numeric or specific prefixes
    if entity_id.startswith('CHEMBL') or entity_id.isdigit():
        return "Drug/Compound"
    # Gene Ontology
    if entity_id.startswith('GO:'):
        return "Gene Ontology"
    # Pathway
    if entity_id.startswith('KEGG') or entity_id.startswith('REACTOME'):
        return "Pathway"
    return "Other"


@st.cache_resource
def load_entity_metadata():
    """Load entity metadata from entity2text_ALL.final.tsv."""
    metadata_index = {}
    try:
        metadata_path = Path("entity2text_ALL.final.tsv")
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                for row in reader:
                    node_id = (row.get("node_id") or "").strip()
                    if not node_id:
                        continue
                    metadata_index[node_id] = {
                        "display_name": (row.get("mapped_to") or "").strip() or node_id,
                        "description": (row.get("text") or "").strip(),
                        "source": (row.get("source") or "").strip(),
                        "node_type": (row.get("node_type") or "").strip(),
                    }
    except Exception:
        pass
    return metadata_index


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
    node_degree = {}
    for (head_id, _rel_id), tails in neighbor_cache.pairs.items():
        node_degree[head_id] = node_degree.get(head_id, 0) + len(tails)
        for tail_id in tails:
            node_degree[tail_id] = node_degree.get(tail_id, 0) + 1

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
        "node_degree": node_degree,
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
    node_degree = artifacts["node_degree"]
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

        cand_name = id2entity.get(cand_id, str(cand_id))
        cand_type = get_entity_type(cand_name)
        
        # Calculate DistMult score (relation embedding dot product)
        with torch.no_grad():
            cand_emb_tensor = entity_emb[cand_id]
            distmult_score = torch.dot(r_emb[0], cand_emb_tensor).item()

        results.append({
            "rank": rank,
            "entity_name": cand_name,
            "entity_id": cand_id,
            "entity_type": cand_type,
            "prediction_score": round(avg_score, 4),
            "distmult_score": round(distmult_score, 4),
            "confidence": disc_pct,
            "node_degree": int(node_degree.get(cand_id, 0)),
            "evidence": evidence if evidence else ["No path found (may be novel prediction)"],
            "attention_neighbors": attn_neighbors,
            "description": None,  # Can be populated from metadata if available
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
    """Load entity mappings, relations, and build neighbor index."""
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
        
        # Load neighbor cache to build available relations per entity
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
entity_metadata = load_entity_metadata()


def _normalize_head_type(entity_key, node_type):
    node_type = (node_type or "").strip()
    if node_type in {"Pathway", "kegg_Pathway"}:
        return "Pathway"
    if node_type in {"Drug", "Compound"}:
        return "Drug/Compound"
    if node_type == "Protein":
        return "Protein"

    key = (entity_key or "").upper()
    if key.startswith("DB") or key.startswith("CHEMBL"):
        return "Drug/Compound"
    if key.startswith("R-HSA") or key.startswith("KEGG") or key.startswith("REACTOME"):
        return "Pathway"
    if len(key) >= 6 and key[0] in {"P", "Q", "A", "O"}:
        return "Protein"
    return "Other"


def build_head_entity_options(entity2id_map, metadata_map, available_rel_map):
    per_type_cap = {
        "Protein": 12,
        "Drug/Compound": 12,
        "Pathway": 12,
    }
    counts = {k: 0 for k in per_type_cap}
    options = []

    for entity_key, entity_id in entity2id_map.items():
        if entity_id not in available_rel_map:
            continue

        meta = metadata_map.get(entity_key, {})
        normalized_type = _normalize_head_type(entity_key, meta.get("node_type"))
        if normalized_type not in per_type_cap:
            continue
        if counts[normalized_type] >= per_type_cap[normalized_type]:
            continue

        display_name = (meta.get("display_name") or entity_key).strip()
        label = f"{display_name} [{normalized_type}] — {entity_key}"
        options.append({
            "label": label,
            "entity": entity_key,
            "type": normalized_type,
        })
        counts[normalized_type] += 1

        if all(counts[t] >= per_type_cap[t] for t in per_type_cap):
            break

    if not options:
        for display, entity_key in ENTITY_NAME_MAPPING.items():
            if entity_key in entity2id_map:
                options.append({"label": display, "entity": entity_key, "type": "Protein"})

    return options


head_entity_options = build_head_entity_options(entity2id, entity_metadata, available_relations_per_entity)
head_categories = ["All", "Protein", "Drug/Compound", "Pathway"]
selected_head_category = st.sidebar.selectbox(
    "Head Entity Category",
    options=head_categories,
    index=0,
    help="Filter head entity choices by type",
)

if selected_head_category == "All":
    visible_head_options = head_entity_options
else:
    visible_head_options = [opt for opt in head_entity_options if opt["type"] == selected_head_category]

# Human-readable entity dropdown
entity_display_options = [opt["label"] for opt in visible_head_options]
if entity_display_options:
    selected_entity_display = st.sidebar.selectbox(
        "Head Entity",
        options=entity_display_options,
        index=0,
        key="entity_select",
        help="Includes proteins, drugs/compounds, and pathways"
    )
else:
    selected_entity_display = None
    st.sidebar.warning("No entities available for this category.")

# Get the actual entity ID from display name
selected_entity_id = None
selected_entity_name = None
if selected_entity_display:
    selected_row = next((opt for opt in visible_head_options if opt["label"] == selected_entity_display), None)
    if selected_row:
        selected_entity_name = selected_row["entity"]
    selected_entity_id = entity2id.get(selected_entity_name)

# Get available relations for this entity
available_rels = []
if selected_entity_id is not None and selected_entity_id in available_relations_per_entity:
    available_rels = sorted([id2rel[r_id] for r_id in available_relations_per_entity[selected_entity_id] if r_id in id2rel])

# Relation dropdown (dynamically filtered based on head entity)
relation = st.sidebar.selectbox(
    "Relation Type",
    options=available_rels if available_rels else ["(no relations available)"],
    index=0 if available_rels else None,
    key="relation_select",
    help="Shows only relations valid for the selected entity"
)

# Predict target type (pre-filter results by type)
st.sidebar.markdown("---")
st.sidebar.markdown("### Result Filters")
target_types = ["Any Type", "Protein", "Disease", "Drug/Compound", "Pathway", "Gene Ontology"]
target_type_filter = st.sidebar.selectbox(
    "Predict Target Type",
    options=target_types,
    index=0,
    help="Pre-filter results to only show predictions of this type"
)



topk = st.sidebar.slider("Top-K Results", 1, 20, 10)
num_samples = st.sidebar.slider("Generator Samples", 1, 20, 10)

if st.sidebar.button("🔍 Predict", use_container_width=True):
    if selected_entity_name is None or not relation or relation == "(no relations available)":
        st.error("Please select both a head entity and a valid relation.")
        st.stop()

    with st.spinner("Loading models and artifacts (first run may take 1-2 minutes)..."):
        try:
            ensure_artifacts()
            artifacts = load_model()
        except Exception as e:
            st.error(f"Failed to load model: {str(e)[:500]}")
            st.stop()

    if selected_entity_name not in artifacts["entity2id"]:
        st.error(f"Entity '{selected_entity_name}' not found in knowledge graph.")
        st.stop()
    if relation not in artifacts["rel2id"]:
        st.error(f"Relation '{relation}' not found.")
        st.stop()

    with st.spinner("Running inference + evidence rollout..."):
        try:
            # If filtering by type, get more results to ensure we have enough of that type
            effective_topk = topk
            if target_type_filter != "Any Type":
                effective_topk = min(100, topk * 5)  # Get 5x more results when filtering
            
            results = predict_and_explain(artifacts, selected_entity_name, relation, effective_topk, num_samples)
        except Exception as e:
            st.error(f"Prediction failed: {str(e)[:500]}")
            st.stop()

    # Load entity metadata
    entity_metadata = load_entity_metadata()
    
    st.markdown(f"### Query: ({selected_entity_name}, {relation}, ?)")
    
    with st.expander("ℹ️ Score Guide", expanded=False):
        st.markdown("- **Prediction Score**: retrieval ranking score; higher is better (query-dependent, no strict upper bound).")
        st.markdown("- **DistMult Score**: relation compatibility score; can be negative or positive; higher is better.")
        st.markdown("- **Confidence**: discriminator probability in **0%–100%**.")
    
    # Filter results by target type from sidebar
    filtered_results = results
    if target_type_filter != "Any Type":
        filtered_results = [r for r in results if r.get("entity_type") == target_type_filter]
        # Take only top-K of the filtered type
        filtered_results = filtered_results[:topk]
    else:
        # Take only top-K of all types
        filtered_results = filtered_results[:topk]
    
    st.markdown(f"**{len(filtered_results)} {target_type_filter if target_type_filter != 'Any Type' else ''} predictions returned**")

    for r in filtered_results:
        conf_color = "#22c55e" if r["confidence"] > 90 else "#eab308" if r["confidence"] > 70 else "#ef4444"
        entity_type = r.get("entity_type", "Other")
        meta = entity_metadata.get(r["entity_name"], {})
        display_name = meta.get("display_name") or r["entity_name"]
        description = meta.get("description") or ""
        source = meta.get("source") or ""
        metadata_type = meta.get("node_type") or entity_type
        
        # Create result card with name, ID, and type
        with st.expander(
            f"**Rank {r['rank']}:** {display_name} (Entity: {r['entity_name']}) [{metadata_type}] — Confidence: {r['confidence']}%",
            expanded=(r["rank"] <= 3),
        ):
            # Header with name, type, and confidence
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"### {display_name}")
                st.caption(f"Entity: `{r['entity_name']}` | Internal ID: `{r['entity_id']}` | Type: **{metadata_type}**")
            with col2:
                st.markdown(
                    f"<div style='font-size:28px; font-weight:bold; color:{conf_color}; text-align:center;'>"
                    f"{r['confidence']}%</div>"
                    f"<div style='text-align:center; font-size:12px; color:{conf_color};'>Confidence</div>",
                    unsafe_allow_html=True,
                )
            
            st.divider()
            
            # Details tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📋 Metadata", "🔗 Evidence", "👥 Context", "📊 Metrics"])
            
            with tab1:
                st.markdown("**Entity Information:**")
                st.write(f"**Display Name:** {display_name}")
                st.write(f"**Entity Key:** {r['entity_name']}")
                st.write(f"**Internal ID:** {r['entity_id']}")
                st.write(f"**Type:** {metadata_type}")
                st.write(f"**Source:** {source if source else 'N/A'}")
                if description:
                    st.write(f"**Description:** {description}")
                else:
                    st.info("No description found in entity2text metadata")
            
            with tab2:
                if r["evidence"]:
                    st.markdown("**Evidence Path (RL-discovered):**")
                    for step in r["evidence"]:
                        st.code(step, language=None)
                else:
                    st.info("No evidence path found - may be a novel prediction")
            
            with tab3:
                if r["attention_neighbors"]:
                    st.markdown("**Top Context Entities (from attention):**")
                    for n in r["attention_neighbors"]:
                        st.markdown(f"- **{n['entity']}** (weight: {n['weight']})")
                else:
                    st.info("No context entities found")
            
            with tab4:
                st.markdown("**Detailed Metrics:**")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Prediction Score", f"{r['prediction_score']:.4f}", help="FAISS + GAN combined score")
                with col_b:
                    st.metric("DistMult Score", f"{r['distmult_score']:.4f}", help="Relation embedding dot product")
                with col_c:
                    st.metric("Confidence", f"{r['confidence']:.1f}%", help="Discriminator confidence")
                
                st.markdown("**Generation Details:**")
                col_d, col_e, col_f = st.columns(3)
                with col_d:
                    st.metric("Generation Samples", num_samples, help="Samples used to generate this prediction")
                with col_e:
                    st.metric("Rank", r['rank'], help=f"Ranking among top-{topk} predictions")
                with col_f:
                    st.metric("Node Degree", r["node_degree"], help="Approximate graph degree from cached neighbors")



    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#94a3b8; font-size:13px;'>"
        "PROT-B-GAN KGC"
        "</div>",
        unsafe_allow_html=True,
    )
else:
    st.info(
        """
        **Welcome to PROT-B-GAN KG!**
        
        Select an entity and relation, then click **Predict** to find likely linked entities.
        """
    )
