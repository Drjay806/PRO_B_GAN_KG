# Prot-B-GAN KG: Link Prediction with GAN and Explainable Context

Knowledge graph completion using GAN-based link prediction with attention-based context and optional RL evidence paths.

## Data Preparation

Before training, you need to prepare your knowledge graph data:

1. **Download dataset** (e.g., OGBL-BioKG)
2. **Prepare** - Convert to our internal format with ID mappings
3. **Extract text** - Pull metadata for semantic encoding (optional but recommended)
4. **Encode semantics** - Generate BERT embeddings from text
5. **Train** - Run Prot-B-GAN with the prepared data

See [DATA_PREPARATION.md](DATA_PREPARATION.md) for step-by-step instructions with OGBL-BioKG example.

## Quick Start on Google Colab

```python
# Clone the repository
!git clone https://github.com/your-username/PRO_B_GAN_KG.git
%cd PRO_B_GAN_KG

# Install dependencies
!pip install -r requirements.txt

# Prepare your config
# Edit config_example.json with your data paths

# Run training
!python train.py --config config_example.json --output_dir ./output
```

## Pipeline Overview

1. **Data ingest**: Parse triples and build stable entity/relation mappings
2. **Embeddings**: Initialize learnable embeddings with optional DistMult pretraining
3. **Structural encoder**: CompGCN or RGCN refine embeddings using graph structure
4. **Fusion**: Combine semantic and structural embeddings
5. **Neighbor cache**: Build train-only adjacency for query context
6. **Attention context**: Compute weighted context from neighbors
7. **GAN training**: Generator creates tail embeddings, discriminator judges realism
8. **Retrieval**: FAISS index maps generated vectors to entity IDs
9. **Evaluation**: Filtered ranking metrics (MRR, Hits@K)
10. **Optional**: RL evidence paths and patch-based reranking

## Data Format

Your triples should be in TSV or CSV format:

```
h	r	t
BRCA1	interacts_with	BRCA2
BRCA2	causes	Breast_Cancer
TP53	binds	MDM2
```

Required files:
- `train.tsv`
- `val.tsv`
- `test.tsv`

## Configuration

See [config_example.json](config_example.json) for all available options.

Key parameters:
- `embedding_dim`: Embedding dimension (default: 256)
- `compgcn_layers`: Number of GCN layers (default: 2)
- `max_epochs_warmup`: Generator warm-up epochs (default: 10)
- `max_epochs_gan`: Adversarial training epochs (default: 30)
- `eval_topk`: Top-K for evaluation retrieval (default: 1000)

## Outputs

Training produces:
- `entity2id.json`, `rel2id.json`: ID mappings
- `train.tsv`, `val.tsv`, `test.tsv`: ID-converted triples
- `neighbors_index.npy`: Train neighbor cache
- `entity_emb_final.pt`: Final entity embeddings
- `faiss.index`: FAISS retrieval index
- `best_model.pt`: Best model checkpoint
- `metrics.json`: Evaluation results
- `run.log`: Training logs

## Inference and Explainability

### Quick Prediction

```python
from pathlib import Path
import torch
from pro_b_gan_kg.inference import load_inference_artifacts, predict
from pro_b_gan_kg.utils import get_device

checkpoint_dir = Path("./output")
device = get_device()

entity_emb, relation_emb, retriever, neighbor_cache = load_inference_artifacts(
    checkpoint_dir, device
)

# Load model components from best_model.pt
# Then predict:
results = predict(
    h_id=0,
    r_id=1,
    entity_emb=entity_emb,
    relation_emb=relation_emb,
    attention=attention_module,
    generator=generator_module,
    neighbor_cache=neighbor_cache,
    retriever=retriever,
    topk=10
)

print("Top candidates:", results["candidates"])
print("Attention weights:", results["attention"])
```

### Full Explainability

For comprehensive explanations including attention weights, DistMult scores, discriminator confidence, and optional RL evidence paths:

```python
from pro_b_gan_kg.explainability import explain_prediction, format_explanation_text

explanation = explain_prediction(
    h_id=0,
    r_id=1,
    entity_emb=entity_emb,
    relation_emb=relation_emb,
    attention=attention_module,
    generator=generator_module,
    discriminator=discriminator_module,
    neighbor_cache=neighbor_cache,
    retriever=retriever,
    id2entity=id2entity,
    id2rel=id2rel,
    topk=10,
    num_samples=10,
)

print(format_explanation_text(explanation))
```

**What you get per candidate:**
- Prediction score (from retrieval)
- DistMult structural score
- Discriminator confidence
- Attention weights (what neighbors influenced the prediction)
- RL evidence path (if exists, otherwise marked as novel)
- Node degree (hub detection)
- Generation diversity across samples

See [explain_example.py](explain_example.py) for a complete working example.

### Understanding Explainability vs Verification

**True Explanation (what actually influenced the GAN):**
- **Attention weights** - These neighbors and their weights were literally fed into the generator as context. High attention = high influence on the generated tail embedding.

**Post-hoc Verification (supporting evidence, not explanation):**
- **RL evidence paths** - Walks the training graph AFTER prediction to find paths connecting head to predicted tail. If no path exists, it may indicate a novel prediction.
- **DistMult score** - Structural plausibility based on triple factorization.
- **Discriminator score** - How "real" the generated triple looks.

When interpreting results: attention explains the generation process, while RL/DistMult/discriminator verify the prediction's plausibility.

## Semantic Embeddings

The model now supports optional semantic embeddings for non-protein entities (GO terms, pathways, diseases).

### Quick start:

**Step 1: Generate semantic embeddings (one-time)**
```bash
python scripts/preprocess_semantics.py \
  --output_dir ./data/embeddings \
  --embedding_dim 256 \
  --text_model allenai/scibert_scivocab_uncased \
  --entity2id_path ./data/entity2id.json
```

**Step 2: Train with semantic embeddings**
```bash
python train.py --config config_with_semantics.json --output_dir ./output
```

### What it does:

1. **Preprocessing** (step 1): Encodes text descriptions (GO definitions, pathway descriptions, disease definitions) using a BERT model and saves embeddings to disk.
2. **Training** (step 2): Loads precomputed embeddings at startup and initializes entity embeddings with them.

### Customization:

Edit `config_with_semantics.json` to:
- Change `embeddings_dir` to your precomputed embeddings location
- Modify `entity_type_mapping` to enable/disable specific entity types
- Adjust `embedding_dim` to match your embeddings

See [SEMANTIC_EMBEDDINGS.md](SEMANTIC_EMBEDDINGS.md) for details.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- FAISS (CPU or GPU)
- NumPy
- tqdm

See [requirements.txt](requirements.txt) for exact versions.

## Architecture Notes

- **No semantic encoders by default**: Embeddings are learned from structure. Add your own BERT/ESM encoders by replacing the initialization in `pro_b_gan_kg/training.py`.
- **FAISS is required**: Vector search is core to the retrieval-based evaluation.
- **Type constraints**: Not implemented yet. Add entity type filtering in retrieval for biological KGs.
- **RL evidence**: Optional explainability module, not used during training.

## Citation

If you use this code, please cite:

```
@software{prot_b_gan_kg,
  title={Prot-B-GAN KG: Link Prediction with GAN and Explainable Context},
  author={Your Name},
  year={2026}
}
```
