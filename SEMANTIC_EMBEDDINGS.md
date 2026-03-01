"""
Integration guide for semantic embeddings in Prot-B-GAN training.

## One-time setup (run once):

```bash
python scripts/preprocess_semantics.py \
  --output_dir ./data/embeddings \
  --embedding_dim 256 \
  --text_model allenai/scibert_scivocab_uncased \
  --entity2id_path ./output/entity2id.json
```

This generates:
- data/embeddings/go_embeddings.pt
- data/embeddings/pathway_embeddings.pt
- data/embeddings/disease_embeddings.pt
- data/embeddings/semantic_metadata.json

## Integration in training:

Add to config.json:
```json
{
  "semantic": {
    "embeddings_dir": "./data/embeddings",
    "entity_type_mapping": {
      "protein": "learned",
      "GO_term": "go_embeddings",
      "pathway": "pathway_embeddings",
      "disease": "disease_embeddings"
    }
  }
}
```

Then in train.py, load and fill embeddings before training:
```python
from pro_b_gan_kg.semantic_encoders.cache import SemanticEmbeddingCache

if run_cfg.semantic and run_cfg.semantic.embeddings_dir:
    semantic_cache = SemanticEmbeddingCache(run_cfg.semantic.embeddings_dir)
    
    # For each entity type, fill the embedding table
    for entity_type, source in run_cfg.semantic.entity_type_mapping.items():
        if source != "learned":
            entity_table = semantic_cache.fill_entity_table(
                source, 
                entity_emb.embedding.weight,
                {}  # mapping dict if available
            )
```

## What happens:

1. Precompute semantic embeddings once (text -> BERT -> vector)
2. Store precomputed vectors on disk
3. At training, load cached vectors and initialize entity embeddings
4. Train ends-to-end with semantic + structural learning
5. No semantic encoder runs during training (fast)
"""
