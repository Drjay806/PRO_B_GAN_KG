# Data Pipeline

## Folder Structure

### 1. `raw/`
Download OGBL-BioKG raw triples here:
```bash
wget https://ogb.stanford.edu/data/biokg/ogbl_biokg.zip -O data/raw/ogbl_biokg.zip
unzip data/raw/ogbl_biokg.zip -d data/raw/
```

Expected files:
- `train_triples.txt` 
- `valid_triples.txt`
- `test_triples.txt`

### 2. `processed/`
Output from step 1 (data preparation):
```bash
python preprocessing/prepare_ogbl_biokg.py \
  --raw_data_path data/raw/ogbl_biokg \
  --output_dir data/processed
```

Produces:
- `entity2id.json` - Entity name → ID mapping
- `rel2id.json` - Relation name → ID mapping
- `train.tsv` - Processed training triples (h_id, r_id, t_id)
- `val.tsv` - Validation triples
- `test.tsv` - Test triples

### 3. `metadata/`
Output from step 2 (metadata fetching):
```bash
python preprocessing/fetch_metadata.py \
  --entity2id_path data/processed/entity2id.json \
  --output_dir data/metadata
```

Produces:
- `all_metadata.json` - Text descriptions for all entities by type
- `cache/go_cache.json` - Gene Ontology definitions (cached)
- `cache/pathway_cache.json` - Pathway descriptions (cached)
- `cache/disease_cache.json` - Disease definitions (cached)
- `cache/protein_cache.json` - Protein info (cached)

### 4. `embeddings/`
Output from step 3 (semantic encoding):
```bash
python preprocessing/preprocess_semantics.py \
  --metadata_path data/metadata/all_metadata.json \
  --output_dir data/embeddings
```

Produces:
- `go_embeddings.pt` - GO term embeddings
- `pathway_embeddings.pt` - Pathway embeddings
- `disease_embeddings.pt` - Disease embeddings
- `protein_embeddings.pt` - Protein embeddings
- `semantic_metadata.json` - Embedding statistics

## Full Pipeline
```bash
# Step 1: Prepare dataset
python preprocessing/prepare_ogbl_biokg.py \
  --raw_data_path data/raw/ogbl_biokg \
  --output_dir data/processed

# Step 2: Fetch metadata from APIs
python preprocessing/fetch_metadata.py \
  --entity2id_path data/processed/entity2id.json \
  --output_dir data/metadata

# Step 3: Generate semantic embeddings
python preprocessing/preprocess_semantics.py \
  --metadata_path data/metadata/all_metadata.json \
  --output_dir data/embeddings

# Step 4: Train model
python train.py --config_path config_with_semantics.json
```
