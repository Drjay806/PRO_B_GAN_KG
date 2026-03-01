"""
Data Preparation Pipeline for OGBL-BioKG

This guide explains how to download, preprocess, and prepare OGBL-BioKG for training.

## Step 1: Download the dataset

Download OGBL-BioKG from the Open Graph Benchmark:
https://ogb.stanford.edu/docs/linkprop/#ogbl-biokg

Place the extracted files in:
data/raw/ogbl_biokg/
├── train_triples.txt
├── val_triples.txt
└── test_triples.txt

Each file format:
head_entity_id <tab> relation_id <tab> tail_entity_id

Example:
protein_1  binds  protein_2
protein_1  associated  disease_5

## Step 2: Prepare the dataset

Convert raw OGBL-BioKG to our internal format:

```bash
python scripts/prepare_ogbl_biokg.py \
  --raw_data_dir ./data/raw/ogbl_biokg \
  --output_dir ./data/ogbl_biokg_prepared \
  --embedding_dim 256
```

This will output:
- entity2id.json                 (entity name -> ID mapping)
- rel2id.json                    (relation name -> ID mapping)
- train.tsv                      (ID-form triples for training)
- val.tsv                        (validation set)
- test.tsv                       (test set)
- protein_texts.json             (extracted text for proteins)
- go_texts.json                  (extracted text for GO terms)
- pathway_texts.json             (pathway descriptions)
- disease_texts.json             (disease descriptions)
- side_effect_texts.json         (side effect descriptions)
- dataset_stats.json             (statistics)

## Step 3: Generate semantic embeddings

Encode the extracted text with a BERT model:

```bash
python scripts/preprocess_semantics.py \
  --output_dir ./data/ogbl_biokg_embeddings \
  --embedding_dim 256 \
  --text_model allenai/scibert_scivocab_uncased \
  --entity2id_path ./data/ogbl_biokg_prepared/entity2id.json
```

This generates:
- embeddings/go_embeddings.pt
- embeddings/pathway_embeddings.pt
- embeddings/disease_embeddings.pt
- semantic_metadata.json

## Step 4: Train Prot-B-GAN

Update config with your paths and run:

```bash
python train.py \
  --config config_with_semantics.json \
  --output_dir ./output/ogbl_biokg_run
```

In config_with_semantics.json, set:
```json
{
  "data": {
    "train_path": "./data/ogbl_biokg_prepared/train.tsv",
    "val_path": "./data/ogbl_biokg_prepared/val.tsv",
    "test_path": "./data/ogbl_biokg_prepared/test.tsv",
    "delimiter": "\t",
    "has_header": false
  },
  "semantic": {
    "embeddings_dir": "./data/ogbl_biokg_embeddings/embeddings"
  }
}
```

## Full pipeline (one command)

```bash
# Download
wget ... # (download from OGB)

# Prepare
python scripts/prepare_ogbl_biokg.py --raw_data_dir ./data/raw/ogbl_biokg --output_dir ./data/ogbl_biokg_prepared

# Semantics
python scripts/preprocess_semantics.py --output_dir ./data/ogbl_biokg_embeddings --entity2id_path ./data/ogbl_biokg_prepared/entity2id.json

# Train
python train.py --config config_with_semantics.json --output_dir ./output/ogbl_biokg_run
```

## Troubleshooting

**"train_triples.txt not found"**
- Ensure you downloaded OGBL-BioKG and placed files in data/raw/ogbl_biokg/

**"Entity not in mappings"**
- Some triples may reference entities or relations not in all three splits
- The converter skips these (logged as DEBUG)
- This is normal and expected

**"No text found for entities"**
- Fallback descriptions are used
- For better results, provide actual descriptions in metadata files
- See BioKGTextExtractor for format
"""
