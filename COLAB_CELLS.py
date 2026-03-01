# PRO_B_GAN_KG - Colab Quick Start
# Copy these cells into a new Colab notebook

# ============================================================
# CELL 1: Setup & Install
# ============================================================
# Clone repository and install dependencies
!git clone https://github.com/YOUR_USERNAME/PRO_B_GAN_KG.git
%cd PRO_B_GAN_KG
!pip install -q -r requirements.txt

# Verify GPU
import torch
print(f"✓ GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ============================================================
# CELL 2: Mount Google Drive (Optional but Recommended)
# ============================================================
from google.colab import drive
drive.mount('/content/drive')

# Copy TSV files from Drive to Colab workspace
!mkdir -p data/prothgt/for_the_model
!cp /content/drive/MyDrive/prothgt_metadata/*.tsv data/prothgt/for_the_model/

# Verify files copied
!ls -lh data/prothgt/for_the_model/


# ============================================================
# CELL 3: Generate Embeddings (Main Pipeline)
# ============================================================
# This is the main command - adjust batch_size based on your GPU
!python preprocessing/run_for_the_model_pipeline.py \
    --tsv_dir data/prothgt/for_the_model \
    --metadata_output_json data/prothgt/metadata/all_metadata.json \
    --embeddings_output_dir data/prothgt/embeddings \
    --model_name allenai/scibert_scivocab_uncased \
    --batch_size 64 \
    --embedding_dim 768

# Expected runtime: ~60 minutes on T4 GPU for 539K entities


# ============================================================
# CELL 4: Verify Embeddings Generated
# ============================================================
import torch
import json

# Check all embedding files
!ls -lh data/prothgt/embeddings/

# Load and inspect one embedding file
protein_emb = torch.load('data/prothgt/embeddings/protein_embeddings.pt')
print(f"\n✓ Protein embeddings: {len(protein_emb)} entities")

# Show metadata
with open('data/prothgt/embeddings/semantic_metadata.json', 'r') as f:
    metadata = json.load(f)
    print(f"\n✓ Model used: {metadata['model']}")
    print(f"✓ Embedding dimension: {metadata['embedding_dim']}")
    print(f"✓ Entity types: {', '.join(metadata['entity_types'])}")
    print(f"\n✓ Entity counts:")
    for entity_type, count in metadata['counts'].items():
        print(f"   - {entity_type}: {count:,}")


# ============================================================
# CELL 5: Copy Results Back to Google Drive
# ============================================================
# Create backup directory in Drive
!mkdir -p /content/drive/MyDrive/prothgt_embeddings

# Copy all embeddings to Drive
!cp -r data/prothgt/embeddings/* /content/drive/MyDrive/prothgt_embeddings/

print("✓ Embeddings backed up to Google Drive at: MyDrive/prothgt_embeddings/")


# ============================================================
# CELL 6: Download Embeddings as ZIP (Alternative to Drive)
# ============================================================
from google.colab import files

# Create zip archive
!cd data/prothgt && zip -r ../../embeddings.zip embeddings/

# Download to local machine
files.download('embeddings.zip')

print("✓ Download started - check your browser downloads")


# ============================================================
# OPTIONAL: Monitor GPU Usage (Run in separate cell)
# ============================================================
# Keep this running in a separate cell to monitor GPU
!watch -n 2 nvidia-smi


# ============================================================
# OPTIONAL: Test with Small Subset First
# ============================================================
# Create small test dataset
!mkdir -p data/prothgt/test
!head -n 1001 data/prothgt/for_the_model/proteins.tsv > data/prothgt/test/proteins.tsv
!head -n 501 data/prothgt/for_the_model/drugs.tsv > data/prothgt/test/drugs.tsv
!head -n 501 data/prothgt/for_the_model/compounds.tsv > data/prothgt/test/compounds.tsv

# Run on test subset (should take ~2 minutes)
!python preprocessing/run_for_the_model_pipeline.py \
    --tsv_dir data/prothgt/test \
    --metadata_output_json data/prothgt/test_metadata.json \
    --embeddings_output_dir data/prothgt/test_embeddings \
    --batch_size 32

print("✓ Test run complete - check data/prothgt/test_embeddings/")


# ============================================================
# TROUBLESHOOTING: If CUDA Out of Memory
# ============================================================
# Reduce batch size and try again
!python preprocessing/run_for_the_model_pipeline.py \
    --tsv_dir data/prothgt/for_the_model \
    --batch_size 16  # Reduced from 64 to 16

# Or restart runtime and free GPU memory
import torch
torch.cuda.empty_cache()
