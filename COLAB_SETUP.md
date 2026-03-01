# Google Colab Setup Guide

Complete guide for running PRO_B_GAN_KG embedding pipeline in Google Colab.

---

## 🚀 Quick Start (Copy-Paste into Colab)

### Step 1: Clone Repository & Install Dependencies

```python
# Clone the repo
!git clone https://github.com/YOUR_USERNAME/PRO_B_GAN_KG.git
%cd PRO_B_GAN_KG

# Install dependencies
!pip install -q -r requirements.txt

# Verify GPU is available
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

---

### Step 2: Upload TSV Metadata Files

**Option A: Mount Google Drive** (recommended for large files)
```python
from google.colab import drive
drive.mount('/content/drive')

# Copy TSV files from Drive to Colab
!mkdir -p data/prothgt/for_the_model
!cp -r /content/drive/MyDrive/prothgt_metadata/*.tsv data/prothgt/for_the_model/
```

**Option B: Direct Upload** (for testing with smaller files)
```python
from google.colab import files
import shutil

# Upload TSV files manually
uploaded = files.upload()

# Move to correct directory
!mkdir -p data/prothgt/for_the_model
for filename in uploaded.keys():
    shutil.move(filename, f'data/prothgt/for_the_model/{filename}')
```

---

### Step 3: Generate Embeddings

**Full Pipeline (TSV → JSON → Embeddings)**
```python
!python preprocessing/run_for_the_model_pipeline.py \
    --tsv_dir data/prothgt/for_the_model \
    --metadata_output_json data/prothgt/metadata/all_metadata.json \
    --embeddings_output_dir data/prothgt/embeddings \
    --model_name allenai/scibert_scivocab_uncased \
    --batch_size 64 \
    --embedding_dim 768
```

**Estimated Time (Colab T4 GPU)**:
- TSV → JSON: ~2 seconds (539K entities)
- JSON → Embeddings: ~45-60 minutes (539K entities, batch_size=64)
- Total: **~1 hour**

---

### Step 4: Download Results

```python
from google.colab import files
import shutil

# Zip embeddings folder
!zip -r embeddings.zip data/prothgt/embeddings/

# Download zip
files.download('embeddings.zip')

# Or copy back to Google Drive
!cp -r data/prothgt/embeddings /content/drive/MyDrive/prothgt_embeddings/
```

---

## 📊 Performance Optimization

### Batch Size Tuning

| GPU Type | Recommended Batch Size | Memory Usage | Speed |
|----------|----------------------|--------------|-------|
| T4 (free) | 64 | ~12 GB | ~60 min |
| V100 | 128 | ~20 GB | ~30 min |
| A100 | 256 | ~32 GB | ~15 min |

**Adjust batch size based on GPU:**
```python
# Check available GPU memory
!nvidia-smi

# Use larger batch if you have V100/A100
!python preprocessing/run_for_the_model_pipeline.py \
    --batch_size 128  # Increase if you have more GPU memory
```

### Memory Issues?

If you get CUDA out-of-memory errors:
```python
# Reduce batch size
!python preprocessing/run_for_the_model_pipeline.py \
    --batch_size 32  # Reduce to 32 or 16
```

---

## 🔍 Monitoring Progress

```python
# Check GPU utilization in real-time (run in separate cell)
!watch -n 1 nvidia-smi
```

```python
# Check embedding file sizes
!du -h data/prothgt/embeddings/*.pt
```

---

## 📁 Expected Output Files

After running the pipeline, you should have:

```
data/prothgt/
├── metadata/
│   └── all_metadata.json          (539K entities, ~120 MB)
└── embeddings/
    ├── protein_embeddings.pt      (326K proteins, ~950 MB)
    ├── compound_embeddings.pt     (133K compounds, ~390 MB)
    ├── go_embeddings.pt           (42K GO terms, ~125 MB)
    ├── disease_embeddings.pt      (2.8K diseases, ~8 MB)
    ├── drug_embeddings.pt         (6K drugs, ~18 MB)
    ├── pathway_embeddings.pt      (4K pathways, ~12 MB)
    ├── side_effect_embeddings.pt  (9K side effects, ~27 MB)
    ├── domain_embeddings.pt       (10K domains, ~30 MB)
    ├── ec_number_embeddings.pt    (4.6K EC numbers, ~14 MB)
    ├── semantic_metadata.json     (Summary stats)
    └── run.log                    (Detailed logs)
```

**Total Size**: ~1.6 GB

---

## 🧪 Testing / Debugging

### Test with Small Subset First

```python
# Create a small test subset
!head -n 1000 data/prothgt/for_the_model/proteins.tsv > data/prothgt/test/proteins.tsv
!head -n 500 data/prothgt/for_the_model/drugs.tsv > data/prothgt/test/drugs.tsv

# Run on test data
!python preprocessing/run_for_the_model_pipeline.py \
    --tsv_dir data/prothgt/test \
    --metadata_output_json data/prothgt/test_metadata.json \
    --embeddings_output_dir data/prothgt/test_embeddings \
    --batch_size 32
```

### Verify Embeddings

```python
import torch

# Load and inspect embeddings
protein_emb = torch.load('data/prothgt/embeddings/protein_embeddings.pt')
print(f"Protein embeddings: {len(protein_emb)} entities")
print(f"Sample embedding shape: {list(protein_emb.values())[0].shape}")  # Should be (768,)
```

---

## ⚠️ Common Issues

### Issue 1: "No module named 'transformers'"
**Solution:** Reinstall requirements
```python
!pip install -q -r requirements.txt
```

### Issue 2: "CUDA out of memory"
**Solution:** Reduce batch size
```python
!python preprocessing/run_for_the_model_pipeline.py --batch_size 16
```

### Issue 3: "TSV file not found"
**Solution:** Verify file paths
```python
!ls -lh data/prothgt/for_the_model/
```

### Issue 4: Session timeout during long runs
**Solution:** Keep Colab active using a JavaScript snippet
```javascript
// Run this in browser console (F12)
function ClickConnect(){
  console.log("Clicking connect");
  document.querySelector("colab-connect-button").click()
}
setInterval(ClickConnect, 60000)
```

---

## 🎯 Next Steps After Embeddings

Once embeddings are generated:

1. **Download** embeddings to local machine OR keep in Google Drive
2. **Run Training** with ProHGT train/val/test splits
3. **Evaluate** model performance

### Example Training Command (in Colab)
```python
!python train.py \
    --config config_prothgt.json \
    --output_dir ./output/prothgt_run
```

---

## 📝 Full Pipeline Example (All Steps)

```python
# === STEP 1: Setup ===
!git clone https://github.com/YOUR_USERNAME/PRO_B_GAN_KG.git
%cd PRO_B_GAN_KG
!pip install -q -r requirements.txt

# === STEP 2: Mount Drive ===
from google.colab import drive
drive.mount('/content/drive')

# === STEP 3: Copy Data ===
!mkdir -p data/prothgt/for_the_model
!cp /content/drive/MyDrive/prothgt_metadata/*.tsv data/prothgt/for_the_model/

# === STEP 4: Generate Embeddings ===
!python preprocessing/run_for_the_model_pipeline.py \
    --tsv_dir data/prothgt/for_the_model \
    --metadata_output_json data/prothgt/metadata/all_metadata.json \
    --embeddings_output_dir data/prothgt/embeddings \
    --batch_size 64

# === STEP 5: Verify Output ===
!ls -lh data/prothgt/embeddings/

# === STEP 6: Copy Back to Drive ===
!cp -r data/prothgt/embeddings /content/drive/MyDrive/prothgt_embeddings/

print("✓ Pipeline complete!")
```

---

## 💡 Tips for Faster Execution

1. **Use Colab Pro** for better GPUs (V100/A100) and longer runtimes
2. **Enable GPU** in Runtime → Change runtime type → Hardware accelerator → GPU
3. **Keep session active** to avoid disconnections
4. **Use Google Drive** for persistent storage across sessions
5. **Run during off-peak hours** for faster API access to HuggingFace models

---

## 🆘 Support

If you encounter issues:
1. Check the `run.log` file in the embeddings directory
2. Verify GPU is available with `!nvidia-smi`
3. Try running with `--batch_size 16` if memory issues persist
4. Open an issue on GitHub with error logs

---

**Last Updated**: March 2026  
**Tested On**: Google Colab (T4 GPU, 15GB RAM)
