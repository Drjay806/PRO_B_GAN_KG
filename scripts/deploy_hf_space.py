"""
Deploy to HuggingFace Spaces (free tier).

Run this AFTER training completes on Colab.

Usage from PowerShell:
  pip install huggingface_hub
  huggingface-cli login
  python scripts/deploy_hf_space.py --space YOUR_HF_USERNAME/PRO-B-GAN-KG --artifacts PATH_TO_TRAINING_OUTPUT
"""
import argparse
import shutil
from pathlib import Path

REQUIRED_FILES = [
    "best_model.pt",
    "entity_emb_final.pt",
    "faiss.index",
    "entity2id.json",
    "rel2id.json",
    "neighbors_index.npy",
    "metrics.json",
]

OPTIONAL_FILES = [
    "rl_policy.pt",
]

CODE_FILES = [
    "app.py",
    "requirements-hf.txt",
]

PACKAGE_DIR = "pro_b_gan_kg"


def main():
    parser = argparse.ArgumentParser(description="Deploy to HuggingFace Spaces")
    parser.add_argument("--space", required=True, help="HF Space ID, e.g. Drjay806/PRO-B-GAN-KG")
    parser.add_argument("--artifacts", required=True, help="Path to training output dir with model files")
    parser.add_argument("--staging", default="hf_staging", help="Local staging folder")
    args = parser.parse_args()

    staging = Path(args.staging)
    artifacts_src = Path(args.artifacts)
    repo_root = Path(__file__).parent.parent

    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    print("1. Copying code files...")
    shutil.copy(repo_root / "app.py", staging / "app.py")
    shutil.copy(repo_root / "requirements-hf.txt", staging / "requirements.txt")

    print("2. Copying package...")
    pkg_dst = staging / PACKAGE_DIR
    shutil.copytree(repo_root / PACKAGE_DIR, pkg_dst,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    print("3. Copying model artifacts...")
    art_dst = staging / "artifacts"
    art_dst.mkdir()

    missing = []
    for f in REQUIRED_FILES:
        src = artifacts_src / f
        if src.exists():
            shutil.copy(src, art_dst / f)
            size_mb = src.stat().st_size / (1024 * 1024)
            print(f"   {f} ({size_mb:.1f} MB)")
        else:
            missing.append(f)

    for f in OPTIONAL_FILES:
        src = artifacts_src / f
        if src.exists():
            shutil.copy(src, art_dst / f)
            size_mb = src.stat().st_size / (1024 * 1024)
            print(f"   {f} ({size_mb:.1f} MB) [optional]")

    if missing:
        print(f"\nWARNING: Missing required files: {missing}")
        print("The Space will not work without these. Check your training output path.")
        return

    readme = staging / "README.md"
    readme.write_text(
        f"""---
title: PRO-B GAN KG
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.32.2"
python_version: "3.11"
app_file: app.py
pinned: false
---

# PRO-B GAN KG

Biomedical Knowledge Graph Link Prediction with CompGCN + GAN + RL Evidence Paths.

Built at Vanderbilt University.
"""
    )

    rt = staging / "runtime.txt"
    rt.write_text("python-3.11.9\n")

    total_mb = sum(
        f.stat().st_size for f in staging.rglob("*") if f.is_file()
    ) / (1024 * 1024)
    print(f"\nStaging complete: {staging} ({total_mb:.0f} MB total)")
    print(f"\nTo upload, run:")
    print(f"  cd {staging}")
    print(f"  git init")
    print(f"  git lfs install")
    print(f'  git lfs track "*.pt" "*.index" "*.npy"')
    print(f"  git add .")
    print(f'  git commit -m "Initial deploy"')
    print(f"  git remote add origin https://huggingface.co/spaces/{args.space}")
    print(f"  git push --force origin main")
    print(f"\nOr use the HF CLI:")
    print(f"  huggingface-cli upload {args.space} {staging} . --repo-type space")


if __name__ == "__main__":
    main()
