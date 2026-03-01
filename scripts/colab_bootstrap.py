import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Colab bootstrap for Prot-B-GAN KG")
    parser.add_argument("--install", action="store_true", help="Install Python dependencies")
    parser.add_argument(
        "--tsv_dir",
        type=str,
        default=None,
        help="Directory with metadata TSV files (e.g., data/prothgt/for_the_model)",
    )
    parser.add_argument(
        "--metadata_output_json",
        type=str,
        default="data/prothgt/metadata/all_metadata.json",
        help="Output path for generated all_metadata.json",
    )
    parser.add_argument(
        "--embeddings_output_dir",
        type=str,
        default="data/prothgt/embeddings",
        help="Output directory for generated embedding .pt files",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="allenai/scibert_scivocab_uncased",
        help="HuggingFace model for semantic embedding",
    )
    parser.add_argument("--embedding_dim", type=int, default=768, help="Embedding dimension metadata value")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for embedding generation")
    parser.add_argument(
        "--prepare_only",
        action="store_true",
        help="Only generate all_metadata.json from TSVs, skip embedding generation",
    )
    parser.add_argument("--run_train", action="store_true", help="Run training after preprocessing")
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON for training")
    parser.add_argument("--output_dir", type=str, default=None, help="Training output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    if args.install:
        requirements = repo_root / "requirements.txt"
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements)])

    if args.tsv_dir:
        preprocess_script = repo_root / "preprocessing" / "run_for_the_model_pipeline.py"
        preprocess_cmd = [
            sys.executable,
            str(preprocess_script),
            "--tsv_dir",
            args.tsv_dir,
            "--metadata_output_json",
            args.metadata_output_json,
            "--embeddings_output_dir",
            args.embeddings_output_dir,
            "--model_name",
            args.model_name,
            "--embedding_dim",
            str(args.embedding_dim),
            "--batch_size",
            str(args.batch_size),
        ]
        if args.prepare_only:
            preprocess_cmd.append("--prepare_only")
        subprocess.check_call(preprocess_cmd)

    if args.run_train:
        if not args.config or not args.output_dir:
            raise ValueError("--config and --output_dir are required when --run_train is set")
        train_script = repo_root / "train.py"
        subprocess.check_call(
            [sys.executable, str(train_script), "--config", args.config, "--output_dir", args.output_dir]
        )


if __name__ == "__main__":
    main()
