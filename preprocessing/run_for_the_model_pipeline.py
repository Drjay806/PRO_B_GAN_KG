import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.build_metadata_json_from_tsv import build_metadata_json
from preprocessing.preprocess_semantics import preprocess_semantic_embeddings


def run_pipeline(
    tsv_dir: Path,
    metadata_output_json: Path,
    embeddings_output_dir: Path,
    model_name: str,
    embedding_dim: int,
    batch_size: int,
    prepare_only: bool,
) -> None:
    build_metadata_json(
        tsv_dir=tsv_dir,
        output_json=metadata_output_json,
    )

    if prepare_only:
        return

    preprocess_semantic_embeddings(
        metadata_path=metadata_output_json,
        output_dir=embeddings_output_dir,
        model_name=model_name,
        embedding_dim=embedding_dim,
        batch_size=batch_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full TSV -> all_metadata.json -> semantic embeddings pipeline"
    )
    parser.add_argument(
        "--tsv_dir",
        type=str,
        default="data/prothgt/for_the_model",
        help="Directory containing metadata TSV files",
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
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=768,
        help="Embedding dimension metadata value",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for transformer inference",
    )
    parser.add_argument(
        "--prepare_only",
        action="store_true",
        help="Only build all_metadata.json from TSV files; skip embedding generation",
    )
    args = parser.parse_args()

    run_pipeline(
        tsv_dir=Path(args.tsv_dir),
        metadata_output_json=Path(args.metadata_output_json),
        embeddings_output_dir=Path(args.embeddings_output_dir),
        model_name=args.model_name,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        prepare_only=args.prepare_only,
    )


if __name__ == "__main__":
    main()
