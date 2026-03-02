import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from pro_b_gan_kg.utils import setup_logging


def preprocess_semantic_embeddings(
    metadata_path: Path,
    output_dir: Path,
    model_name: str = "allenai/scibert_scivocab_uncased",
    embedding_dim: int = 768,
    batch_size: int = 32,
) -> None:
    logger = setup_logging(output_dir)
    logger.info(f"Preprocessing semantic embeddings with model: {model_name}")
    logger.info(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading model on device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    
    # Enable memory-efficient attention if available (for large batches in Colab)
    if hasattr(model, 'enable_xformers_memory_efficient_attention'):
        try:
            model.enable_xformers_memory_efficient_attention()
            logger.info("Enabled xformers memory-efficient attention")
        except Exception:
            pass

    embeddings_by_type = {}

    for entity_type, entities_dict in metadata.items():
        logger.info(f"Processing {entity_type}...")
        embeddings = {}

        entity_ids = sorted(entities_dict.keys())
        for i in tqdm(range(0, len(entity_ids), batch_size), desc=entity_type):
            batch_ids = entity_ids[i : i + batch_size]
            texts = [entities_dict[eid] for eid in batch_ids]

            inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                embeddings_batch = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token

            for eid, emb in zip(batch_ids, embeddings_batch):
                embeddings[eid] = emb.cpu().numpy().tolist()
            
            # Clear GPU cache periodically
            if device.type == 'cuda' and i % (batch_size * 10) == 0:
                torch.cuda.empty_cache()

        embeddings_by_type[entity_type] = embeddings
        logger.info(f"  Generated embeddings for {len(embeddings)} {entity_type} entities")

    for entity_type, embeddings in embeddings_by_type.items():
        output_file = output_dir / f"{entity_type}_embeddings.pt"
        torch.save(embeddings, output_file)
        logger.info(f"Saved {entity_type} embeddings to {output_file}")

    metadata_out = {
        "model": model_name,
        "embedding_dim": embedding_dim,
        "entity_types": list(embeddings_by_type.keys()),
        "counts": {et: len(emb) for et, emb in embeddings_by_type.items()},
    }
    with open(output_dir / "semantic_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata_out, f, indent=2)
    logger.info("Preprocessing complete")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess semantic embeddings from metadata")
    parser.add_argument("--metadata_path", type=str, required=True, help="Path to all_metadata.json")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for embeddings")
    parser.add_argument(
        "--model_name",
        type=str,
        default="allenai/scibert_scivocab_uncased",
        help="HuggingFace model name for text encoding",
    )
    parser.add_argument("--embedding_dim", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    args = parser.parse_args()

    preprocess_semantic_embeddings(
        metadata_path=Path(args.metadata_path),
        output_dir=Path(args.output_dir),
        model_name=args.model_name,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
