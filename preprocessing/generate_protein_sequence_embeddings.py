import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def _pick_column(columns: List[str], candidates: List[str]) -> str:
    lower_to_original = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    raise ValueError(f"Could not find any of columns: {candidates}. Available: {columns}")


def _load_sequences(tsv_path: Path) -> List[Tuple[str, str]]:
    df = pd.read_csv(tsv_path, sep="\t", dtype=str).fillna("")

    id_col = _pick_column(df.columns.tolist(), ["entity_id", "protein_id", "uniprot_id", "id"])
    seq_col = _pick_column(df.columns.tolist(), ["sequence", "protein_sequence", "aa_sequence", "amino_acid_sequence"])

    rows: List[Tuple[str, str]] = []
    for _, row in df.iterrows():
        entity_id = str(row[id_col]).strip()
        sequence = str(row[seq_col]).strip().upper().replace(" ", "")
        if not entity_id or not sequence:
            continue
        rows.append((entity_id, sequence))

    if not rows:
        raise ValueError("No valid (entity_id, sequence) rows found in TSV")

    return rows


def _random_projection_matrix(in_dim: int, out_dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    matrix = torch.randn(in_dim, out_dim, generator=generator) / (out_dim ** 0.5)
    return matrix


def generate_embeddings(
    sequence_rows: List[Tuple[str, str]],
    model_name: str,
    batch_size: int,
    max_length: int,
    target_dim: int,
    device: str,
    projection_seed: int,
) -> Dict[str, torch.Tensor]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    embeddings: Dict[str, torch.Tensor] = {}
    projection: torch.Tensor | None = None

    for start in tqdm(range(0, len(sequence_rows), batch_size), desc="Protein sequence embeddings"):
        batch = sequence_rows[start : start + batch_size]
        batch_ids = [item[0] for item in batch]
        batch_sequences = [item[1] for item in batch]

        inputs = tokenizer(
            batch_sequences,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            cls_vectors = outputs.last_hidden_state[:, 0, :].detach().cpu()

        if cls_vectors.shape[1] != target_dim:
            if projection is None:
                projection = _random_projection_matrix(cls_vectors.shape[1], target_dim, projection_seed)
            cls_vectors = cls_vectors @ projection

        for entity_id, vector in zip(batch_ids, cls_vectors):
            embeddings[entity_id] = vector.float()

        if device == "cuda":
            torch.cuda.empty_cache()

    return embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate protein embeddings from amino acid sequences")
    parser.add_argument("--sequences_tsv", type=str, required=True, help="TSV file containing entity_id and sequence")
    parser.add_argument("--output_pt", type=str, required=True, help="Output path for protein_embeddings.pt")
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/esm2_t30_150M_UR50D",
        help="Hugging Face protein model",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence token length")
    parser.add_argument(
        "--target_dim",
        type=int,
        default=768,
        help="Output embedding dim used by training model; vectors are projected if needed",
    )
    parser.add_argument("--projection_seed", type=int, default=7, help="Seed for deterministic projection")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sequence_rows = _load_sequences(Path(args.sequences_tsv))
    embeddings = generate_embeddings(
        sequence_rows=sequence_rows,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        target_dim=args.target_dim,
        device=device,
        projection_seed=args.projection_seed,
    )

    output_path = Path(args.output_pt)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, output_path)

    print(f"Saved {len(embeddings)} protein embeddings to {output_path}")
    print(f"Output embedding dim: {args.target_dim}")


if __name__ == "__main__":
    main()
