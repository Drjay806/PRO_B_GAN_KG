"""
Example usage of the explainability module for interpreting predictions.
"""
import argparse
import json
from pathlib import Path

import torch

from pro_b_gan_kg.explainability import explain_prediction, format_explanation_text
from pro_b_gan_kg.inference import load_inference_artifacts
from pro_b_gan_kg.utils import get_device, load_json


def load_full_model(checkpoint_dir: Path, device: torch.device):
    """Load all model components needed for explanation."""
    from pro_b_gan_kg.attention import ContextAttention
    from pro_b_gan_kg.gan import Discriminator, Generator
    
    checkpoint = torch.load(checkpoint_dir / "best_model.pt", map_location=device)
    config = load_json(checkpoint_dir / "metrics.json")["config"]
    
    entity_emb, relation_emb, retriever, neighbor_cache = load_inference_artifacts(
        checkpoint_dir, device
    )
    
    dim = config["model"]["embedding_dim"]
    
    attention = ContextAttention(
        dim=dim,
        hidden=config["model"]["attention_hidden"],
        dropout=config["model"]["dropout"],
    ).to(device)
    attention.load_state_dict(checkpoint["attention"])
    attention.eval()
    
    generator = Generator(
        dim=dim,
        hidden=config["model"]["generator_hidden"],
        noise_dim=config["model"]["noise_dim"],
    ).to(device)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()
    
    discriminator = Discriminator(
        dim=dim,
        hidden=config["model"]["discriminator_hidden"],
    ).to(device)
    discriminator.load_state_dict(checkpoint["discriminator"])
    discriminator.eval()
    
    entity2id = load_json(checkpoint_dir / "entity2id.json")
    rel2id = load_json(checkpoint_dir / "rel2id.json")
    id2entity = {v: k for k, v in entity2id.items()}
    id2rel = {v: k for k, v in rel2id.items()}
    
    return (
        entity_emb, relation_emb, attention, generator, discriminator,
        retriever, neighbor_cache, id2entity, id2rel
    )


def main():
    parser = argparse.ArgumentParser(description="Query the trained model")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to the training output directory")
    parser.add_argument("--head", type=str, default="BRCA1",
                        help="Head entity name to query")
    parser.add_argument("--relation", type=str, default="causes",
                        help="Relation name to query")
    parser.add_argument("--topk", type=int, default=10,
                        help="Number of top results to return")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    device = get_device()
    
    print("Loading model components...")
    (entity_emb, relation_emb, attention, generator, discriminator,
     retriever, neighbor_cache, id2entity, id2rel) = load_full_model(checkpoint_dir, device)
    
    entity2id = {v: k for k, v in id2entity.items()}
    rel2id = {v: k for k, v in id2rel.items()}
    
    query_head = args.head
    query_relation = args.relation
    
    if query_head not in entity2id or query_relation not in rel2id:
        print(f"Query not found in mappings. Available entities: {len(entity2id)}, relations: {len(rel2id)}")
        print("Using first entity and relation as example...")
        h_id = 0
        r_id = 0
    else:
        h_id = entity2id[query_head]
        r_id = rel2id[query_relation]
    
    print(f"\nExplaining query: ({id2entity[h_id]}, {id2rel[r_id]}, ?)")
    print("=" * 60)
    
    with torch.no_grad():
        explanation = explain_prediction(
            h_id=h_id,
            r_id=r_id,
            entity_emb=entity_emb,
            relation_emb=relation_emb,
            attention=attention,
            generator=generator,
            discriminator=discriminator,
            neighbor_cache=neighbor_cache,
            retriever=retriever,
            id2entity=id2entity,
            id2rel=id2rel,
            topk=args.topk,
            num_samples=10,
            evidence_policy=None,
            evidence_budget=3,
        )
    
    text_output = format_explanation_text(explanation)
    print(text_output)
    
    output_path = checkpoint_dir / "explanation_example.txt"
    with output_path.open("w", encoding="utf-8") as f:
        f.write(text_output)
    
    print(f"\nExplanation saved to: {output_path}")


if __name__ == "__main__":
    main()
