"""
Example: Type-aware prediction filtering for biological queries.
Shows how to filter predictions by entity type (protein, compound, disease, etc.)
"""
from pathlib import Path

import torch

from pro_b_gan_kg.inference import predict, load_inference_artifacts
from pro_b_gan_kg.type_filter import (
    filter_candidates_by_type,
    filter_by_relation_signature,
    get_entity_type,
    get_type_statistics,
)
from pro_b_gan_kg.utils import get_device, load_json


def main():
    checkpoint_dir = Path("./output")
    device = get_device()
    
    print("Loading inference artifacts...")
    entity_emb, relation_emb, retriever, neighbor_cache = load_inference_artifacts(
        checkpoint_dir, device
    )
    
    entity2id = load_json(checkpoint_dir / "entity2id.json")
    rel2id = load_json(checkpoint_dir / "rel2id.json")
    id2entity = {v: k for k, v in entity2id.items()}
    id2rel = {v: k for k, v in rel2id.items()}
    
    # Example query: What proteins interact with UniProt:P38398?
    query_head = "UniProt:P38398"  # Replace with your entity
    query_relation = "interacts_with"  # Replace with your relation
    
    if query_head not in entity2id or query_relation not in rel2id:
        print(f"Query not found. Using first protein for demo...")
        # Find first protein
        query_head = next((k for k in entity2id.keys() if k.startswith("UniProt:")), list(entity2id.keys())[0])
        query_relation = list(rel2id.keys())[0]
    
    h_id = entity2id[query_head]
    r_id = rel2id[query_relation]
    head_type = get_entity_type(query_head)
    
    print(f"\n{'='*70}")
    print(f"Query: ({query_head}, {query_relation}, ?)")
    print(f"Head type: {head_type}")
    print(f"{'='*70}\n")
    
    # Get predictions (load attention and generator from checkpoint if needed)
    # For this example, we'll simulate candidates
    print("Getting predictions (top 50)...")
    
    # TODO: Load actual attention & generator models here
    # For now, use retriever directly for demo
    h_emb = entity_emb[h_id].unsqueeze(0).cpu().numpy()
    scores, ids = retriever.search(h_emb, 50)
    candidates = [(int(i), float(s)) for i, s in zip(ids[0], scores[0])]
    
    # Show type distribution of all candidates
    print("\n--- Type Distribution (All Candidates) ---")
    type_stats = get_type_statistics(candidates, id2entity)
    for entity_type, count in sorted(type_stats.items(), key=lambda x: -x[1]):
        print(f"  {entity_type:15s}: {count:3d}")
    
    # Filter 1: Get only protein candidates
    print("\n--- Filter 1: Proteins Only ---")
    protein_candidates = filter_candidates_by_type(
        candidates, id2entity, allowed_types={"protein"}
    )
    print(f"Found {len(protein_candidates)} protein candidates:")
    for i, (cand_id, score) in enumerate(protein_candidates[:10], 1):
        print(f"  {i}. {id2entity[cand_id]:30s} (score: {score:.4f})")
    
    # Filter 2: Get only compounds and drugs
    print("\n--- Filter 2: Compounds & Drugs ---")
    compound_candidates = filter_candidates_by_type(
        candidates, id2entity, allowed_types={"compound", "drug"}
    )
    print(f"Found {len(compound_candidates)} compound/drug candidates:")
    for i, (cand_id, score) in enumerate(compound_candidates[:10], 1):
        entity_name = id2entity[cand_id]
        entity_type = get_entity_type(entity_name)
        print(f"  {i}. {entity_name:30s} [{entity_type}] (score: {score:.4f})")
    
    # Filter 3: Use relation signature to filter biologically valid types
    print("\n--- Filter 3: Relation Signature Filtering ---")
    signature_filtered = filter_by_relation_signature(
        candidates, id2entity, query_relation, head_type
    )
    print(f"Found {len(signature_filtered)} candidates matching relation signature:")
    for i, (cand_id, score) in enumerate(signature_filtered[:10], 1):
        entity_name = id2entity[cand_id]
        entity_type = get_entity_type(entity_name)
        print(f"  {i}. {entity_name:30s} [{entity_type}] (score: {score:.4f})")
    
    # Filter 4: Multiple types (e.g., proteins OR pathways)
    print("\n--- Filter 4: Proteins OR Pathways ---")
    multi_type = filter_candidates_by_type(
        candidates, id2entity, allowed_types={"protein", "pathway"}
    )
    print(f"Found {len(multi_type)} protein/pathway candidates:")
    for i, (cand_id, score) in enumerate(multi_type[:10], 1):
        entity_name = id2entity[cand_id]
        entity_type = get_entity_type(entity_name)
        print(f"  {i}. {entity_name:30s} [{entity_type}] (score: {score:.4f})")
    
    print(f"\n{'='*70}")
    print("Type filtering complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
