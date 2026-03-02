"""
Extract triples from ProHGT HeteroData using biological ID mappings from id_samples.json
"""

import argparse
import json
import torch
from pathlib import Path


def load_id_mappings(mapping_path: Path) -> dict:
    """Load node ID mappings from JSON file."""
    with open(mapping_path) as f:
        mappings = json.load(f)
    
    # Expand mappings: convert list to dict {idx: biological_id}
    expanded = {}
    for node_type, id_list in mappings.items():
        expanded[node_type] = {i: bio_id for i, bio_id in enumerate(id_list)}
    
    return expanded


def extract_triples_with_bio_ids(hetero_graph, id_mappings: dict, output_tsv: Path):
    """Extract triples using biological ID mappings."""
    print(f"✓ Extracting triples with biological IDs")
    
    triples = []
    edge_count = 0
    
    # Iterate through edge types
    for edge_key in hetero_graph.edge_types:
        if isinstance(edge_key, tuple) and len(edge_key) == 3:
            head_type, rel_type, tail_type = edge_key
        else:
            continue
        
        edge_index = hetero_graph[edge_key].edge_index
        num_edges = edge_index.shape[1]
        
        # Get mappings for this edge type
        head_mapping = id_mappings.get(head_type, {})
        tail_mapping = id_mappings.get(tail_type, {})
        
        # Extract triples
        for i in range(num_edges):
            h_idx = edge_index[0, i].item()
            t_idx = edge_index[1, i].item()
            
            # Map numeric indices to biological IDs
            if h_idx in head_mapping:
                h = head_mapping[h_idx]
            else:
                h = f"{head_type}:{h_idx}"  # Fallback to numeric
            
            if t_idx in tail_mapping:
                t = tail_mapping[t_idx]
            else:
                t = f"{tail_type}:{t_idx}"  # Fallback to numeric
            
            triples.append((h, rel_type, t))
            edge_count += 1
        
        if edge_count % 100000 == 0:
            print(f"  Processed {edge_count} edges...")
    
    return triples


def extract_triples_from_graph(graph_path: Path, mapping_path: Path, output_tsv: Path):
    """Extract triples from a PyTorch HeteroData graph using ID mappings."""
    print(f"\n{'='*70}")
    print(f"Loading graph from: {graph_path.name}")
    print(f"Using mappings from: {mapping_path.name}")
    print(f"{'='*70}")
    
    # Load ID mappings
    try:
        id_mappings = load_id_mappings(mapping_path)
        print(f"✓ Loaded ID mappings for: {list(id_mappings.keys())}")
    except Exception as e:
        print(f"❌ ERROR loading mappings: {e}")
        return
    
    # Load graph
    try:
        graph = torch.load(graph_path, map_location='cpu')
    except Exception as e:
        print(f"❌ ERROR loading graph: {e}")
        return
    
    print(f"✓ Graph type: {type(graph).__name__}")
    print(f"✓ Node types: {graph.node_types}")
    print(f"✓ Edge types: {len(graph.edge_types)}")
    
    # Extract triples
    triples = extract_triples_with_bio_ids(graph, id_mappings, output_tsv)
    
    # Save as TSV
    if triples:
        output_tsv.parent.mkdir(parents=True, exist_ok=True)
        with output_tsv.open('w', encoding='utf-8') as f:
            for h, r, t in triples:
                f.write(f"{h}\t{r}\t{t}\n")
        
        print(f"\n✅ Extracted {len(triples)} triples")
        print(f"✅ Saved to: {output_tsv}")
        
        # Print sample triples
        print(f"\nSample triples:")
        for h, r, t in triples[:5]:
            print(f"  {h}\t{r}\t{t}")
    else:
        print(f"⚠️  No triples extracted")


def main():
    parser = argparse.ArgumentParser(
        description="Extract triples from ProHGT HeteroData using biological ID mappings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_with_mappings.py \\
    --train_graph prothgt-train-graph.pt \\
    --val_graph prothgt-val-graph.pt \\
    --test_graph prothgt-test-graph.pt \\
    --mapping id_samples_train.json \\
    --output_dir ./triples_tsv
        """
    )
    parser.add_argument('--train_graph', type=str, required=True,
                        help='Path to train graph .pt file')
    parser.add_argument('--val_graph', type=str, required=True,
                        help='Path to val graph .pt file')
    parser.add_argument('--test_graph', type=str, required=True,
                        help='Path to test graph .pt file')
    parser.add_argument('--mapping', type=str, required=True,
                        help='Path to id_samples_train.json mapping file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save TSV files')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    print("\n" + "="*70)
    print("EXTRACTING TRIPLES FROM PROTHGT HETERODATA WITH BIOLOGICAL IDs")
    print("="*70)
    
    mapping_path = Path(args.mapping)
    
    extract_triples_from_graph(Path(args.train_graph), mapping_path, output_dir / 'train.tsv')
    extract_triples_from_graph(Path(args.val_graph), mapping_path, output_dir / 'val.tsv')
    extract_triples_from_graph(Path(args.test_graph), mapping_path, output_dir / 'test.tsv')
    
    print(f"\n{'='*70}")
    print(f"✅ All splits extracted with biological IDs")
    print(f"✅ Output directory: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
