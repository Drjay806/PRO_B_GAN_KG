"""
Extract triples from ProHGT HeteroData with proper biological entity ID mapping.
Maps numeric node IDs back to biological IDs (UniProt, CHEMBL, DOID, GO:, etc.).
"""

import argparse
import torch
from pathlib import Path
from collections import defaultdict


def extract_from_heterodata_with_mapping(hetero_graph, output_tsv: Path):
    """Extract triples from HeteroData using node ID mappings."""
    print(f"✓ Processing HeteroData with node mappings")
    
    triples = []
    
    # Get node mappings for each node type
    # HeteroData stores these as attributes
    node_mappings = {}  # {node_type: {node_idx: biological_id}}
    
    # Check all possible node type mapping attributes
    for node_type in hetero_graph.node_types:
        # Try different possible mapping attribute names
        mapping_attr = None
        for possible_name in [f'{node_type}_mapping', f'{node_type}_id', f'{node_type}_original_id', 
                             f'id2{node_type}', f'{node_type}_2_id']:
            if hasattr(hetero_graph, mapping_attr):
                break
        
        # If no mapping found, try to load from graph data
        if node_type in hetero_graph.node_stores:
            node_store = hetero_graph[node_type]
            # Look for mapping in node store
            for attr in dir(node_store):
                if 'id' in attr.lower() and not attr.startswith('_'):
                    try:
                        val = getattr(node_store, attr)
                        if isinstance(val, dict) and len(val) > 0:
                            node_mappings[node_type] = val
                            print(f"  Found mapping for {node_type}: {len(val)} entries")
                            break
                    except:
                        pass
    
    print(f"\nNode type mappings found: {list(node_mappings.keys())}")
    print(f"Available node types in graph: {hetero_graph.node_types}")
    print(f"Edge types: {hetero_graph.edge_types}")
    
    # Iterate through edge types
    for edge_key in hetero_graph.edge_types:
        if isinstance(edge_key, tuple) and len(edge_key) == 3:
            head_type, rel_type, tail_type = edge_key
        else:
            continue
        
        edge_index = hetero_graph[edge_key].edge_index
        num_edges = edge_index.shape[1]
        
        # Extract triples using mappings if available
        for i in range(num_edges):
            h_idx = edge_index[0, i].item()
            t_idx = edge_index[1, i].item()
            
            # Try to get biological IDs from mappings
            if head_type in node_mappings:
                h = node_mappings[head_type].get(h_idx, f"{head_type}:{h_idx}")
            else:
                h = f"{head_type}:{h_idx}"
            
            if tail_type in node_mappings:
                t = node_mappings[tail_type].get(t_idx, f"{tail_type}:{t_idx}")
            else:
                t = f"{tail_type}:{t_idx}"
            
            triples.append((h, rel_type, t))
    
    return triples


def extract_triples_from_graph(graph_path: Path, output_tsv: Path):
    """Extract triples from a PyTorch graph file and save as TSV."""
    print(f"\n{'='*70}")
    print(f"Loading graph from: {graph_path.name}")
    print(f"{'='*70}")
    
    try:
        graph = torch.load(graph_path, map_location='cpu')
    except Exception as e:
        print(f"❌ ERROR loading graph: {e}")
        return
    
    print(f"Graph type: {type(graph).__name__}")
    triples = []
    
    # Handle HeteroData
    if type(graph).__name__ == 'HeteroData':
        print(f"✓ HeteroData graph detected")
        print(f"  Node types: {graph.node_types}")
        print(f"  Edge types: {len(graph.edge_types)}")
        
        # Try to find node ID mappings
        # Look for common attribute names in HeteroData
        print("\n🔍 Searching for node ID mappings...")
        node_id_maps = {}
        
        for node_type in graph.node_types:
            # Check if there's a mapping stored in the graph
            # HeteroData sometimes stores this as metadata
            if hasattr(graph, '_metadata'):
                print(f"  Found graph metadata")
            
            # Check node store
            try:
                node_data = graph[node_type]
                # Print available attributes
                attrs = [a for a in dir(node_data) if not a.startswith('_')]
                if 'id' in str(attrs).lower():
                    print(f"  {node_type} attrs with 'id': {[a for a in attrs if 'id' in a.lower()]}")
            except:
                pass
        
        print("\n⚠️  Node ID mappings not found in graph metadata")
        print("Creating generic numeric mappings...")
        
        # Extract with generic numeric IDs
        for edge_key in graph.edge_types:
            if isinstance(edge_key, tuple) and len(edge_key) == 3:
                head_type, rel_type, tail_type = edge_key
                edge_index = graph[edge_key].edge_index
                
                for i in range(edge_index.shape[1]):
                    h_idx = edge_index[0, i].item()
                    t_idx = edge_index[1, i].item()
                    
                    # Use format: Type:numeric_id
                    h = f"{head_type}:{h_idx}"
                    t = f"{tail_type}:{t_idx}"
                    
                    triples.append((h, rel_type, t))
        
        print(f"\n⚠️  Using generic numeric node IDs (Protein:123, etc.)")
        print(f"This will NOT match biological IDs in embeddings!")
        print(f"You may need to:")
        print(f"  1. Check if HeteroData has node mapping attributes")
        print(f"  2. Use original biological IDs to re-extract triples")
        print(f"  3. Re-create embeddings with numeric IDs")
    
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
        description="Extract triples from ProHGT HeteroData with biological ID mapping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_triples_from_heterodata.py \\
    --train_graph train-graph.pt \\
    --val_graph val-graph.pt \\
    --test_graph test-graph.pt \\
    --output_dir ./triples_tsv
        """
    )
    parser.add_argument('--train_graph', type=str, required=True,
                        help='Path to train graph .pt file')
    parser.add_argument('--val_graph', type=str, required=True,
                        help='Path to val graph .pt file')
    parser.add_argument('--test_graph', type=str, required=True,
                        help='Path to test graph .pt file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save TSV files')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    print("\n" + "="*70)
    print("EXTRACTING TRIPLES FROM PROTHGT HETERODATA")
    print("="*70)
    
    extract_triples_from_graph(Path(args.train_graph), output_dir / 'train.tsv')
    extract_triples_from_graph(Path(args.val_graph), output_dir / 'val.tsv')
    extract_triples_from_graph(Path(args.test_graph), output_dir / 'test.tsv')
    
    print(f"\n{'='*70}")
    print(f"WARNING: Check if triples use biological IDs or numeric IDs!")
    print(f"If numeric IDs, embeddings won't match - need to fix mapping")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
