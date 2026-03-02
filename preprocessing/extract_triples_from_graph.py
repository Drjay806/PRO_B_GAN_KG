"""
Extract triples from ProHGT PyTorch graph files and save as TSV.
Supports HeteroData (PyTorch Geometric heterogeneous graphs).
"""

import argparse
import torch
from pathlib import Path


def extract_from_heterodata(hetero_graph, output_tsv: Path):
    """Extract triples from a PyTorch Geometric HeteroData object."""
    print(f"✓ Processing HeteroData with edge types:")
    triples = []
    
    # Get node type mappings if available
    node_mappings = {}
    
    # Check for node mappings in metadata
    if hasattr(hetero_graph, 'node_id_mapping'):
        node_mappings = hetero_graph.node_id_mapping
        print(f"  Found node_id_mapping with {len(node_mappings)} entries")
    
    # Iterate through all edge types
    for edge_key in hetero_graph.edge_types:
        print(f"  Processing edge type: {edge_key}")
        
        edge_index = hetero_graph[edge_key].edge_index
        num_edges = edge_index.shape[1]
        print(f"    → {num_edges} edges")
        
        # Extract head, relation, tail from edge key
        if isinstance(edge_key, tuple) and len(edge_key) == 3:
            head_type, rel_type, tail_type = edge_key
        else:
            print(f"    ⚠️  Unexpected edge_key format: {edge_key}")
            continue
        
        # Extract triples
        for i in range(num_edges):
            h_id = edge_index[0, i].item()
            t_id = edge_index[1, i].item()
            
            # Create entity names from type + id (ProHGT convention)
            # You may need to adjust these prefixes based on your actual data
            head = f"{head_type}:{h_id}"
            tail = f"{tail_type}:{t_id}"
            
            triples.append((head, rel_type, tail))
    
    # Save as TSV
    if triples:
        output_tsv.parent.mkdir(parents=True, exist_ok=True)
        with output_tsv.open('w', encoding='utf-8') as f:
            for h, r, t in triples:
                f.write(f"{h}\t{r}\t{t}\n")
        print(f"✅ Extracted {len(triples)} triples")
        print(f"✅ Saved to: {output_tsv}")
        return True
    else:
        print(f"⚠️  No triples extracted")
        return False


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
    
    # Case 1: HeteroData (PyG heterogeneous graph)
    if type(graph).__name__ == 'HeteroData':
        extract_from_heterodata(graph, output_tsv)
        return
    
    triples = []
    
    # Case 2: Dict with edge_index and edge_type
    if isinstance(graph, dict):
        print(f"Graph keys: {list(graph.keys())}")
        
        if 'edge_index' in graph and 'edge_type' in graph:
            print("✓ Found edge_index and edge_type")
            edge_index = graph['edge_index']
            edge_type = graph['edge_type']
            
            id2entity = graph.get('id2entity', {})
            id2rel = graph.get('id2rel', {})
            
            print(f"  edge_index shape: {edge_index.shape if hasattr(edge_index, 'shape') else 'N/A'}")
            print(f"  edge_type shape: {edge_type.shape if hasattr(edge_type, 'shape') else 'N/A'}")
            print(f"  id2entity: {len(id2entity)} entries")
            print(f"  id2rel: {len(id2rel)} entries")
            
            if not isinstance(edge_index, torch.Tensor):
                edge_index = torch.tensor(edge_index)
            if not isinstance(edge_type, torch.Tensor):
                edge_type = torch.tensor(edge_type)
            
            num_edges = edge_index.shape[1] if len(edge_index.shape) > 1 else len(edge_index) // 2
            
            for i in range(num_edges):
                h_id = edge_index[0, i].item() if len(edge_index.shape) > 1 else edge_index[2*i].item()
                t_id = edge_index[1, i].item() if len(edge_index.shape) > 1 else edge_index[2*i+1].item()
                r_id = edge_type[i].item()
                
                h = id2entity.get(h_id, f"entity_{h_id}") if id2entity else f"entity_{h_id}"
                t = id2entity.get(t_id, f"entity_{t_id}") if id2entity else f"entity_{t_id}"
                r = id2rel.get(r_id, f"relation_{r_id}") if id2rel else f"relation_{r_id}"
                
                triples.append((h, r, t))
        
        elif 'triples' in graph:
            print("✓ Found 'triples' key")
            triples = graph['triples']
        
        else:
            print(f"❌ Unknown dict format. Keys: {list(graph.keys())}")
            return
    
    # Case 3: DGL/PyG Data object
    elif hasattr(graph, 'edge_index'):
        print("✓ Data object with edge_index")
        edge_index = graph.edge_index
        
        if hasattr(graph, 'edge_type'):
            edge_type = graph.edge_type
        elif hasattr(graph, 'edge_attr'):
            edge_type = graph.edge_attr
        else:
            print("❌ No edge_type/edge_attr found")
            return
        
        num_edges = edge_index.shape[1]
        
        for i in range(num_edges):
            h_id = edge_index[0, i].item()
            t_id = edge_index[1, i].item()
            r_id = edge_type[i].item() if hasattr(edge_type[i], 'item') else edge_type[i]
            triples.append((f"entity_{h_id}", f"relation_{r_id}", f"entity_{t_id}"))
    
    else:
        print(f"❌ Unknown graph type: {type(graph).__name__}")
        return
    
    # Save as TSV
    if triples:
        output_tsv.parent.mkdir(parents=True, exist_ok=True)
        with output_tsv.open('w', encoding='utf-8') as f:
            for h, r, t in triples:
                f.write(f"{h}\t{r}\t{t}\n")
        print(f"✅ Extracted {len(triples)} triples")
        print(f"✅ Saved to: {output_tsv}")
    else:
        print(f"⚠️  No triples extracted")


def main():
    parser = argparse.ArgumentParser(
        description="Extract triples from ProHGT graph files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_triples_from_graph.py \\
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
    print("EXTRACTING TRIPLES FROM PROTHGT GRAPH FILES")
    print("="*70)
    
    extract_triples_from_graph(Path(args.train_graph), output_dir / 'train.tsv')
    extract_triples_from_graph(Path(args.val_graph), output_dir / 'val.tsv')
    extract_triples_from_graph(Path(args.test_graph), output_dir / 'test.tsv')
    
    print(f"\n{'='*70}")
    print(f"✅ All splits converted to TSV format")
    print(f"✅ Output directory: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()


