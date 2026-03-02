"""
Extract triples from ProHGT PyTorch graph files and save as TSV.
"""

import argparse
import torch
from pathlib import Path


def extract_triples_from_graph(graph_path: Path, output_tsv: Path):
    """Extract triples from a PyTorch graph file and save as TSV."""
    print(f"Loading graph from {graph_path}...")
    graph = torch.load(graph_path, map_location='cpu')
    
    # Handle different graph formats
    triples = []
    
    # Check if it's a dict with edge data
    if isinstance(graph, dict):
        print(f"Graph keys: {list(graph.keys())}")
        
        # Try common ProHGT/DGL formats
        if 'edge_index' in graph and 'edge_type' in graph:
            # Format: edge_index (2, num_edges), edge_type (num_edges,)
            edge_index = graph['edge_index']
            edge_type = graph['edge_type']
            
            # Get entity/relation mappings if available
            id2entity = graph.get('id2entity', {})
            id2rel = graph.get('id2rel', {})
            
            for i in range(edge_index.shape[1]):
                h_id = edge_index[0, i].item()
                t_id = edge_index[1, i].item()
                r_id = edge_type[i].item()
                
                h = id2entity.get(h_id, f"entity_{h_id}")
                t = id2entity.get(t_id, f"entity_{t_id}")
                r = id2rel.get(r_id, f"relation_{r_id}")
                
                triples.append((h, r, t))
        
        elif 'triples' in graph:
            # Direct triple format
            triples = graph['triples']
            
        else:
            print(f"Unknown graph format. Available keys: {list(graph.keys())}")
            print(f"Sample data structure:")
            for k, v in list(graph.items())[:5]:
                print(f"  {k}: {type(v)} - {v.shape if hasattr(v, 'shape') else len(v) if hasattr(v, '__len__') else v}")
    
    elif hasattr(graph, 'edge_index') and hasattr(graph, 'edge_type'):
        # PyTorch Geometric Data object
        edge_index = graph.edge_index
        edge_type = graph.edge_type
        
        id2entity = getattr(graph, 'id2entity', {})
        id2rel = getattr(graph, 'id2rel', {})
        
        for i in range(edge_index.shape[1]):
            h_id = edge_index[0, i].item()
            t_id = edge_index[1, i].item()
            r_id = edge_type[i].item()
            
            h = id2entity.get(h_id, f"entity_{h_id}")
            t = id2entity.get(t_id, f"entity_{t_id}")
            r = id2rel.get(r_id, f"relation_{r_id}")
            
            triples.append((h, r, t))
    
    else:
        print(f"Unknown graph type: {type(graph)}")
        print(f"Graph attributes: {dir(graph)}")
        return
    
    # Save as TSV
    if triples:
        output_tsv.parent.mkdir(parents=True, exist_ok=True)
        with output_tsv.open('w', encoding='utf-8') as f:
            for h, r, t in triples:
                f.write(f"{h}\t{r}\t{t}\n")
        print(f"✅ Extracted {len(triples)} triples to {output_tsv}")
    else:
        print(f"❌ No triples extracted")


def main():
    parser = argparse.ArgumentParser(description="Extract triples from ProHGT graph files")
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
    
    extract_triples_from_graph(Path(args.train_graph), output_dir / 'train.tsv')
    extract_triples_from_graph(Path(args.val_graph), output_dir / 'val.tsv')
    extract_triples_from_graph(Path(args.test_graph), output_dir / 'test.tsv')
    
    print(f"\n✅ All splits converted to TSV format in {output_dir}")


if __name__ == '__main__':
    main()
