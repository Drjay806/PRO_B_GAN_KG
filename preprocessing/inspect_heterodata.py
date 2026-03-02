"""
Inspect HeteroData structure to find node ID mappings.
"""

import torch
from pathlib import Path


def inspect_heterodata(graph_path: Path):
    """Deep inspection of HeteroData to find biological ID mappings."""
    print(f"\n{'='*70}")
    print(f"Inspecting: {graph_path.name}")
    print(f"{'='*70}")
    
    graph = torch.load(graph_path, map_location='cpu')
    
    print(f"\nGraph type: {type(graph).__name__}")
    print(f"Node types: {graph.node_types}")
    print(f"Edge types count: {len(graph.edge_types)}")
    
    # Check for metadata/mappings at graph level
    print(f"\n🔍 Top-level attributes:")
    attrs = [a for a in dir(graph) if not a.startswith('__')]
    for attr in attrs[:20]:
        try:
            val = getattr(graph, attr)
            if not callable(val):
                if isinstance(val, dict):
                    print(f"  {attr}: dict({len(val)} items)")
                elif isinstance(val, (list, tuple)):
                    print(f"  {attr}: {type(val).__name__}({len(val)} items)")
                elif hasattr(val, 'shape'):
                    print(f"  {attr}: Tensor{val.shape}")
                else:
                    print(f"  {attr}: {type(val).__name__}")
        except:
            pass
    
    # Check each node type store
    print(f"\n🔍 Node type stores:")
    for node_type in graph.node_types:
        node_store = graph[node_type]
        num_nodes = node_store.num_nodes
        print(f"\n  {node_type}:")
        print(f"    Nodes: {num_nodes}")
        
        # List attributes
        attrs = [a for a in dir(node_store) if not a.startswith('_')]
        relevant_attrs = [a for a in attrs if 'id' in a.lower() or 'name' in a.lower() or 'mapping' in a.lower()]
        
        if relevant_attrs:
            print(f"    Potential mapping attributes: {relevant_attrs}")
            for attr in relevant_attrs[:3]:
                try:
                    val = getattr(node_store, attr)
                    if isinstance(val, dict):
                        print(f"      {attr}: dict({len(val)} items)")
                        # Show first 3 entries
                        for k, v in list(val.items())[:3]:
                            print(f"        {k} → {v}")
                    elif isinstance(val, (list, tuple)):
                        print(f"      {attr}: {type(val).__name__}({len(val)} items)")
                        print(f"        First 3: {val[:3]}")
                except:
                    pass
        else:
            print(f"    No mapping found (attrs: {attrs[:10]})")
    
    # Check first edge type for structure
    print(f"\n🔍 Sample edge data:")
    first_edge = graph.edge_types[0]
    h_type, rel, t_type = first_edge
    edge_index = graph[first_edge].edge_index
    
    print(f"  Edge type: {first_edge}")
    print(f"  edge_index shape: {edge_index.shape}")
    print(f"  Sample edges (first 5):")
    for i in range(min(5, edge_index.shape[1])):
        h_idx = edge_index[0, i].item()
        t_idx = edge_index[1, i].item()
        print(f"    {h_type}:{h_idx} --{rel}--> {t_type}:{t_idx}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Inspect HeteroData structure")
    parser.add_argument('--file', type=str, required=True, help='Path to graph .pt file')
    args = parser.parse_args()
    
    inspect_heterodata(Path(args.file))
