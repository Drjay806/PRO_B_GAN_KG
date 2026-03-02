"""
Diagnostic tool to inspect ProHGT graph file structure.
Run this to understand the format before extraction.
"""

import argparse
import torch
from pathlib import Path


def inspect_graph(graph_path: Path):
    """Inspect what's inside a graph .pt file."""
    print(f"\n{'='*70}")
    print(f"Inspecting: {graph_path.name}")
    print(f"{'='*70}")
    
    try:
        graph = torch.load(graph_path, map_location='cpu')
    except Exception as e:
        print(f"❌ ERROR loading {graph_path.name}: {e}")
        return
    
    print(f"Top-level type: {type(graph).__name__}\n")
    
    # Case 1: Dictionary
    if isinstance(graph, dict):
        print(f"📦 Dictionary with {len(graph)} keys:")
        for key in sorted(graph.keys()):
            value = graph[key]
            
            if isinstance(value, dict):
                print(f"  ├─ {key:20s}: dict with {len(value)} items")
                if value:
                    first_k, first_v = next(iter(value.items()))
                    print(f"  │   └─ Example: {first_k} → {type(first_v).__name__}")
            
            elif isinstance(value, torch.Tensor):
                print(f"  ├─ {key:20s}: Tensor shape {value.shape}, dtype {value.dtype}")
            
            elif isinstance(value, (list, tuple)):
                print(f"  ├─ {key:20s}: {type(value).__name__} with {len(value)} items")
                if value:
                    print(f"  │   └─ First item type: {type(value[0]).__name__}")
            
            else:
                print(f"  ├─ {key:20s}: {type(value).__name__}")
    
    # Case 2: PyTorch Geometric Data object
    elif hasattr(graph, 'edge_index'):
        print(f"🔗 PyG Data object")
        print(f"  Attributes:")
        for attr in dir(graph):
            if not attr.startswith('_'):
                try:
                    value = getattr(graph, attr)
                    if callable(value):
                        continue
                    if isinstance(value, torch.Tensor):
                        print(f"    ├─ {attr:20s}: Tensor {value.shape}")
                    elif isinstance(value, dict):
                        print(f"    ├─ {attr:20s}: dict with {len(value)} items")
                    else:
                        print(f"    ├─ {attr:20s}: {type(value).__name__}")
                except:
                    pass
    
    # Case 3: List/Tuple of tensors
    elif isinstance(graph, (list, tuple)):
        print(f"📋 {type(graph).__name__} with {len(graph)} items:")
        for i, item in enumerate(graph[:5]):
            if isinstance(item, torch.Tensor):
                print(f"  [{i}] Tensor: shape {item.shape}, dtype {item.dtype}")
            elif isinstance(item, (list, tuple)):
                print(f"  [{i}] {type(item).__name__}: {len(item)} items")
            else:
                print(f"  [{i}] {type(item).__name__}: {item}")
        if len(graph) > 5:
            print(f"  ... and {len(graph) - 5} more items")
    
    # Case 4: Single tensor
    elif isinstance(graph, torch.Tensor):
        print(f"📊 Single Tensor: shape {graph.shape}, dtype {graph.dtype}")
    
    else:
        print(f"❓ Unknown type: {type(graph).__name__}")
        print(f"   Attributes: {[x for x in dir(graph) if not x.startswith('_')][:15]}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect ProHGT graph file structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inspect_graph.py --file train-graph.pt
  python inspect_graph.py --file /path/to/prothgt-train-graph.pt
        """
    )
    parser.add_argument('--file', type=str, required=True,
                        help='Path to graph .pt file')
    
    args = parser.parse_args()
    inspect_graph(Path(args.file))


if __name__ == '__main__':
    main()
