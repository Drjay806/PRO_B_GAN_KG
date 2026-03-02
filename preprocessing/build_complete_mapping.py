"""
Build complete node ID mapping from ProHGT metadata TSV files.
Maps numeric node indices to biological IDs (UniProt, CHEMBL, etc.)
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from collections import OrderedDict


def load_tsv_ids(tsv_path: Path, id_column: str = 'entity_id') -> list:
    """Load entity IDs from TSV file in order."""
    try:
        df = pd.read_csv(tsv_path, sep='\t', encoding='utf-8')
        if id_column in df.columns:
            ids = df[id_column].tolist()
            print(f"  ✓ Loaded {len(ids)} IDs from {tsv_path.name}")
            return ids
        else:
            print(f"  ⚠️  No '{id_column}' column in {tsv_path.name}")
            print(f"     Available columns: {list(df.columns)}")
            return []
    except Exception as e:
        print(f"  ❌ Error loading {tsv_path.name}: {e}")
        return []


def build_complete_mapping(metadata_dir: Path, output_path: Path):
    """Build complete node ID mapping from TSV metadata files."""
    print(f"\n{'='*70}")
    print(f"Building complete node ID mapping")
    print(f"From: {metadata_dir}")
    print(f"{'='*70}\n")
    
    # Map TSV filenames to node types
    tsv_to_node_type = {
        'protein_metadata.tsv': 'Protein',
        'compound_metadata.tsv': 'Compound',
        'drug_metadata.tsv': 'Drug',
        'disease_metadata.tsv': 'Disease',
        'go_metadata.tsv': ['GO_term_F', 'GO_term_P', 'GO_term_C'],  # GO terms split by aspect
        'pathway_metadata.tsv': 'Pathway',
        'kegg_pathway_metadata.tsv': 'kegg_Pathway',
        'side_effect_metadata.tsv': 'Side_effect',
        'domain_metadata.tsv': 'Domain',
        'ec_number_metadata.tsv': 'EC_number',
        'hpo_metadata.tsv': 'HPO',
    }
    
    complete_mapping = {}
    
    for tsv_file, node_type in tsv_to_node_type.items():
        tsv_path = metadata_dir / tsv_file
        
        if not tsv_path.exists():
            print(f"⚠️  {tsv_file} not found, skipping")
            continue
        
        print(f"Processing {tsv_file}...")
        
        # Special handling for GO terms (split by aspect)
        if isinstance(node_type, list):
            df = pd.read_csv(tsv_path, sep='\t', encoding='utf-8')
            if 'entity_id' in df.columns and 'aspect' in df.columns:
                for aspect_type in node_type:
                    aspect_char = aspect_type.split('_')[-1]  # F, P, or C
                    aspect_df = df[df['aspect'] == aspect_char]
                    ids = aspect_df['entity_id'].tolist()
                    complete_mapping[aspect_type] = ids
                    print(f"  ✓ {aspect_type}: {len(ids)} IDs")
            else:
                print(f"  ⚠️  Could not split GO terms by aspect")
        else:
            ids = load_tsv_ids(tsv_path)
            if ids:
                complete_mapping[node_type] = ids
    
    # Save mapping
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(complete_mapping, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✅ Complete mapping saved to: {output_path}")
    print(f"\nSummary:")
    total_entities = 0
    for node_type, ids in complete_mapping.items():
        print(f"  {node_type:15s}: {len(ids):7d} entities")
        total_entities += len(ids)
    print(f"  {'TOTAL':15s}: {total_entities:7d} entities")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Build complete node ID mapping from metadata TSVs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python build_complete_mapping.py \\
    --metadata_dir /path/to/prothgt/for_the_model \\
    --output id_mapping_complete.json
        """
    )
    parser.add_argument('--metadata_dir', type=str, required=True,
                        help='Directory containing metadata TSV files')
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSON file for complete mapping')
    
    args = parser.parse_args()
    
    build_complete_mapping(
        metadata_dir=Path(args.metadata_dir),
        output_path=Path(args.output)
    )


if __name__ == '__main__':
    main()
