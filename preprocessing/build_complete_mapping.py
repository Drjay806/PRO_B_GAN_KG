"""
Build complete node ID mapping from ProHGT metadata TSV files.
Maps numeric node indices to biological IDs (UniProt, CHEMBL, etc.)
"""

import argparse
import json
import pandas as pd
from pathlib import Path


def _detect_id_column(df: pd.DataFrame) -> str | None:
    preferred = ["entity_id", "id", "uniprot_id", "go_id", "pathway_id", "disease_id"]
    for col in preferred:
        if col in df.columns:
            return col
    for col in df.columns:
        lower = col.lower()
        if lower.endswith("_id") or lower == "id":
            return col
    return None


def load_tsv_ids(tsv_path: Path) -> list:
    """Load entity IDs from TSV file in row order."""
    try:
        df = pd.read_csv(tsv_path, sep='\t', encoding='utf-8')
        id_column = _detect_id_column(df)
        if id_column:
            ids = [str(x) for x in df[id_column].tolist() if pd.notna(x)]
            print(f"  ✓ Loaded {len(ids)} IDs from {tsv_path.name}")
            return ids
        else:
            print(f"  ⚠️  Could not detect ID column in {tsv_path.name}")
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
    
    # Map actual ProHGT TSV filenames to node types
    tsv_to_node_type = {
        'proteins.tsv': 'Protein',
        'compounds.tsv': 'Compound',
        'drugs.tsv': 'Drug',
        'diseases.tsv': 'Disease',
        'go_terms.tsv': ['GO_term_F', 'GO_term_P', 'GO_term_C'],
        'pathways.tsv': ['Pathway', 'kegg_Pathway'],
        'side_effects.tsv': 'HPO',
        'domains.tsv': 'Domain',
        'ec_numbers.tsv': 'EC_number',
    }
    
    complete_mapping = {}
    
    for tsv_file, node_type in tsv_to_node_type.items():
        tsv_path = metadata_dir / tsv_file
        
        if not tsv_path.exists():
            print(f"⚠️  {tsv_file} not found, skipping")
            continue
        
        print(f"Processing {tsv_file}...")
        
        # Special handling for GO terms split by aspect F/P/C
        if tsv_file == 'go_terms.tsv':
            df = pd.read_csv(tsv_path, sep='\t', encoding='utf-8')
            id_col = _detect_id_column(df)
            aspect_col = 'aspect' if 'aspect' in df.columns else None
            if id_col and aspect_col:
                for aspect_type in ['GO_term_F', 'GO_term_P', 'GO_term_C']:
                    aspect_char = aspect_type.split('_')[-1]
                    aspect_df = df[df[aspect_col].astype(str).str.upper() == aspect_char]
                    ids = [str(x) for x in aspect_df[id_col].tolist() if pd.notna(x)]
                    complete_mapping[aspect_type] = ids
                    print(f"  ✓ {aspect_type}: {len(ids)} IDs")
            else:
                print(f"  ⚠️  Could not split GO terms by aspect in {tsv_file}")
                print(f"     Available columns: {list(df.columns)}")

        # Special handling for Pathway vs kegg_Pathway split
        elif tsv_file == 'pathways.tsv':
            df = pd.read_csv(tsv_path, sep='\t', encoding='utf-8')
            id_col = _detect_id_column(df)
            if not id_col:
                print(f"  ⚠️  Could not detect ID column for pathways.tsv")
                continue

            source_col = None
            for candidate in ['source', 'database', 'db', 'pathway_source', 'type']:
                if candidate in df.columns:
                    source_col = candidate
                    break

            if source_col:
                source_series = df[source_col].astype(str).str.lower()
                kegg_df = df[source_series.str.contains('kegg', na=False)]
                non_kegg_df = df[~source_series.str.contains('kegg', na=False)]
            else:
                id_series = df[id_col].astype(str)
                kegg_mask = id_series.str.contains('kegg|map\d+|hsa\d+', case=False, regex=True)
                kegg_df = df[kegg_mask]
                non_kegg_df = df[~kegg_mask]

            complete_mapping['Pathway'] = [str(x) for x in non_kegg_df[id_col].tolist() if pd.notna(x)]
            complete_mapping['kegg_Pathway'] = [str(x) for x in kegg_df[id_col].tolist() if pd.notna(x)]
            print(f"  ✓ Pathway: {len(complete_mapping['Pathway'])} IDs")
            print(f"  ✓ kegg_Pathway: {len(complete_mapping['kegg_Pathway'])} IDs")

        # Special handling for HPO (if file uses HP:* terms)
        elif tsv_file == 'side_effects.tsv':
            ids = load_tsv_ids(tsv_path)
            if ids:
                hpo_like = [entity_id for entity_id in ids if str(entity_id).startswith('HP:')]
                if hpo_like:
                    complete_mapping['HPO'] = hpo_like
                    print(f"  ✓ HPO: {len(hpo_like)} IDs (from side_effects.tsv)")
                else:
                    complete_mapping['HPO'] = ids
                    print(f"  ✓ HPO: {len(ids)} IDs (fallback from side_effects.tsv)")

        elif isinstance(node_type, list):
            # Any future list-based split files not explicitly handled
            ids = load_tsv_ids(tsv_path)
            if ids:
                complete_mapping[node_type[0]] = ids
                print(f"  ✓ {node_type[0]}: {len(ids)} IDs (unsplit fallback)")

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
