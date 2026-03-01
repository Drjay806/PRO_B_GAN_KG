import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pro_b_gan_kg.utils import setup_logging


def _first_non_empty(values: List[Optional[str]]) -> str:
    for value in values:
        if value is not None:
            cleaned = value.strip()
            if cleaned:
                return cleaned
    return ""


def _load_tsv_rows(file_path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with file_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    return rows, fieldnames


def _pick_text_column(fieldnames: List[str], preferred: List[str]) -> Optional[str]:
    lowercase_to_original = {name.lower(): name for name in fieldnames}
    for candidate in preferred:
        if candidate.lower() in lowercase_to_original:
            return lowercase_to_original[candidate.lower()]
    return None


def build_metadata_json(tsv_dir: Path, output_json: Path) -> Dict[str, Dict[str, str]]:
    logger = setup_logging(output_json.parent)

    file_specs = {
        "protein": ("proteins.tsv", ["description_full", "description", "description_short", "name"]),
        "go": ("go_terms.tsv", ["definition", "description", "name"]),
        "pathway": ("pathways.tsv", ["description", "definition", "name"]),
        "disease": ("diseases.tsv", ["definition", "description", "name"]),
        "drug": ("drugs.tsv", ["description", "definition", "name"]),
        "side_effect": ("side_effects.tsv", ["definition", "description", "name"]),
        "compound": ("compounds.tsv", ["description", "definition", "name"]),
        "domain": ("domains.tsv", ["description", "definition", "name"]),
        "ec_number": ("ec_numbers.tsv", ["description", "definition", "name"]),
    }

    all_metadata: Dict[str, Dict[str, str]] = {}

    for entity_type, (file_name, preferred_text_columns) in file_specs.items():
        file_path = tsv_dir / file_name
        if not file_path.exists():
            logger.info(f"Skipping {file_name} (not found)")
            continue

        rows, fieldnames = _load_tsv_rows(file_path)
        if not rows:
            logger.info(f"Skipping {file_name} (no rows)")
            continue

        if "entity_id" not in fieldnames:
            logger.warning(f"Skipping {file_name} (missing required column: entity_id)")
            continue

        text_column = _pick_text_column(fieldnames, preferred_text_columns)

        entity_to_text: Dict[str, str] = {}
        skipped_empty = 0

        for row in rows:
            entity_id = _first_non_empty([row.get("entity_id")])
            if not entity_id:
                continue

            if text_column:
                text_value = _first_non_empty([row.get(text_column)])
            else:
                text_value = ""

            if not text_value:
                text_value = _first_non_empty([
                    row.get("name"),
                    row.get("description"),
                    row.get("definition"),
                    row.get("description_full"),
                    row.get("description_short"),
                ])

            if not text_value:
                skipped_empty += 1
                continue

            entity_to_text[entity_id] = text_value

        if entity_to_text:
            all_metadata[entity_type] = entity_to_text

        logger.info(
            f"Loaded {entity_type}: total_rows={len(rows)}, usable={len(entity_to_text)}, "
            f"skipped_empty_text={skipped_empty}, text_column={text_column}"
        )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(all_metadata, handle, indent=2)

    total_entities = sum(len(v) for v in all_metadata.values())
    logger.info(f"Wrote {output_json} with {total_entities} entities across {len(all_metadata)} types")

    return all_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Build all_metadata.json from TSV metadata files")
    parser.add_argument(
        "--tsv_dir",
        type=str,
        required=True,
        help="Directory containing TSV files (proteins.tsv, go_terms.tsv, etc.)",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="Path to output all_metadata.json",
    )
    args = parser.parse_args()

    build_metadata_json(
        tsv_dir=Path(args.tsv_dir),
        output_json=Path(args.output_json),
    )


if __name__ == "__main__":
    main()
