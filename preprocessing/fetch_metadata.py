import argparse
import logging
from pathlib import Path
from typing import Dict, Set

from pro_b_gan_kg.data import load_json
from pro_b_gan_kg.utils import save_json, setup_logging

from .metadata_fetchers import (
    DiseaseFetcher,
    GOTermFetcher,
    PathwayFetcher,
    ProteinFetcher,
)


def extract_entities_by_type(entity2id: Dict[str, int]) -> Dict[str, Set[str]]:
    by_type = {}
    for entity_name in entity2id.keys():
        if entity_name.startswith("protein"):
            entity_type = "protein"
        elif entity_name.startswith("go"):
            entity_type = "go"
        elif entity_name.startswith("pathway"):
            entity_type = "pathway"
        elif entity_name.startswith("disease"):
            entity_type = "disease"
        elif entity_name.startswith("side_effect"):
            entity_type = "side_effect"
        else:
            entity_type = "other"

        if entity_type not in by_type:
            by_type[entity_type] = set()
        by_type[entity_type].add(entity_name)

    return by_type


def fetch_all_metadata(
    entity2id_path: Path,
    output_dir: Path,
) -> None:
    logger = setup_logging(output_dir)
    logger.info("Starting metadata fetching")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(exist_ok=True)

    entity2id = load_json(entity2id_path)
    logger.info(f"Loaded {len(entity2id)} entities")

    by_type = extract_entities_by_type(entity2id)
    logger.info(f"Entity types: {list(by_type.keys())}")

    all_metadata = {}

    if "go" in by_type:
        logger.info(f"Fetching definitions for {len(by_type['go'])} GO terms...")
        go_fetcher = GOTermFetcher(cache_dir / "go_cache.json")
        go_list = sorted(list(by_type["go"]))
        go_metadata = go_fetcher.fetch_batch(go_list)
        go_fetcher.save_cache()
        all_metadata["go"] = go_metadata
        logger.info(f"Fetched {len(go_metadata)} GO term definitions")

    if "pathway" in by_type:
        logger.info(f"Fetching descriptions for {len(by_type['pathway'])} pathways...")
        pathway_fetcher = PathwayFetcher(cache_dir / "pathway_cache.json")
        pathway_list = sorted(list(by_type["pathway"]))
        pathway_metadata = pathway_fetcher.fetch_batch(pathway_list)
        pathway_fetcher.save_cache()
        all_metadata["pathway"] = pathway_metadata
        logger.info(f"Fetched {len(pathway_metadata)} pathway descriptions")

    if "disease" in by_type:
        logger.info(f"Fetching descriptions for {len(by_type['disease'])} diseases...")
        disease_fetcher = DiseaseFetcher(cache_dir / "disease_cache.json")
        disease_list = sorted(list(by_type["disease"]))
        disease_metadata = disease_fetcher.fetch_batch(disease_list)
        disease_fetcher.save_cache()
        all_metadata["disease"] = disease_metadata
        logger.info(f"Fetched {len(disease_metadata)} disease descriptions")

    if "protein" in by_type:
        logger.info(f"Fetching info for {len(by_type['protein'])} proteins...")
        protein_fetcher = ProteinFetcher(cache_dir / "protein_cache.json")
        protein_list = sorted(list(by_type["protein"]))
        protein_metadata = protein_fetcher.fetch_batch(protein_list)
        protein_fetcher.save_cache()
        all_metadata["protein"] = protein_metadata
        logger.info(f"Fetched {len(protein_metadata)} protein descriptions")

    save_json(all_metadata, output_dir / "all_metadata.json")
    logger.info("Metadata fetching complete")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch metadata for entities from public databases")
    parser.add_argument("--entity2id_path", type=str, required=True, help="Path to entity2id.json")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for metadata and caches")
    args = parser.parse_args()

    fetch_all_metadata(
        entity2id_path=Path(args.entity2id_path),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
