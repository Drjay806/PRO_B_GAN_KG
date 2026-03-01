import argparse
from pathlib import Path

from pro_b_gan_kg.data_loaders.converters import KGConverter
from pro_b_gan_kg.data_loaders.ogbl_biokg import OGBLBioKGLoader
from pro_b_gan_kg.data_loaders.text_extractors import BioKGTextExtractor
from pro_b_gan_kg.utils import setup_logging


def prepare_ogbl_biokg(
    raw_data_path: Path,
    output_dir: Path,
) -> None:
    logger = setup_logging(output_dir)
    logger.info("Preparing OGBL-BioKG dataset")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Step 1: Loading raw triples...")
    loader = OGBLBioKGLoader(raw_data_path)
    train_triples, val_triples, test_triples = loader.load()
    logger.info(f"  Train: {len(train_triples)}, Val: {len(val_triples)}, Test: {len(test_triples)}")

    logger.info("Step 2: Extracting entity text descriptions...")
    extractor = BioKGTextExtractor()
    text_map = extractor.extract_all_texts(train_triples + val_triples + test_triples)
    logger.info(f"  Extracted descriptions for {len(text_map)} entities")

    logger.info("Step 3: Converting to ID format...")
    converter = KGConverter()
    converter.convert(
        train_triples=train_triples,
        val_triples=val_triples,
        test_triples=test_triples,
        text_descriptions=text_map,
        output_dir=output_dir,
    )
    logger.info("  Conversion complete")
    logger.info(f"  Output files:")
    logger.info(f"    - entity2id.json")
    logger.info(f"    - rel2id.json")
    logger.info(f"    - train.tsv")
    logger.info(f"    - val.tsv")
    logger.info(f"    - test.tsv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare OGBL-BioKG dataset for training")
    parser.add_argument("--raw_data_path", type=str, required=True, help="Path to raw OGBL-BioKG data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for processed data")
    args = parser.parse_args()

    prepare_ogbl_biokg(
        raw_data_path=Path(args.raw_data_path),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
