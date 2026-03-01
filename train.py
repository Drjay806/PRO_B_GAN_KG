import argparse
import json
from pathlib import Path

from pro_b_gan_kg.training import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prot-B-GAN KG training")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    run_training(config=config, output_dir=output_dir)


if __name__ == "__main__":
    main()
