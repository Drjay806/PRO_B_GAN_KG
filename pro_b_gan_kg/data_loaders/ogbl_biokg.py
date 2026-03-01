import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class OGBLBioKGLoader:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.entities: Set[str] = set()
        self.relations: Set[str] = set()

    def load_triples(self, split: str = "train") -> List[Tuple[str, str, str]]:
        triple_file = self.data_dir / f"{split}_triples.txt"
        if not triple_file.exists():
            logger.warning(f"File not found: {triple_file}")
            return []

        triples = []
        with triple_file.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    h, r, t = parts[0], parts[1], parts[2]
                    triples.append((h, r, t))
                    self.entities.add(h)
                    self.entities.add(t)
                    self.relations.add(r)

        logger.info(f"Loaded {len(triples)} triples from {split}")
        return triples

    def load_all_splits(self) -> Dict[str, List[Tuple[str, str, str]]]:
        data = {}
        for split in ["train", "val", "test"]:
            data[split] = self.load_triples(split)
        return data

    def get_entity_type(self, entity_id: str) -> Optional[str]:
        if entity_id.startswith("protein"):
            return "protein"
        elif entity_id.startswith("go"):
            return "go"
        elif entity_id.startswith("pathway"):
            return "pathway"
        elif entity_id.startswith("disease"):
            return "disease"
        elif entity_id.startswith("side_effect"):
            return "side_effect"
        else:
            return None

    def split_by_type(
        self, triples: List[Tuple[str, str, str]]
    ) -> Dict[str, List[Tuple[str, str, str]]]:
        by_type = {}
        for h, r, t in triples:
            h_type = self.get_entity_type(h)
            t_type = self.get_entity_type(t)
            key = f"{h_type}-{t_type}"
            if key not in by_type:
                by_type[key] = []
            by_type[key].append((h, r, t))
        return by_type
