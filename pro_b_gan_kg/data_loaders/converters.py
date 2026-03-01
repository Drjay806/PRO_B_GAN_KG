import logging
from pathlib import Path
from typing import Dict, List, Tuple

from ..data import MappingArtifacts, save_id_triples, save_mappings

logger = logging.getLogger(__name__)


class KGConverter:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def convert_to_ids(
        self, triples: Dict[str, List[Tuple[str, str, str]]], mappings: MappingArtifacts
    ) -> Dict[str, List[Tuple[int, int, int]]]:
        id_triples = {}
        for split_name, split_triples in triples.items():
            id_split = []
            for h, r, t in split_triples:
                if h in mappings.entity2id and r in mappings.rel2id and t in mappings.entity2id:
                    h_id = mappings.entity2id[h]
                    r_id = mappings.rel2id[r]
                    t_id = mappings.entity2id[t]
                    id_split.append((h_id, r_id, t_id))
                else:
                    missing = []
                    if h not in mappings.entity2id:
                        missing.append(f"entity {h}")
                    if r not in mappings.rel2id:
                        missing.append(f"relation {r}")
                    if t not in mappings.entity2id:
                        missing.append(f"entity {t}")
                    logger.debug(f"Skipping triple: {missing}")

            id_triples[split_name] = id_split
            logger.info(f"Converted {len(id_split)} / {len(split_triples)} {split_name} triples")

        return id_triples

    def save_converted_data(
        self, id_triples: Dict[str, List[Tuple[int, int, int]]], mappings: MappingArtifacts
    ) -> None:
        save_mappings(self.output_dir, mappings)
        for split_name, split_triples in id_triples.items():
            save_id_triples(self.output_dir / f"{split_name}.tsv", split_triples)
        logger.info(f"Saved converted data to {self.output_dir}")
