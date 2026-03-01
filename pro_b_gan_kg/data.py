import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .utils import save_json


Triple = Tuple[str, str, str]
TripleIds = Tuple[int, int, int]


@dataclass
class MappingArtifacts:
    entity2id: Dict[str, int]
    rel2id: Dict[str, int]


@dataclass
class IdTriples:
    train: List[TripleIds]
    val: List[TripleIds]
    test: List[TripleIds]


class NeighborCache:
    def __init__(self, pairs: Dict[Tuple[int, int], List[int]]) -> None:
        self.pairs = pairs

    def get(self, h_id: int, r_id: int) -> List[int]:
        return self.pairs.get((h_id, r_id), [])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.pairs, allow_pickle=True)

    @staticmethod
    def load(path: Path) -> "NeighborCache":
        data = np.load(path, allow_pickle=True).item()
        return NeighborCache(data)


def _detect_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        sample = f.readline()
    return "\t" if "\t" in sample else ","


def read_triples(path: Path, delimiter: Optional[str], has_header: bool) -> List[Triple]:
    delim = delimiter or _detect_delimiter(path)
    triples: List[Triple] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delim)
        if has_header:
            next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            h, r, t = row[0].strip(), row[1].strip(), row[2].strip()
            triples.append((h, r, t))
    return triples


def build_mappings(triples: Iterable[Triple]) -> MappingArtifacts:
    entities = {}
    relations = {}
    for h, r, t in triples:
        if h not in entities:
            entities[h] = len(entities)
        if t not in entities:
            entities[t] = len(entities)
        if r not in relations:
            relations[r] = len(relations)
    return MappingArtifacts(entity2id=entities, rel2id=relations)


def to_ids(triples: Iterable[Triple], mappings: MappingArtifacts) -> List[TripleIds]:
    id_triples = []
    for h, r, t in triples:
        id_triples.append((mappings.entity2id[h], mappings.rel2id[r], mappings.entity2id[t]))
    return id_triples


def save_id_triples(path: Path, triples: Iterable[TripleIds]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for h, r, t in triples:
            f.write(f"{h}\t{r}\t{t}\n")


def save_mappings(output_dir: Path, mappings: MappingArtifacts) -> None:
    save_json(mappings.entity2id, output_dir / "entity2id.json")
    save_json(mappings.rel2id, output_dir / "rel2id.json")


def build_neighbor_cache(train_ids: List[TripleIds]) -> NeighborCache:
    pairs: Dict[Tuple[int, int], List[int]] = {}
    for h, r, t in train_ids:
        pairs.setdefault((h, r), []).append(t)
    return NeighborCache(pairs)


def load_and_prepare(
    train_path: Path,
    val_path: Path,
    test_path: Path,
    delimiter: Optional[str],
    has_header: bool,
    output_dir: Path,
) -> Tuple[MappingArtifacts, IdTriples]:
    train = read_triples(train_path, delimiter, has_header)
    val = read_triples(val_path, delimiter, has_header)
    test = read_triples(test_path, delimiter, has_header)

    mappings = build_mappings(train + val + test)
    train_ids = to_ids(train, mappings)
    val_ids = to_ids(val, mappings)
    test_ids = to_ids(test, mappings)

    save_mappings(output_dir, mappings)
    save_id_triples(output_dir / "train.tsv", train_ids)
    save_id_triples(output_dir / "val.tsv", val_ids)
    save_id_triples(output_dir / "test.tsv", test_ids)

    return mappings, IdTriples(train=train_ids, val=val_ids, test=test_ids)
