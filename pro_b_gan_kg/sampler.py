import random
from typing import Dict, List, Tuple

import torch

TripleIds = Tuple[int, int, int]


class NegativeSampler:
    def __init__(
        self,
        num_entities: int,
        train_pairs: Dict[Tuple[int, int], List[int]],
        easy_ratio: float,
        medium_ratio: float,
        hard_ratio: float,
    ) -> None:
        self.num_entities = num_entities
        self.train_pairs = train_pairs
        self.easy_ratio = easy_ratio
        self.medium_ratio = medium_ratio
        self.hard_ratio = hard_ratio
        self.hard_pool: Dict[Tuple[int, int], List[int]] = {}

    def update_hard_pool(self, pool: Dict[Tuple[int, int], List[int]]) -> None:
        self.hard_pool = pool

    def _sample_easy(self, h: int, r: int, num: int) -> List[int]:
        negatives = []
        while len(negatives) < num:
            cand = random.randint(0, self.num_entities - 1)
            if cand not in self.train_pairs.get((h, r), []):
                negatives.append(cand)
        return negatives

    def _sample_medium(self, h: int, r: int, num: int) -> List[int]:
        pool = self.train_pairs.get((h, r), [])
        negatives = []
        while len(negatives) < num:
            cand = random.randint(0, self.num_entities - 1)
            if cand not in pool:
                negatives.append(cand)
        return negatives

    def _sample_hard(self, h: int, r: int, num: int) -> List[int]:
        pool = self.hard_pool.get((h, r), [])
        negatives = []
        if pool:
            for _ in range(num):
                negatives.append(random.choice(pool))
            return negatives
        return self._sample_easy(h, r, num)

    def sample(self, triples: List[TripleIds], num_negatives: int) -> torch.Tensor:
        negatives = []
        for h, r, _ in triples:
            num_easy = int(num_negatives * self.easy_ratio)
            num_medium = int(num_negatives * self.medium_ratio)
            num_hard = num_negatives - num_easy - num_medium

            sampled = []
            sampled += self._sample_easy(h, r, num_easy)
            sampled += self._sample_medium(h, r, num_medium)
            sampled += self._sample_hard(h, r, num_hard)

            negatives.append(sampled)
        return torch.tensor(negatives, dtype=torch.long)
