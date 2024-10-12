import base64
import json

from core.tokenizer.base import Tokenizer
from core.utils import get_stats, merge


class BasicTokenizer(Tokenizer):
    """
    Basic tokenizer to encode and decode text using byte pair encoding.
    """

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        while len(ids) > 1:
            stats = get_stats(ids)
            # the pair with the smallest index is selected
            # the order is part of the tokenization algo to ensure deterministic output
            # otherwise may have issue if dealing with "xyz" where "xy" and "yz" are both different tokens
            # "xyz" is not a token
            pair = min(stats, key=lambda x: self.merges.get(x, float("inf")))
            if pair not in self.merges:
                break
            id = self.merges[pair]
            ids = merge(ids, pair, id)
        return ids

    def decode(self, ids: list[int]) -> str:
        return b"".join([self.vocab[id] for id in ids]).decode("utf-8")

    def train(
        self,
        text: str,
        vocab_size: int,
        vocab_file: str = "vocab.json",
        merges_file: str = "merges.json",
        verbose: bool = False,
    ) -> None:
        ids = list(text.encode("utf-8"))
        self.merges = {}
        while len(self.vocab) < vocab_size:
            stats = get_stats(ids)
            pair = max(stats, key=stats.get)
            id = len(self.vocab)
            ids = merge(ids, pair, id)
            self.merges[pair] = id
            self.vocab[id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose:
                print(f"merge {pair} -> {id}, occurences: {stats[pair]}")
