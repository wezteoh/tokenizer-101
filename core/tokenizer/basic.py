import json
from core.utils import get_stats, merge


class BasicTokenizer:
    """
    Basic tokenizer to encode and decode text using byte pair encoding.
    """

    def __init__(self, vocab_file: str, merges_file: str):
        with open(vocab_file, "r") as f:
            self.vocab = json.load(f)
        with open(merges_file, "r") as f:
            self.merges = json.load(f)

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        while len(ids) > 1:
            stats = get_stats(ids)
            # the pair with the smallest index is selected
            # the order is part of the tokeinization algo to ensure deterministic output
            # otherwise may have issue if dealing with "xyz" where "xy" and "yz" are both different tokens
            # "xyz" is not a token
            pair = min(stats, key=lambda x: self.merges.get(x, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def decode(self, ids: list[int]) -> str:
        return b"".join([self.vocab[id] for id in ids]).decode("utf-8")

    @classmethod
    def train(
        cls,
        text: str,
        vocab_size: int,
        vocab_file: str = "vocab.json",
        merges_file: str = "merges.json",
        verbose: bool = False,
    ) -> None:
        ids = list(text.encode("utf-8"))
        # create initial utf8 vocab
        vocab = {idx: bytes([idx]) for idx in range(256)}
        merges = {}
        while len(vocab) < vocab_size:
            stats = get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = len(vocab)
            ids = merge(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(f"merge {pair} -> {idx}, occurers: {stats[pair]}")

        with open(vocab_file, "w") as f:
            json.dump(vocab, f)
        with open(merges_file, "w") as f:
            json.dump(merges, f)
