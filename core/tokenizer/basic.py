import base64
import json

from core.utils import get_stats, merge


class BasicTokenizer:
    """
    Basic tokenizer to encode and decode text using byte pair encoding.
    """

    def __init__(self, vocab_file: str, merges_file: str):
        with open(vocab_file, "r") as f:
            self.vocab = json.load(f)
        # decode the base64 encoded bytes
        self.vocab = {int(id): base64.b64decode(value) for id, value in self.vocab.items()}

        with open(merges_file, "r") as f:
            self.merges = json.load(f)
        # convert the key back to tuple
        self.merges = {eval(k): v for k, v in self.merges.items()}

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
        vocab = {id: bytes([id]) for id in range(256)}
        merges = {}
        while len(vocab) < vocab_size:
            stats = get_stats(ids)
            pair = max(stats, key=stats.get)
            id = len(vocab)
            ids = merge(ids, pair, id)
            merges[pair] = id
            vocab[id] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(f"merge {pair} -> {id}, occurers: {stats[pair]}")

        # byte string cannot be serialized to json
        for id, value in vocab.items():
            vocab[id] = base64.b64encode(value).decode("utf-8")

        # tuple cannot be serialized to json
        merges = {str(k): v for k, v in merges.items()}

        with open(vocab_file, "w") as f:
            json.dump(vocab, f, indent=4)
        with open(merges_file, "w") as f:
            json.dump(merges, f, indent=4)
        return cls(vocab_file, merges_file)
