from typing import Literal, Union

import regex as re

from core.tokenizer.base import Tokenizer
from core.utils import get_stats, merge

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RegexTokenizer(Tokenizer):
    def __init__(self, pattern: str = None):
        super().__init__()
        if pattern is None:
            pattern = GPT4_SPLIT_PATTERN
        self.pattern = re.compile(GPT4_SPLIT_PATTERN)
        self.special_tokens = {}
        self.inversed_special_tokens = {}

    def register_special_tokens(self, special_tokens):
        # special_tokens is a dictionary of str -> int
        # example: {"<|endoftext|>": 100257}
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def _encode_chunk(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        while len(ids) > 1:
            stats = get_stats(ids)
            pair = min(stats, key=lambda x: self.merges.get(x, float("inf")))
            if pair not in self.merges:
                break
            id = self.merges[pair]
            ids = merge(ids, pair, id)
        return ids

    def _encode_ordinary(self, text: str) -> list[int]:
        split_texts = re.findall(self.pattern, text)
        ids = []
        for text in split_texts:
            ids.extend(self._encode_chunk(text))
        return ids

    def encode(self, text: str, allowed_special: Union[set, Literal["all"]] = {}) -> list[int]:
        """
        Encode the text into a list of ids.
        if allowed_special is not empty, when we encounter corresponding special tokens,
        need to encode them as well based of self.special_tokens
        """

        if not allowed_special:
            return self._encode_ordinary(text)

        if allowed_special == "all":
            special = self.special_tokens
        else:
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}

        pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        chunks = re.split(pattern, text)
        ids = []
        for chunk in chunks:
            if chunk in special:
                ids.extend(special[chunk])
            else:
                ids.extend(self._encode_ordinary(chunk))
        return ids

    def decode(self, ids: list[int]) -> str:
        return b"".join([self.vocab[id] for id in ids]).decode("utf-8")

    def train(
        self,
        text: str,
        vocab_size: int,
        verbose: bool = False,
    ) -> None:

        self.merges = {}
        text_segments = re.findall(self.pattern, text)
        id_segments = [list(t.encode("utf-8")) for t in text_segments]

        while len(self.vocab) < vocab_size:
            stats = {}
            for ids in id_segments:
                stats = get_stats(ids, stats)
            pair = max(stats, key=stats.get)
            id = len(self.vocab)
            self.merges[pair] = id
            # merge the ids in the existing segments and continue
            id_segments = [merge(ids, pair, id) for ids in id_segments]
            self.vocab[id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose:
                print(f"merge {pair} -> {id}, occurencess: {stats[pair]}")
