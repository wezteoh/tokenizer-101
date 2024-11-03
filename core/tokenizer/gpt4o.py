import tiktoken

from core.tokenizer.regex import RegexTokenizer
from core.utils import get_stats, merge

GPT4O_SPLIT_PATTERN = "|".join(
    [
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""\p{N}{1,3}""",
        r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
        r"""\s*[\r\n]+""",
        r"""\s+(?!\S)""",
        r"""\s+""",
    ]
)
GPT4O_SPECIAL_TOKENS = {"<|endoftext|>": 199999, "<|endofprompt|>": 200018}


def bpe(token: str, mergeable_ranks: dict, max_rank: int) -> list:
    parts = [bytes([x]) for x in token]
    while True:
        min_rank = None
        for idx, (part1, part2) in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(part1 + part2)
            if rank is not None and (min_rank is None or rank < min_rank):
                min_rank = rank
                min_idx = idx
        if min_rank is None or min_rank >= max_rank:
            break
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2 :]
    return parts


def recover_merges(mergeable_ranks: dict) -> dict:
    merges = {}
    for token, rank in mergeable_ranks.items():
        if len(token) == 1:
            continue
        pair = bpe(token, mergeable_ranks, max_rank=rank)
        assert len(pair) == 2
        idx0 = mergeable_ranks[pair[0]]
        idx1 = mergeable_ranks[pair[1]]
        merges[(idx0, idx1)] = rank

    return merges


class GPT4OTokenizer(RegexTokenizer):
    def __init__(self):
        super().__init__(GPT4O_SPLIT_PATTERN)
        enc = tiktoken.get_encoding("o200k_base")
        self.merges = recover_merges(enc._mergeable_ranks)

        # for historical reason, the mapping of raw byte idxs to token idxs are not in order
        # so we recover:
        # 1. mapping of byte idxs to tiktoken token idxs for encoding
        # 2. mapping of token idxs to raw bytes in vocab for decoding
        self.byteidx2tokenidx = {x: enc._mergeable_ranks[bytes([x])] for x in range(256)}
        self.vocab = {v: k for k, v in enc._mergeable_ranks.items() if v < 256}
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
        self.register_special_tokens(GPT4O_SPECIAL_TOKENS)

    def _encode_chunk(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        ids = [self.byteidx2tokenidx[x] for x in ids]
        while len(ids) > 1:
            stats = get_stats(ids)
            pair = min(stats, key=lambda x: self.merges.get(x, float("inf")))
            if pair not in self.merges:
                break
            id = self.merges[pair]
            ids = merge(ids, pair, id)
        return ids

    def register_special_tokens(self, special_tokens):
        return super().register_special_tokens(special_tokens)

    def train(
        self,
        text: str,
        vocab_size: int,
        verbose: bool = False,
    ) -> None:
        raise NotImplementedError("Training is not supported for GPT4OTokenizer")

    def save(self, vocab_file: str, merges_file: str) -> None:
        return NotImplementedError("Saving is not supported for GPT4OTokenizer")

    def load(self, vocab_file: str, merges_file: str) -> None:
        raise NotImplementedError("Loading is not supported for GPT4OTokenizer")
