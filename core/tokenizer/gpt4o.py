import tiktoken

from core.tokenizer.regex import RegexTokenizer


def recover_merges(mergeable_ranks: dict) -> dict:
    pass


class GPT4oTokenizer(RegexTokenizer):
    def __init__(self, pattern: str = None):
        super().__init__(pattern)
        enc = tiktoken.get_encoding("o200k_base")
        self.merges = recover_merges(enc._mergeable_ranks)
