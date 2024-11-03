import sentencepiece as spm

from core.tokenizer.base import Tokenizer


class SentencePieceGemmaTokenizer(Tokenizer):
    def __init__(self):
        self.sp = spm.SentencePieceProcessor(model_file=".temp/gemma_tokenizer.model")

    def encode(self, text: str) -> list:
        return self.sp.encode_as_ids(text)

    def decode(self, tokens: list) -> str:
        return self.sp.decode_ids(tokens)

    def save(self, vocab_file: str, merges_file: str) -> None:
        raise NotImplementedError("Saving is not supported for SentencePieceLlama2")

    def load(self, vocab_file: str, merges_file: str) -> None:
        raise NotImplementedError("Loading is not supported for SentencePieceLlama2")

    def train(
        self,
        text: str,
        vocab_size: int,
        verbose: bool = False,
    ) -> None:
        raise NotImplementedError("Training is not supported for SentencePieceLlama2")
