import base64
import json


class Tokenizer:
    """Tokenizer base class for tokenizing text data."""

    def __init__(self):
        # utf-8 encoding for all 256 bytes
        self.vocab = {id: bytes([id]) for id in range(256)}

    def encode(self, text: str) -> list[int]:
        """Encode the text into a list of ids."""
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        """Decode the ids into text."""
        raise NotImplementedError

    def train(
        self,
        text: str,
        vocab_size: int,
        verbose: bool = False,
    ) -> None:
        """Train the tokenizer on the given text."""
        raise NotImplementedError

    def save(self, vocab_file: str, merges_file: str) -> None:
        """Save the tokenizer to disk."""
        # byte string cannot be serialized to json
        vocab = {id: base64.b64encode(value).decode("utf-8") for id, value in self.vocab.items()}

        # tuple cannot be serialized to json
        merges = {str(k): v for k, v in self.merges.items()}

        with open(vocab_file, "w") as f:
            json.dump(vocab, f, indent=4)
        with open(merges_file, "w") as f:
            json.dump(merges, f, indent=4)

    def load(self, vocab_file: str, merges_file: str) -> None:
        with open(vocab_file, "r") as f:
            self.vocab = json.load(f)
        # decode the base64 encoded bytes
        self.vocab = {int(id): base64.b64decode(value) for id, value in self.vocab.items()}

        with open(merges_file, "r") as f:
            self.merges = json.load(f)
        # convert the key back to tuple
        self.merges = {eval(k): v for k, v in self.merges.items()}
