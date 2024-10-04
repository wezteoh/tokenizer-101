import fire
from core.tokenizer.basic import BasicTokenizer


def main(vocab_file: str, merges_file: str, vocab_size: int = 1000):
    with open(vocab_file, "r") as f:
        vocab = json.load(f)
    with open(merges_file, "r") as f:
        merges = json.load(f)

    tokenizer = BasicTokenizer(vocab, merges)
    while True:
        try:
            text = input("Enter text to tokenize: ")
            ids = tokenizer.encode(text)
            print(f"Encoded: {ids}")
            decoded = tokenizer.decode(ids)
            print(f"Decoded: {decoded}")
        except KeyboardInterrupt:
            break
