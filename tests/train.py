import fire

from core.tokenizer.basic import BasicTokenizer


def main(train_file: str, vocab_file: str, merges_file: str, vocab_size: int = 1000):
    with open(train_file, "r") as f:
        train_text = f.read()
    tokenizer = BasicTokenizer.train(
        text=train_text,
        vocab_size=vocab_size,
        vocab_file=vocab_file,
        merges_file=merges_file,
        verbose=True,
    )


if __name__ == "__main__":
    fire.Fire(main)
