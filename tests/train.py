import fire

from core.tokenizer import BasicTokenizer, RegexTokenizer

tokenizer_mapping = {
    "basic": BasicTokenizer,
    "regex": RegexTokenizer,
}


def main(
    train_file: str, tokenizer_type: str, vocab_file: str, merges_file: str, vocab_size: int = 512
):
    with open(train_file, "r") as f:
        train_text = f.read()
    tokenizer = tokenizer_mapping[tokenizer_type]()
    tokenizer.train(
        text=train_text,
        vocab_size=vocab_size,
        verbose=True,
    )
    tokenizer.save(vocab_file, merges_file)


if __name__ == "__main__":
    fire.Fire(main)
