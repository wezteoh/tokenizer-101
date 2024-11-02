import fire

from core.tokenizer import BasicTokenizer, RegexTokenizer
from core.tokenizer.basic import BasicTokenizer

tokenizer_mapping = {
    "basic": BasicTokenizer,
    "regex": RegexTokenizer,
}


def main(tokenizer_type: str, vocab_file: str, merges_file: str):
    tokenizer = tokenizer_mapping[tokenizer_type]()
    tokenizer.load(vocab_file, merges_file)
    while True:
        try:
            text = input("Enter text to tokenize: ")
            ids = tokenizer.encode(text)
            print(f"Encoded: {ids}")
            decoded = tokenizer.decode(ids)
            print(f"Decoded: {decoded}")
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    fire.Fire(main)
