import fire

from core.tokenizer.basic import BasicTokenizer


def main(vocab_file: str, merges_file: str):
    tokenizer = BasicTokenizer(vocab_file, merges_file)
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
