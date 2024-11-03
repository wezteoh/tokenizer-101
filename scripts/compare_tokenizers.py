import fire

from core.tokenizer.gpt4o import GPT4OTokenizer
from core.tokenizer.sentencepiece_gemma import SentencePieceGemmaTokenizer

tokenizer_dict = {
    "gpt4o_tokenizer": GPT4OTokenizer,
    "gemma_tokenizer": SentencePieceGemmaTokenizer,
}


def main(input_text_file: str):
    with open(input_text_file, "r") as f:
        text = f.read()

    for tokenizer_name, tokenizer_cls in tokenizer_dict.items():
        tokenizer = tokenizer_cls()
        tokens = tokenizer.encode(text)
        text_to_token_ratio = round(float(len(text)) / len(tokens), 3)
        print(f"Tokenizer: {tokenizer_name}, text_to_token_ratio: {text_to_token_ratio}")


if __name__ == "__main__":
    fire.Fire(main)
