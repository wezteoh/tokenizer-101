import tiktoken

from core.tokenizer.gpt4o import GPT4OTokenizer


def test_gpt4o_tokenizer():
    enc = tiktoken.get_encoding("o200k_base")
    target_tokens = enc.encode("Hello, world!")

    tokenizer = GPT4OTokenizer()
    tokens = tokenizer.encode("Hello, world!")

    assert tokens == target_tokens
