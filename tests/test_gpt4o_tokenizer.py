import tiktoken

from core.tokenizer.gpt4o import GPT4OTokenizer

target_text = "Hello, world! 12345678 She's best"


def test_gpt4o_tokenizer():
    enc = tiktoken.get_encoding("o200k_base")
    target_tokens = enc.encode(target_text)

    tokenizer = GPT4OTokenizer()
    tokens = tokenizer.encode(target_text)

    assert tokens == target_tokens
