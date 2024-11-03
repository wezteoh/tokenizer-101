# tokenizer-101
This repo is developed along my deep-dive into the details of tokenization for LLM development, following this amaing tutorial [[Andrej Karpathy's Tutorial](https://www.youtube.com/watch?v=zduSFxRajkE)]. It heavily references this example repo [[MinBPE](https://github.com/karpathy/minbpe)].

## Text-to-token ratio
I ran a comparison of the text-to-token ratios (length of text divided by count of tokens encoded from the text) between the gpt4o tokenizer and gemma tokenizer (the tokenizer used for gemini series). 

In some sense, this ratio is reflecting the efficiency of a tokenization scheme: if one token can capture more text, we are saving more model's capacity from modelling the relationship among units of subtexts captured within the token.

![Tokenization Efficiency Comparison](assets/comparison.png)

## Additional References
1. The gemma tokenizer model is downloaded from the [[Gemma Repo](https://github.com/google/gemma_pytorch/tree/main)].
2. math_text.txt is extracted from the
[[MATH Repo](https://github.com/hendrycks/math)].