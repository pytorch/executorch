# Tokenizers

C++ tokenizer implementations with Python bindings. Located in `extension/llm/tokenizers/`.

## Installation
```bash
pip install -e ./extension/llm/tokenizers/
```

## Python API

```python
from pytorch_tokenizers import get_tokenizer

# Auto-detect tokenizer type from file
tokenizer = get_tokenizer("path/to/tokenizer.model")  # or .json

# Encode/decode
tokens = tokenizer.encode("Hello world")
text = tokenizer.decode(tokens)
```

## Available Tokenizers

| Class | Format | Use Case |
|-------|--------|----------|
| `HuggingFaceTokenizer` | `.json` | HuggingFace models |
| `TiktokenTokenizer` | `.model` | OpenAI/Llama 3 |
| `Llama2cTokenizer` | `.model` | Llama 2, SentencePiece |
| `CppSPTokenizer` | `.model` | SentencePiece (C++) |

## Direct Usage

```python
from pytorch_tokenizers import HuggingFaceTokenizer, TiktokenTokenizer, Llama2cTokenizer

# HuggingFace (tokenizer.json)
tokenizer = HuggingFaceTokenizer("tokenizer.json", "tokenizer_config.json")

# Tiktoken (Llama 3, etc.)
tokenizer = TiktokenTokenizer(model_path="tokenizer.model")

# Llama2c/SentencePiece
tokenizer = Llama2cTokenizer(model_path="tokenizer.model")
```

## C++ Tokenizers

For C++ runners, include headers from `extension/llm/tokenizers/include/`:
- `hf_tokenizer.h` - HuggingFace
- `tiktoken.h` - Tiktoken
- `sentencepiece.h` - SentencePiece
- `llama2c_tokenizer.h` - Llama2c
- `tekken.h` - Mistral Tekken v7
