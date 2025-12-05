# Sentence Transformers for ExecuTorch

This example demonstrates how to export and run sentence-transformer models with ExecuTorch. Sentence transformers generate semantically meaningful embeddings for text, useful for semantic search, clustering, duplicate detection, and similarity tasks.

## Overview

This implementation provides a **generic framework** that works with any encoder-based sentence-transformer model from HuggingFace. The exported models can run efficiently on CPU with XNNPack optimization.

### Key Features

- ✅ **Generic Implementation**: Works with any sentence-transformer model
- ✅ **Verified Quality**: Perfect embedding match (cosine similarity = 1.0)
- ✅ **Optimized Backends**: XNNPack (optimized CPU) and generic CPU
- ✅ **Production Ready**: Tested export, benchmarking, and validation tools

### Tested Models

The following models have been validated to export and run correctly:

| Model                 | Architecture       | Embedding Dim | Parameters | Use Case          |
| --------------------- | ------------------ | ------------- | ---------- | ----------------- |
| **all-MiniLM-L6-v2**  | MiniLM (6 layers)  | 384           | 22.7M      | Lightweight, fast |
| **all-MiniLM-L12-v2** | MiniLM (12 layers) | 384           | 33.4M      | Better quality    |
| **all-mpnet-base-v2** | MPNet              | 768           | 109M       | Highest quality   |

## Quick Start

### Prerequisites

**Option 1: Quick Install (Recommended)**

```bash
# Install all dependencies with one command
cd examples/models/sentence_transformer
./install_requirements.sh
```

**Option 2: Manual Install**

```bash
# Install ExecuTorch (from repo root)
./install_requirements.sh

# Install sentence transformer dependencies
pip install transformers tokenizers scikit-learn numpy
```

### 1. Export a Model

```bash
cd examples/models/sentence_transformer

# Export all-MiniLM-L6-v2 with XNNPack backend (recommended)
python examples/models/sentence_transformer/export_sentence_transformer.py \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --backend xnnpack \
    --output-dir ./xnnpack_model

# Export all-MiniLM-L12-v2 (larger, better quality)
python examples/models/sentence_transformer/export_sentence_transformer.py \
    --model sentence-transformers/all-MiniLM-L12-v2 \
    --backend xnnpack \
    --output-dir ./xnnpack_model

# Export all-mpnet-base-v2 (highest quality, 768-dim)
python examples/models/sentence_transformer/export_sentence_transformer.py \
    --model sentence-transformers/all-mpnet-base-v2 \
    --backend xnnpack \
    --output-dir ./xnnpack_model
```

**Output:** `model.pte` file in the output directory

### 2. Validate the Export

```bash
# Compare exported model with original transformers model
python examples/models/sentence_transformer/compare_embeddings.py \
    --model-path ./xnnpack_model/model.pte \
    --model-name sentence-transformers/all-MiniLM-L6-v2 \
    --sentences "This is a test sentence."
```

**Expected Results:**

```
✅ Cosine Similarity:  1.000000  (perfect match!)
✅ L2 Distance:        0.000005  (essentially zero)
✅ Mean Abs Error:     0.000000  (perfect!)

Verdict: EXCELLENT - ExecuTorch model matches original model perfectly!
```

### 3. Benchmark Performance

```bash
# Compare XNNPack vs CPU backend performance
python examples/models/sentence_transformer/benchmark_backends.py --iterations 100
```

## Running with C++ (executor_runner)

After exporting your model, you can run it using the generic `executor_runner` tool without Python dependencies. This is useful for production deployment and validating the C++ runtime.

### Build executor_runner

```bash
make executor-runner-xnnpack
```

### Run Your Model

**Step 1: Create input binary files**

The `executor_runner` requires binary input files (not text strings). Use the provided script to tokenize text and generate these files:

```bash
# Default text: "This is an example sentence."
python examples/models/sentence_transformer/create_input_bins.py

# Custom text with output directory
python examples/models/sentence_transformer/create_input_bins.py \
    --text "Machine learning is great" \
    --output-dir ./my_inputs
```

This creates:

- `input_ids.bin` - Tokenized input IDs
- `attention_mask.bin` - Attention mask

**Step 2: Run executor_runner**

```bash
# Run with the generated inputs
./cmake-out/executor_runner \
    --model_path=./xnnpack_model/model.pte \
    --inputs=input_ids.bin,attention_mask.bin

# Optionally save the output
./cmake-out/executor_runner \
    --model_path=./xnnpack_model/model.pte \
    --inputs=input_ids.bin,attention_mask.bin \
    --output_file=./output
```

**Alternative: Quick test with default inputs**

If you don't provide `--inputs`, the runner fills tensors with ones (not meaningful embeddings):

```bash
./cmake-out/executor_runner --model_path=./xnnpack_model/model.pte
```

**Note:** For production use with text input, you'll need to:

1. Tokenize text to `input_ids` and `attention_mask` (using a C++ tokenizer or the Python script above)
2. Pass tensors to the model
3. Extract the embedding output

See the [ExecuTorch Runtime documentation](https://pytorch.org/executorch/main/runtime-overview.html) for building custom C++ applications.

## Python Usage

### Basic Example

```python
import torch
from transformers import AutoTokenizer
from executorch.extension.pybindings.portable_lib import _load_for_executorch

# Load the exported model
model = _load_for_executorch("my_model/model.pte")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Encode text
text = "This is a sample sentence to encode."
encoded = tokenizer(
    text,
    padding="max_length",
    max_length=128,
    truncation=True,
    return_tensors="pt"
)

# Run inference
outputs = model.forward((encoded["input_ids"], encoded["attention_mask"]))
embedding = outputs[0]  # Shape: [1, 384]

print(f"Embedding: {embedding.shape}")  # torch.Size([1, 384])
```

### Semantic Search Example

```python
def get_embedding(text, model, tokenizer, max_length=128):
    """Get sentence embedding."""
    encoded = tokenizer(
        text,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt"
    )
    outputs = model.forward((encoded["input_ids"], encoded["attention_mask"]))
    return outputs[0].numpy()

# Encode corpus
corpus = [
    "A man is eating food.",
    "A man is eating a piece of bread.",
    "The girl is carrying a baby.",
    "A man is riding a horse.",
]

corpus_embeddings = [get_embedding(text, model, tokenizer) for text in corpus]

# Encode query
query = "A person is eating."
query_embedding = get_embedding(query, model, tokenizer)

# Compute cosine similarities
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity([query_embedding], corpus_embeddings)[0]

# Get top matches
top_k = 2
top_indices = similarities.argsort()[-top_k:][::-1]

print("Query:", query)
for idx in top_indices:
    print(f"  {corpus[idx]} (similarity: {similarities[idx]:.4f})")
```

## Model Architecture

The sentence transformer model consists of:

1. **Tokenizer**: Converts text to token IDs (using HuggingFace tokenizers)
2. **Transformer Encoder**: Processes tokens through multiple layers
3. **Mean Pooling**: Aggregates token embeddings into sentence embedding
4. **Output**: Dense vector representation

```
Input Text
    ↓
Tokenization → [input_ids, attention_mask]
    ↓
Transformer Encoder → [batch, seq_len, hidden_size]
    ↓
Mean Pooling (attention-mask weighted)
    ↓
Sentence Embedding → [batch, embedding_dim]
```

## Available Tools

### export_sentence_transformer.py

Export sentence-transformer models to ExecuTorch format.

**Options:**

- `--model`: HuggingFace model name (default: all-MiniLM-L6-v2)
- `--backend`: Backend to use (`xnnpack` or `cpu`)
- `--output-dir`: Output directory for exported model
- `--max-seq-length`: Maximum sequence length (default: 128)

**Examples:**

```bash
# Export with default settings
python export_sentence_transformer.py

# Export with custom model and settings
python export_sentence_transformer.py \
    --model sentence-transformers/all-mpnet-base-v2 \
    --backend xnnpack \
    --output-dir ./mpnet_export \
    --max-seq-length 256
```

### compare_embeddings.py

Validate exported model by comparing embeddings with the original model.

**Options:**

- `--model-path`: Path to exported .pte file (required)
- `--model-name`: HuggingFace model name
- `--sentences`: Test sentences to compare
- `--max-length`: Maximum sequence length

**Examples:**

```bash
# Compare single sentence
python compare_embeddings.py \
    --model-path exported_model/model.pte \
    --model-name sentence-transformers/all-MiniLM-L6-v2

# Compare multiple sentences
python compare_embeddings.py \
    --model-path exported_model/model.pte \
    --model-name sentence-transformers/all-MiniLM-L6-v2 \
    --sentences "Hello world" "Machine learning" "Semantic search"
```

### benchmark_backends.py

Compare performance between XNNPack and CPU backends.

**Options:**

- `--max-seq-length`: Maximum sequence length (default: 128)
- `--warmup-iterations`: Warmup iterations (default: 10)
- `--iterations`: Benchmark iterations (default: 100)
- `--output-dir`: Output directory (default: ./benchmark_results)
- `--skip-export`: Skip model export (use existing models)

**Examples:**

```bash
# Run full benchmark
python benchmark_backends.py

# Run with more iterations for stable results
python benchmark_backends.py --iterations 1000

# Use existing exported models
python benchmark_backends.py --skip-export
```

## Performance

### Backend Comparison

Based on benchmarking with all-MiniLM-L6-v2 on **Apple M3 Pro**:

| Backend     | Average Latency | Throughput       | Speedup         |
| ----------- | --------------- | ---------------- | --------------- |
| **XNNPack** | **~15.9 ms**    | **~63 sent/sec** | **~76x**        |
| **CPU**     | ~1,212 ms       | ~0.83 sent/sec   | 1.0x (baseline) |

_**Note:** XNNPack provides a massive **76x speedup** over baseline CPU! Run `benchmark_backends.py` on your hardware for platform-specific numbers._

## Validation Results

All tested models produce **identical embeddings** to the original transformers models:

| Model             | Cosine Similarity | L2 Distance | Status     |
| ----------------- | ----------------- | ----------- | ---------- |
| all-MiniLM-L6-v2  | 1.000000          | 0.000005    | ✅ Perfect |
| all-MiniLM-L12-v2 | 1.000000          | 0.000006    | ✅ Perfect |
| all-mpnet-base-v2 | 1.000000          | 0.000004    | ✅ Perfect |

## License

This example is licensed under the BSD-style license. See the LICENSE file in the root directory of this source tree.

The sentence-transformers models are licensed under Apache 2.0.
