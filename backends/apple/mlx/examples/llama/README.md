# Llama MLX Example

This example demonstrates how to export and run Llama models using the MLX delegate for Apple Silicon.

## Features

- **Export**: Convert HuggingFace Llama models to ExecutorCh format with MLX delegate
- **Quantization**: Optional INT4/INT8 weight quantization via TorchAO
- **KV Cache**: Efficient KV cache implementation for autoregressive generation
- **Custom Ops**: Uses `mlx::apply_rope` for optimal MLX execution
- **Pybindings**: Run inference using ExecutorCh Python bindings

## Requirements

```bash
pip install transformers torchao
```

## Usage

### Export a Model

```bash
# Export Llama 3.2 1B (unquantized)
python -m executorch.backends.apple.mlx.examples.llama.export_llama \
    --model-id "unsloth/Llama-3.2-1B-Instruct" \
    --output llama_1b.pte

# Export with INT4 quantization (smaller model size)
python -m executorch.backends.apple.mlx.examples.llama.export_llama \
    --model-id "unsloth/Llama-3.2-1B-Instruct" \
    --output llama_1b_int4.pte \
    --quantize int4

# Export larger models
python -m executorch.backends.apple.mlx.examples.llama.export_llama \
    --model-id "meta-llama/Llama-3.2-3B-Instruct" \
    --output llama_3b_int4.pte \
    --quantize int4
```

### Run Inference

```bash
# Basic generation
python -m executorch.backends.apple.mlx.examples.llama.run_llama \
    --model llama_1b.pte \
    --prompt "What is the capital of France?"

# With chat template (for instruct models)
python -m executorch.backends.apple.mlx.examples.llama.run_llama \
    --model llama_1b.pte \
    --prompt "Explain quantum computing in simple terms" \
    --use-chat-template \
    --max-new-tokens 256

# Greedy decoding (temperature=0)
python -m executorch.backends.apple.mlx.examples.llama.run_llama \
    --model llama_1b.pte \
    --prompt "1 + 1 = " \
    --temperature 0
```

## Options

### Export Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model-id` | `unsloth/Llama-3.2-1B-Instruct` | HuggingFace model ID |
| `--output` | `llama_mlx.pte` | Output .pte file path |
| `--quantize` | `None` | Quantization: `int4`, `int8`, or none |
| `--max-seq-len` | `4096` | Maximum sequence length for KV cache |

### Inference Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | (required) | Path to .pte file |
| `--tokenizer` | (auto) | Path to tokenizer (defaults to model_path_tokenizer) |
| `--prompt` | `Hello, how are you?` | Input prompt |
| `--max-new-tokens` | `128` | Maximum tokens to generate |
| `--temperature` | `0.7` | Sampling temperature (0 for greedy) |
| `--top-p` | `0.9` | Top-p sampling threshold |
| `--use-chat-template` | `False` | Apply chat template |
| `--no-stream` | `False` | Don't stream output |

## Architecture

The example uses a custom model wrapper (`LlamaWithFunctionalKV`) that:

1. **Replaces RMSNorm** with `torch.nn.functional.rms_norm` - which maps to MLX's efficient RMSNorm implementation via the aten.rms_norm handler

2. **Replaces Attention** with `KVCacheAttention` which:
   - Uses `torch.ops.mlx.apply_rope` for rotary position embeddings
   - Implements functional KV cache updates (compatible with `torch.export`)
   - Supports Grouped Query Attention (GQA)

3. **Pattern Matching** during export:
   - `scaled_dot_product_attention` → MLX's fused SDPA kernel
   - `slice + copy + slice_scatter` → MLX's in-place slice update
   - `dequantize_affine + linear` → MLX's quantized matmul

## Supported Models

- Llama 3.2 (1B, 3B)
- Llama 3.1 (8B - requires sufficient memory)
- Other Llama-architecture models (Mistral, etc.)

## Performance Notes

- **Prefill**: Processes the entire prompt in parallel
- **Decode**: Generates one token at a time with KV cache
- **Quantization**: INT4 reduces model size ~4x with minimal quality loss
- **Memory**: KV cache is pre-allocated based on `max-seq-len`

## Troubleshooting

### Out of Memory

Reduce `max-seq-len` or use quantization:
```bash
python -m executorch.backends.apple.mlx.examples.llama.export_llama \
    --max-seq-len 1024 \
    --quantize int4
```

### Slow Generation

Ensure you're using a Mac with Apple Silicon (M1/M2/M3/M4).

### Model Not Found

Install transformers with `pip install transformers` and ensure you have network access to download the model.
