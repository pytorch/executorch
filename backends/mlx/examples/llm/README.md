# Llama MLX Example

This example demonstrates how to export and run Llama models using the MLX delegate for Apple Silicon.

## Features

- **Export**: Convert HuggingFace Llama models to ExecuTorch format with MLX delegate
- **Quantization**: Optional INT4/INT8 weight quantization via TorchAO
- **KV Cache**: Efficient KV cache implementation for autoregressive generation
- **Custom Ops**: Uses `mlx::rope` for optimal RoPE execution on MLX
- **Pybindings**: Run inference using ExecuTorch Python bindings

## Requirements

```bash
pip install transformers torchao
```

For the `export_llm_hf` path (optimum-executorch pipeline):

```bash
pip install transformers torch optimum-executorch
```

## Export Scripts

There are two export scripts:

| Script | Description |
|--------|-------------|
| `export_llama` | Custom model wrapper with functional KV cache and `mlx::rope` |
| `export_llm_hf` | Uses optimum-executorch's `CausalLMExportableModule` pipeline |

### `export_llama` (custom wrapper)

```bash
# Export Llama 3.2 1B (bf16, no quantization)
python -m executorch.backends.mlx.examples.llm.export_llama \
    --model-id "unsloth/Llama-3.2-1B-Instruct" \
    --output llama_1b.pte

# Export with INT4 quantized linear layers
python -m executorch.backends.mlx.examples.llm.export_llama \
    --model-id "unsloth/Llama-3.2-1B-Instruct" \
    --output llama_1b_int4.pte \
    --quantize-linear int4

# Export with both linear and embedding quantization
python -m executorch.backends.mlx.examples.llm.export_llama \
    --model-id "unsloth/Llama-3.2-1B-Instruct" \
    --output llama_1b_int4.pte \
    --quantize-linear int4 \
    --quantize-embeddings int4
```

#### Export Options (`export_llama`)

| Option | Default | Description |
|--------|---------|-------------|
| `--model-id` | `unsloth/Llama-3.2-1B-Instruct` | HuggingFace model ID |
| `--output` | *(required)* | Output .pte file path |
| `--max-seq-len` | `1024` | Maximum sequence length for KV cache |
| `--dtype` | `bf16` | Model dtype (`fp32`, `fp16`, `bf16`) |
| `--quantize-linear` | None | Quantization for linear layers (`int4`, `int8`) |
| `--quantize-embeddings` | None | Quantization for embedding layers (`int4`, `int8`) |
| `--no-tie-word-embeddings` | `False` | Disable re-tying lm_head to embedding after quantization |

```bash
# Baseline export using optimum-executorch
python -m executorch.backends.mlx.examples.llm.export_llm_hf \
    --model-id "unsloth/Llama-3.2-1B-Instruct" \
    --output llama_hf.pte

# With custom MLX components
python -m executorch.backends.mlx.examples.llm.export_llm_hf \
    --model-id "unsloth/Llama-3.2-1B-Instruct" \
    --output llama_hf_mlx.pte \
    --use-custom-sdpa \
    --use-custom-kv-cache
```

#### Export Options (`export_llm_hf`)

| Option | Default | Description |
|--------|---------|-------------|
| `--model-id` | `unsloth/Llama-3.2-1B-Instruct` | HuggingFace model ID |
| `--output` | *(required)* | Output .pte file path |
| `--max-seq-len` | `1024` | Maximum sequence length for KV cache |
| `--dtype` | `bf16` | Model dtype (`fp32`, `fp16`, `bf16`) |
| `--quantize-linear` | None | Quantization for linear layers (`int4`, `int8`) |
| `--quantize-embeddings` | None | Quantization for embedding layers (`int4`, `int8`) |
| `--no-tie-word-embeddings` | `False` | Disable re-tying lm_head to embedding after quantization |
| `--use-custom-sdpa` | `False` | Use MLX custom SDPA (`mlx::custom_sdpa`) |
| `--use-custom-kv-cache` | `False` | Use MLX custom KV cache (`mlx::kv_cache_update`) |

## Run Scripts

There are two corresponding run scripts:

| Script | For models exported with | Tokenizer source |
|--------|--------------------------|------------------|
| `run_llama` | `export_llama` | Loaded from HuggingFace by model ID |
| `run_llm_hf` | `export_llm_hf` | Loaded from HuggingFace by model ID |

### `run_llama`

```bash
python -m executorch.backends.mlx.examples.llm.run_llama \
    --pte llama_1b.pte \
    --model-id unsloth/Llama-3.2-1B-Instruct \
    --prompt "What is the capital of France?"
```

#### Inference Options (`run_llama`)

| Option | Default | Description |
|--------|---------|-------------|
| `--pte` | `/tmp/llama_test.pte` | Path to .pte file |
| `--model-id` | `unsloth/Llama-3.2-1B-Instruct` | HuggingFace model ID (for tokenizer) |
| `--prompt` | `The quick brown fox` | Input prompt |
| `--max-new-tokens` | `50` | Maximum tokens to generate |

### `run_llm_hf`

```bash
python -m executorch.backends.mlx.examples.llm.run_llm_hf \
    --pte llama_hf.pte \
    --model-id unsloth/Llama-3.2-1B-Instruct \
    --prompt "Explain quantum computing in simple terms"
```

#### Inference Options (`run_llm_hf`)

| Option | Default | Description |
|--------|---------|-------------|
| `--pte` | `llama_hf.pte` | Path to .pte file |
| `--model-id` | `unsloth/Llama-3.2-1B-Instruct` | HuggingFace model ID (for tokenizer) |
| `--prompt` | `The quick brown fox` | Input prompt |
| `--max-new-tokens` | `50` | Maximum tokens to generate |

## Architecture

### `export_llama` model wrapper

The `export_llama` script uses a custom model wrapper (`LlamaWithFunctionalKV`) that:

1. **Replaces RMSNorm** with `torch.nn.functional.rms_norm` — which maps to MLX's efficient `fast::rms_norm` implementation via the `aten.rms_norm` handler.

2. **Replaces Attention** with `KVCacheAttention` which:
   - Uses `torch.ops.mlx.rope` for rotary position embeddings
   - Implements functional KV cache updates (compatible with `torch.export`)
   - Supports Grouped Query Attention (GQA) via `repeat_interleave`

3. **Pattern Matching** during export:
   - `scaled_dot_product_attention` → MLX's fused SDPA kernel
   - KV cache updates → MLX's index update ops
   - `dequantize_affine + linear` → MLX's quantized matmul

## Supported Models

- Llama 3.2 (1B, 3B)
- Llama 3.1 (8B — requires sufficient memory)
- Other Llama-architecture models (Mistral, etc.)

## Performance Notes

- **Prefill**: Processes the entire prompt in parallel
- **Decode**: Generates one token at a time with KV cache
- **Quantization**: INT4 reduces model size ~4x with minimal quality loss
- **Memory**: KV cache is pre-allocated based on `--max-seq-len`

## Troubleshooting

### Out of Memory

Reduce `--max-seq-len` or use quantization:
```bash
python -m executorch.backends.mlx.examples.llm.export_llama \
    --max-seq-len 512 \
    --quantize-linear int4 \
    --output llama_512.pte
```

### Slow Generation

Ensure you're using a Mac with Apple Silicon (M1/M2/M3/M4).

### Model Not Found

Install transformers with `pip install transformers` and ensure you have network access to download the model.
