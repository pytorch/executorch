# LLM MLX Example

This example demonstrates how to export and run LLMs using the MLX delegate for Apple Silicon.

## Features

- **Export**: Convert HuggingFace LLMs to ExecuTorch format with MLX delegate
- **Quantization**: Optional INT4/INT8 weight quantization via TorchAO
- **KV Cache**: Efficient KV cache implementation for autoregressive generation
- **Custom Ops**: Uses `mlx::rope` for optimal RoPE execution on MLX
- **Pybindings**: Run inference using ExecuTorch Python bindings

## Requirements

```bash
pip install transformers
```

For the `export_llm_hf` path (optimum-executorch pipeline), install optimum-executorch after installing ExecuTorch:

```bash
pip install optimum-executorch
```

## Scripts Overview

| Script | Description |
|--------|-------------|
| `export_llama` | Custom model wrapper with functional KV cache and `mlx::rope` |
| `run_llama` | Run models exported with `export_llama` |
| `export_llm_hf` | Uses optimum-executorch pipeline, with optional custom MLX SDPA/KV cache |
| `run_llm_hf` | Run models exported with `export_llm_hf` |

---

## `export_llama`

Custom model wrapper (`LlamaWithFunctionalKV`) with functional KV cache and `mlx::rope`.

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

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model-id` | `unsloth/Llama-3.2-1B-Instruct` | HuggingFace model ID |
| `--output` | *(required)* | Output .pte file path |
| `--max-seq-len` | `1024` | Maximum sequence length for KV cache |
| `--dtype` | `bf16` | Model dtype (`fp32`, `fp16`, `bf16`) |
| `--quantize-linear` | None | Quantization for linear layers (`int4`, `int8`) |
| `--quantize-embeddings` | None | Quantization for embedding layers (`int4`, `int8`) |
| `--no-tie-word-embeddings` | `False` | Disable re-tying lm_head to embedding after quantization |

---

## `run_llama`

Run models exported with `export_llama`. Loads the tokenizer from HuggingFace and applies the chat template before inference.

```bash
python -m executorch.backends.mlx.examples.llm.run_llama \
    --pte llama_1b.pte \
    --model-id unsloth/Llama-3.2-1B-Instruct \
    --prompt "What is the capital of France?"
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--pte` | `/tmp/llama_test.pte` | Path to .pte file |
| `--model-id` | `unsloth/Llama-3.2-1B-Instruct` | HuggingFace model ID (for tokenizer) |
| `--prompt` | `The quick brown fox` | Input prompt |
| `--max-new-tokens` | `50` | Maximum tokens to generate |

---

## `export_llm_hf`

Uses optimum-executorch's `CausalLMExportableModule` by default. Optional flags enable custom MLX-optimized components (custom SDPA and/or KV cache).

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

### Options

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

---

## `run_llm_hf`

Run models exported with `export_llm_hf`. Supports both full-prompt prefill (dynamic seq len exports) and token-by-token prefill (fixed seq len exports).

```bash
python -m executorch.backends.mlx.examples.llm.run_llm_hf \
    --pte llama_hf.pte \
    --model-id unsloth/Llama-3.2-1B-Instruct \
    --prompt "Explain quantum computing in simple terms"
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--pte` | `llama_hf.pte` | Path to .pte file |
| `--model-id` | `unsloth/Llama-3.2-1B-Instruct` | HuggingFace model ID (for tokenizer) |
| `--prompt` | `The quick brown fox` | Input prompt |
| `--max-new-tokens` | `50` | Maximum tokens to generate |

---

## Architecture

### `export_llama` model wrapper

The `export_llama` script uses a custom model wrapper (`LlamaWithFunctionalKV`) that:

1. **Replaces RMSNorm** with `torch.nn.RMSNorm` — which emits the `aten.rms_norm` op, mapped to MLX's efficient `fast::rms_norm` implementation.

2. **Replaces Attention** with `KVCacheAttention` which:
   - Uses `torch.ops.mlx.rope` for rotary position embeddings
   - Implements functional KV cache updates (compatible with `torch.export`)
   - Supports Grouped Query Attention (GQA) via `repeat_interleave`

3. **Pattern Matching** during export:
   - `scaled_dot_product_attention` → MLX's fused SDPA kernel
   - KV cache updates → MLX's index update ops
   - `dequantize_affine + linear` → MLX's quantized matmul

### `export_llm_hf` pipeline

The `export_llm_hf` script uses optimum-executorch's `CausalLMExportableModule` by default. When custom flags are enabled, it uses `TorchExportableModuleWithStaticCache` from HuggingFace transformers, with optional MLX-specific replacements:

- `--use-custom-sdpa`: Registers `mlx::custom_sdpa` attention implementation
- `--use-custom-kv-cache`: Replaces HF's `StaticCache` with `HFStaticCache` using `mlx::kv_cache_update`
