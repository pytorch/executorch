# LLM MLX Example

This example demonstrates how to export and run LLMs using the MLX delegate for Apple Silicon.

## Features

- **Export**: Convert HuggingFace LLMs to ExecuTorch format with MLX delegate
- **Quantization**: Optional INT4/INT8 weight quantization via TorchAO
- **KV Cache**: Efficient KV cache implementation for autoregressive generation
- **Custom Ops**: Uses `mlx::custom_sdpa` and `mlx::kv_cache_update` for optimal execution on MLX
- **Pybindings**: Run inference using ExecuTorch Python bindings
- **Gemma 4**: Text-only export and run flow supports processor-backed checkpoints such as `google/gemma-4-E2B-it`

## Requirements

```bash
pip install transformers optimum-executorch
```

## Scripts Overview

| Script | Description |
|--------|-------------|
| `export_llm_hf` | Export LLMs using optimum-executorch pipeline, with optional custom MLX SDPA/KV cache |
| `run_llm_hf` | Run exported models with token-by-token generation |

For exporting via the ExecuTorch LLM pipeline (e.g. `examples/models/llama`), use `--mlx` to enable the MLX delegate.

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

# With 4-bit quantization
python -m executorch.backends.mlx.examples.llm.export_llm_hf \
    --model-id "unsloth/Llama-3.2-1B-Instruct" \
    --output llama_hf_int4.pte \
    --use-custom-sdpa \
    --use-custom-kv-cache \
    --qlinear 4w \
    --qembedding 4w

# Gemma 4 text-only export
python -m executorch.backends.mlx.examples.llm.export_llm_hf \
    --model-id "google/gemma-4-E2B-it" \
    --output gemma4_hf_int4.pte \
    --use-custom-sdpa \
    --use-custom-kv-cache \
    --qlinear 4w
```

Gemma 4 support is currently validated for the text-only path using
`--use-custom-sdpa --use-custom-kv-cache --qlinear 4w`.

Validated with `transformers` commit
`61461a7bcb458db7cf6eeea49678b9ab776a7821`:

```bash
pip install -U "transformers @ git+https://github.com/huggingface/transformers.git@61461a7bcb458db7cf6eeea49678b9ab776a7821"
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model-id` | `unsloth/Llama-3.2-1B-Instruct` | HuggingFace model ID |
| `--output` | *(required)* | Output .pte file path |
| `--max-seq-len` | `1024` | Maximum sequence length for KV cache |
| `--dtype` | `bf16` | Model dtype (`fp32`, `fp16`, `bf16`) |
| `--qlinear` | None | Quantization for linear layers (`4w`, `8w`, `nvfp4`) |
| `--qembedding` | None | Quantization for embedding layers (`4w`, `8w`, `nvfp4`) |
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

Gemma 4 checkpoints may use `AutoProcessor` instead of `AutoTokenizer`; `run_llm_hf` now supports both paths automatically for text-only prompts.

Validated Gemma 4 run command:

```bash
python -m executorch.backends.mlx.examples.llm.run_llm_hf \
    --pte gemma4_hf_int4.pte \
    --model-id google/gemma-4-E2B-it \
    --prompt "What is the capital of France?" \
    --max-new-tokens 50
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--pte` | `llama_hf.pte` | Path to .pte file |
| `--model-id` | `unsloth/Llama-3.2-1B-Instruct` | HuggingFace model ID (for tokenizer or processor) |
| `--prompt` | `The quick brown fox` | Input prompt |
| `--max-new-tokens` | `50` | Maximum tokens to generate |

---

## Architecture

The `export_llm_hf` script uses optimum-executorch's `CausalLMExportableModule` by default. When custom flags are enabled, it uses `TorchExportableModuleWithStaticCache` from HuggingFace transformers, with optional MLX-specific replacements:

- `--use-custom-sdpa`: Registers `mlx::custom_sdpa` attention implementation
- `--use-custom-kv-cache`: Replaces HF's `StaticCache` with `HFStaticCache` using `mlx::kv_cache_update`
