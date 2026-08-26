# LLM MLX Example

This example demonstrates how to export and run LLMs using the MLX delegate for Apple Silicon.

## Features

- **Export**: Convert HuggingFace LLMs to ExecuTorch format with MLX delegate
- **Quantization**: Optional INT4/INT8 weight quantization via TorchAO
- **KV Cache**: Efficient KV cache implementation for autoregressive generation
- **Off-graph KV Cache**: Optional runtime-owned cache, sized and configured per run instead of at export
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
| `run_llm_hf.cpp` | C++ runner; the run path for off-graph-cache exports, and runs in-graph ones too |

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

# Off-graph KV cache: the cache is owned by the runtime, not baked into the .pte
python -m executorch.backends.mlx.examples.llm.export_llm_hf \
    --model-id "unsloth/gemma-3-1b-it" \
    --output gemma3_offgraph.pte \
    --use-offgraph-cache \
    --max-ctx-len 1024

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
| `--max-ctx-len` | `1024` | Maximum context length / KV cache capacity |
| `--dtype` | `bf16` | Model dtype (`fp32`, `fp16`, `bf16`) |
| `--qlinear` | None | Quantization for linear layers (`4w`, `8w`, `nvfp4`) |
| `--qembedding` | None | Quantization for embedding layers (`4w`, `8w`, `nvfp4`) |
| `--no-tie-word-embeddings` | `False` | Disable re-tying lm_head to embedding after quantization |
| `--use-custom-sdpa` | `False` | Use MLX custom SDPA (`mlx::custom_sdpa`) |
| `--use-custom-kv-cache` | `False` | Use MLX custom KV cache (`mlx::kv_cache_update`) |
| `--use-offgraph-cache` | `False` | Use the off-graph KV cache (`kvcache::update_and_attend`); replaces the two flags above |
| `--prefill-chunk-size` | `512` | Max tokens per forward step. Bounds the traced `seq_len` dimension and is published as `get_prefill_chunk_size` for the runner. It is also the largest single cache write, so a ring layer is sized `window + chunk - 1`; it may not exceed the sliding window or the context length. Ignored on the optimum-executorch path, which owns its own `seq_len` bound |

Off-graph exports keep no cache in the `.pte`, so the pybindings `run_llm_hf`
cannot run them — use [`mlx_run_llm_hf`](#mlx_run_llm_hf-c) below, which builds
the cache and binds it at load time.

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

## `mlx_run_llm_hf` (C++)

Native runner for models exported with `--use-offgraph-cache`. That cache lives
outside the graph and is handed to the backend by key at load time, which the
pybindings `run_llm_hf` above cannot do. It also runs in-graph `.pte` files
unchanged — omit `--kv-max-capacity` — so the same binary compares both cache
paths. Greedy decode.

### Build

A standalone `find_package(executorch)` project, so ExecuTorch must be installed
first:

```bash
cmake --preset mlx-release
cmake --build cmake-out --target install -j$(( $(sysctl -n hw.ncpu) - 1 ))

cmake -S backends/mlx/examples/llm -B cmake-out/backends/mlx/examples/llm \
    -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-out/backends/mlx/examples/llm -j$(( $(sysctl -n hw.ncpu) - 1 ))
```

The binary lands at `cmake-out/backends/mlx/examples/llm/mlx_run_llm_hf`.

### Run

```bash
python -m executorch.backends.mlx.examples.llm.export_llm_hf \
    --model-id unsloth/gemma-3-1b-it \
    --output gemma3_offgraph.pte \
    --use-offgraph-cache \
    --max-ctx-len 1024

cmake-out/backends/mlx/examples/llm/mlx_run_llm_hf \
    --pte gemma3_offgraph.pte \
    --tokenizer ~/.cache/huggingface/hub/models--unsloth--gemma-3-1b-it/snapshots/*/tokenizer.json \
    --chat gemma \
    --kv-max-capacity 1024 \
    --prompt "What is the capital of France?" \
    --max-new-tokens 50
```

`--chat` selects the instruct template — `llama3`, `gemma`, `gemma4`, or `0` for
raw text. It matters: raw text confuses an instruct model into emitting turn
markers, and using the wrong template invalidates a comparison between two
`.pte` files.

### Configuring the cache at run time

Only the cache *geometry* is fixed at export — how many caches, their KV heads,
head dims and windows, which the export publishes as constant methods the runner
reads before building the cache. Everything else is chosen per run, with no
re-export. The runner reports what it built:

```
[cache] off-graph seq | capacity=1024 initial=512 kv_dtype=BFloat16
        26 layers: 4 flat + 22 ring(window 512)
```

`--kv-windows` overrides the attention pattern, repeating a comma-separated list
over the caches (`0` = flat). The geometry each cache declared is untouched, so
this cannot desync from the graph. For gemma-3-1b, whose 26 layers are 22
sliding at 512 and 4 full:

```bash
# (omitted)                         26 layers: 4 flat + 22 ring(window 512)
--kv-windows 512,512,512,512,512,0  # the same, spelling out gemma-3's 5:1 period
--kv-windows 0                      # 26 layers: 26 flat
--kv-windows 512                    # 26 layers: 0 flat + 26 ring(window 512)
--kv-windows 512,256                # 26 layers: 0 flat + 13 ring(256) + 13 ring(512)
```

A ring layer allocates its whole `window + chunk - 1` slots up front, while a
flat layer starts at `--kv-initial-capacity` and doubles as the sequence grows,
so an all-flat cache can look smaller than a sliding one early in a run and
larger later. Prefill runs in steps of the chunk size the export published, so
the ring is bounded by the chunk rather than by the prompt: prefilling 24k
tokens in one step would need 541 MiB of ring here, against 24 MiB at the 512
default.

### Options

`--help` lists every flag with its default.

| Option | Default | Description |
|--------|---------|-------------|
| `--pte` | *(required)* | Path to .pte file |
| `--tokenizer` | *(required)* | Path to `tokenizer.json` |
| `--prompt` | `The quick brown fox` | Input prompt |
| `--max-new-tokens` | `50` | Tokens to generate, excluding the prompt |
| `--temperature` | `0` | Sampling temperature; 0 is greedy argmax, which is what makes two `.pte` files comparable |
| `--chat` | `llama3` | Chat template: `llama3`, `gemma`, `gemma4`, or `0` to disable |
| `--kv-max-capacity` | `0` | Off-graph: history the cache may hold. Setting it selects the off-graph path |
| `--kv-storage-dtype` | `bf16` | Off-graph: KV storage dtype (`bf16`, `fp16`, `fp32`) |
| `--kv-initial-capacity` | `-1` | Off-graph: starting pool size; grows by doubling up to capacity |
| `--kv-windows` | *(model's own)* | Off-graph: attention pattern override, e.g. `512` |
| `--interactive` | `false` | Multi-turn chat on stdin; off-graph only |
| `--warmup` | `false` | Run once before measuring, to absorb JIT and pool growth |

---

## Architecture

The `export_llm_hf` script uses optimum-executorch's `CausalLMExportableModule` by default. When custom flags are enabled, it uses `TorchExportableModuleWithStaticCache` from HuggingFace transformers, with optional MLX-specific replacements:

- `--use-custom-sdpa`: Registers `mlx::custom_sdpa` attention implementation
- `--use-custom-kv-cache`: Replaces HF's `StaticCache` with `HFStaticCache` using `mlx::kv_cache_update`
