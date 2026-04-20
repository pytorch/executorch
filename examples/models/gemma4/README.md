# Gemma4 on ExecuTorch

[google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) on
ExecuTorch CPU + XNNPACK. **Status: text generation validated end-to-end —
output matches HuggingFace reference token-for-token** ("The capital of
France is **Paris**." for the canonical prompt).

Gemma4 E2B is a ~2B effective parameter model with 35 transformer layers,
sliding/full attention (window=512, mixed head_dim 256/512), YOCO KV sharing
(last 20 layers), per-layer input (PLI) gating, dual-theta RoPE (sliding θ=10k
default; full θ=1e6 proportional). Vision + audio modalities are deferred to a
later pass.

## Quick Start

### 1. Download and convert weights

```bash
huggingface-cli download google/gemma-4-E2B-it --local-dir ~/models/gemma-4-E2B-it

python -m executorch.examples.models.gemma4.convert_weights \
    ~/models/gemma-4-E2B-it \
    ~/models/gemma-4-E2B-it/model_et.pth
```

### 2. Export

The export driver uses Hydra config syntax. The Gemma4 sliding-window
pattern is auto-derived from `config/e2b_config.json:layer_types`, so you
do NOT need to pass `--local_global_attention` manually.

```bash
# FP32 XNNPACK (validated)
python -m executorch.extension.llm.export.export_llm \
    base.model_class=gemma4 \
    base.params=examples/models/gemma4/config/e2b_config.json \
    base.checkpoint=~/models/gemma-4-E2B-it/model_et.pth \
    'base.metadata="{\"get_bos_id\":2,\"get_eos_ids\":[1,106,50]}"' \
    model.use_sdpa_with_kv_cache=true \
    model.use_kv_cache=true \
    export.max_seq_length=512 \
    export.max_context_length=512 \
    backend.xnnpack.enabled=true
# Output: ./gemma4.pte (~19 GB)
```

The `base.metadata` embeds Gemma4's three EOS token IDs into the model so
the runner stops correctly: `1` (`<eos>`), `106` (`<turn|>`), `50`.
Without this the runner generates endless `<turn|>` tokens after the answer.

Quantized (`8da4w`) and CUDA paths exist via the same Hydra knobs but are
not yet validated for Gemma4 E2B numerical parity.

### 3. Build

```bash
# CPU + XNNPACK
make gemma4-cpu

# CUDA
make gemma4-cuda
```

### 4. Run

#### Simple prompt (built-in template)

```bash
./cmake-out/examples/models/gemma4/gemma4_runner \
    --model_path ./gemma4.pte \
    --tokenizer_path ~/models/gemma-4-E2B-it/tokenizer.json \
    --prompt "What is the capital of France?" \
    --seq_len 100
```

Expected output (stops cleanly at `<turn|>`, no runaway tokens):
```
<|turn>user
What is the capital of France?<turn|>
<|turn>model
<|channel>thought
<channel|>...<channel|>The capital of France is **Paris**.<turn|>
```

#### With official jinja chat template (system prompt / tool calls / thinking)

Use `render_chat.py` to render the official vLLM Gemma4 template, then pass
via `--raw_prompt`:

```bash
# Plain user message
PROMPT=$(python examples/models/gemma4/render_chat.py \
    --user "What is the capital of France?")

./cmake-out/examples/models/gemma4/gemma4_runner \
    --model_path ./gemma4.pte \
    --tokenizer_path ~/models/gemma-4-E2B-it/tokenizer.json \
    --raw_prompt --prompt "$PROMPT" \
    --seq_len 100

# With system prompt
PROMPT=$(python examples/models/gemma4/render_chat.py \
    --system "You are a concise assistant." \
    --user "What is the capital of France?")
./cmake-out/examples/models/gemma4/gemma4_runner ... --raw_prompt --prompt "$PROMPT"

# Reasoning / thinking mode
PROMPT=$(python examples/models/gemma4/render_chat.py \
    --user "Solve x^2 = 4" --enable-thinking)
./cmake-out/examples/models/gemma4/gemma4_runner ... --raw_prompt --prompt "$PROMPT"
```

The `render_chat.py` script uses `examples/models/gemma4/chat_template.jinja`
(the official vLLM template) and requires `jinja2` (`pip install jinja2`).

### 5. Verify per-layer parity vs HF

```bash
python examples/models/gemma4/tests/test_parity.py
```

All six tests must pass — token_embedding, rmsnorm, pli_inputs, rope
(sliding + full), decoder_layer_0, full_forward — with `max_abs_diff <= 6e-6`
and identical top-1 next-token vs HuggingFace `Gemma4ForConditionalGeneration`.

## Architecture

| Component | Details |
|-----------|---------|
| Text decoder | 35 layers, 1536 hidden, 8 heads (GQA 8:1) |
| Attention | 28 sliding (window=512, head_dim=256) + 7 full (head_dim=512) |
| KV sharing | YOCO: last 20 layers share KV from donors |
| PLI | Per-layer input gating (256-dim bottleneck per layer) |
| MLP | SwiGLU with GELU, double-wide on KV-shared layers |
| Vision | ViT encoder (768 hidden, 16 layers, patch=16) |
| Audio | Conformer encoder (1024 hidden, 12 layers) |

## Backend Support

| Backend | Validated | Quantization |
|---------|-----------|--------------|
| XNNPACK | ✅ fp32 (text) | 8da4w paths exist, parity not yet measured |
| Portable | ⚠️ paths exist, not validated | fp32 |
| CUDA | ⚠️ paths exist, not validated | fp32, bf16, 4w |

## Runner Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model_path` | `gemma4.pte` | Path to exported model |
| `--data_path` | | Path to CUDA data file (.ptd) |
| `--tokenizer_path` | `tokenizer.json` | Path to tokenizer |
| `--prompt` | `"Hello, how are you?"` | Text prompt |
| `--raw_prompt` | `false` | Pass `--prompt` verbatim (no template wrapping); use with `render_chat.py` |
| `--image_path` | | Optional image for multimodal |
| `--temperature` | `0.0` | Sampling temperature (0 = greedy) |
| `--seq_len` | `512` | Max new tokens |
| `--cpu_threads` | `-1` | CPU threads (-1 = auto) |
| `--target_size` | `896` | Image resize target |
| `--warmup` | `false` | Run warmup before generation |

## Performance (FP32 XNNPACK on CPU)

| Phase | Throughput |
|-------|------------|
| Prefill | ~56 tok/s |
| Decode | ~13 tok/s |

(Measured on the canonical "What is the capital of France?" prompt with the
14–19 GB unquantized FP32 .pte. Quantized variants and llama.cpp comparison
are pending.)

## Implementation notes

For full details see `PROGRESS.md`. Key Gemma4-specific deltas vs Gemma3
(now correctly handled in shared `examples/models/llama/`):

- **Per-layer-type RoPE** (dual-buffer): sliding uses θ=10k partial=1.0
  default; full uses θ=1e6 partial=0.25 `proportional`.
- **Mixed head dimensions**: 256 for sliding, 512 (`global_head_dim`) for
  full attention.
- **YOCO KV sharing** (last 20 layers) with per-type donor map.
- **Per-Layer Input (PLI)** bottleneck gate per layer, 256-dim, sourced from
  a separate `embed_tokens_per_layer` table.
- **Attention scale = 1.0** (no implicit `1/sqrt(head_dim)` divide).
- **Embedding scale** `sqrt(hidden_size)`, **final logit softcap** `c·tanh`.
- **`<|turn>...<turn|>`** chat template (NOT Gemma3's `<start_of_turn>`).
