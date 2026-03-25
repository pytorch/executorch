## Qwen3-TTS

ExecuTorch implementation of `Qwen/Qwen3-TTS-12Hz-0.6B-Base`.

Supports three backends: **XNNPACK** (CPU), **Metal/AOTI** (Apple GPU), and **portable** (fallback).

### Performance

| Backend | 26 codes decode | Realtime | Export time |
|---------|----------------|----------|-------------|
| XNNPACK (8da4w quantized) | **728 ms** | 2.9x RT | ~20 min |
| Metal/AOTI (fp32) | **728 ms** | 2.9x RT | ~8 min |
| Portable (no backend) | 72,761 ms | 0.03x RT | ~2 min |

Model load + warmup: ~5-7s (one-time at startup).

Warm XNNPACK multi-prompt benchmark in one process (`top_k=50`, generation-only,
no WAV writes):

- Legacy host loop (`--disable_fused_cp_generate`): `3.61s`, `12.46s`, `21.56s`
- Fused `cp_generate` v2: `5.35s`, `12.92s`, `14.69s`
- The stable speed win is in `codegen_ms` per generated codec step:
  roughly `159/135/146 ms` down to `124/127/136 ms` on the benchmark prompts.
  End-to-end wall time still depends on how many codec steps the sampler emits.

### Model Sizes

| Config | Size |
|--------|------|
| XNNPACK 8da4w + 4w embedding | **1,027 MB** |
| XNNPACK 8da4w (no emb quant) | 2,065 MB |
| Metal fp32 (mixed w/ XNNPACK decoder) | 4,636 MB |

## Prerequisites

```bash
conda activate executorch
pip install qwen-tts

# For Metal backend only:
sudo mkdir -p /opt/llvm-openmp/lib
sudo ln -sf /opt/homebrew/Cellar/libomp/*/lib/libomp.dylib /opt/llvm-openmp/lib/libomp.dylib
```

## 1) Convert Weights

```bash
python examples/models/qwen3-tts/convert_weights.py \
  Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  examples/models/qwen3-tts/qwen3_tts_artifacts \
  --model-id-or-path Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --save-talker

python examples/models/qwen3-tts/convert_talker_weights.py \
  --talker-checkpoint examples/models/qwen3-tts/qwen3_tts_artifacts/qwen3_tts_talker.pth \
  --output-dir examples/models/qwen3-tts/qwen3_tts_artifacts/talker_converted
```

## 2) Export

### XNNPACK (CPU, quantized — recommended for mobile)

```bash
python examples/models/qwen3-tts/export_unified.py \
  --converted-dir examples/models/qwen3-tts/qwen3_tts_artifacts \
  --talker-dir examples/models/qwen3-tts/qwen3_tts_artifacts/talker_converted \
  --output-dir examples/models/qwen3-tts/qwen3_tts_exports_xnnpack \
  --backend xnnpack --qlinear 8da4w
```

### Metal/AOTI (Apple GPU — recommended for Mac)

```bash
python examples/models/qwen3-tts/export_unified.py \
  --converted-dir examples/models/qwen3-tts/qwen3_tts_artifacts \
  --talker-dir examples/models/qwen3-tts/qwen3_tts_artifacts/talker_converted \
  --output-dir examples/models/qwen3-tts/qwen3_tts_exports_metal \
  --backend metal --dtype fp32
```

Metal exports talker/code predictor to GPU, decoder stays on XNNPACK CPU
(Metal lacks `cumsum` fallback needed by the decoder).

### Portable (no acceleration — for debugging)

```bash
python examples/models/qwen3-tts/export_unified.py \
  --converted-dir examples/models/qwen3-tts/qwen3_tts_artifacts \
  --talker-dir examples/models/qwen3-tts/qwen3_tts_artifacts/talker_converted \
  --output-dir examples/models/qwen3-tts/qwen3_tts_exports_portable \
  --backend portable --dtype fp32
```

## 3) Generate Test Codes

```bash
python examples/models/qwen3-tts/generate_codes.py \
  --model-id-or-path Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --text "Hello from ExecuTorch." \
  --output-codes /tmp/hello_codes.bin
```

## 4) Build Runner

```bash
cmake --build cmake-out/examples/models/qwen3-tts --target qwen3_tts_unified_runner
```

The runner automatically links XNNPACK and Metal backends if built.

## 5) Run

### XNNPACK decode

```bash
cmake-out/examples/models/qwen3-tts/qwen3_tts_unified_runner \
  --model_path examples/models/qwen3-tts/qwen3_tts_exports_xnnpack/model.pte \
  --codes_path /tmp/hello_codes.bin \
  --output_wav /tmp/hello_xnnpack.wav
```

### XNNPACK text-only end to end

```bash
cmake-out/examples/models/qwen3-tts/qwen3_tts_unified_runner \
  --model_path examples/models/qwen3-tts/qwen3_tts_exports_unified/model.pte \
  --tokenizer_path examples/models/qwen3-tts/qwen3-tts-12Hz-0.6B-Base/tokenizer.json \
  --text "Hello from ExecuTorch." \
  --max_new_tokens 200 \
  --output_wav /tmp/hello_text.wav
```

### Warm XNNPACK multi-prompt benchmark

```bash
cmake-out/examples/models/qwen3-tts/qwen3_tts_unified_runner \
  --model_path examples/models/qwen3-tts/qwen3_tts_exports_unified/model.pte \
  --tokenizer_path examples/models/qwen3-tts/qwen3-tts-12Hz-0.6B-Base/tokenizer.json \
  --prompts_path examples/models/qwen3-tts/benchmark_prompts.txt \
  --repeat 1 \
  --max_new_tokens 128 \
  --temperature 1.0 \
  --top_k 50
```

To compare against the legacy host-side sub-code loop in the same binary, add:

```bash
  --disable_fused_cp_generate
```

### Metal decode

```bash
cmake-out/examples/models/qwen3-tts/qwen3_tts_unified_runner \
  --model_path examples/models/qwen3-tts/qwen3_tts_exports_metal/model.pte \
  --codes_path /tmp/hello_codes.bin \
  --output_wav /tmp/hello_metal.wav
```

### Play output

```bash
afplay /tmp/hello_xnnpack.wav
afplay /tmp/hello_metal.wav
```

## Architecture

Single `model.pte` with 7 named methods:

| Method | Backend | Purpose |
|--------|---------|---------|
| `encode_text` | Metal/XNNPACK | Text tokens → projected embeddings |
| `talker` | Metal/XNNPACK | 28-layer transformer with KV cache |
| `code_predictor` | Metal/XNNPACK | 5-layer sub-talker with KV cache |
| `codec_embed` | Portable | Codec token embedding lookup |
| `cp_head` | Metal/XNNPACK | Per-group LM head selection |
| `cp_generate` | Metal/XNNPACK | Fused sampled 15-step code predictor fast path |
| `decode_audio` | XNNPACK | Vocoder: codes → waveform (dynamic shapes) |

The runner calls `decode_audio` for codes→audio (decode-only mode) or orchestrates
all methods for text-only full text→audio synthesis through the assistant-wrapped
prompt contract used by the Python helper.

## Files

| File | Purpose |
|------|---------|
| `export_unified.py` | Multi-method export (XNNPACK/Metal/portable) |
| `main_unified.cpp` | CLI runner with decode-only and text modes |
| `qwen3_tts_unified_runner.*` | C++ runner with lazy loading and warmup |
| `generate_codes.py` | Python talker: text → codec tokens |
| `convert_weights.py` | HF → ExecuTorch weight conversion |
| `convert_talker_weights.py` | Talker weights to Llama format |
| `model.py` | Export wrappers and binary codec I/O |
| `metal_benchmark.md` | Metal backend benchmark results |
| `single_export.md` | Development progress log |

## Notes

- The decoder uses dynamic shapes with patched `CausalConvNet` padding
  (`math.ceil` → integer ceiling division for `torch.export` compatibility).
- XNNPACK has a one-time warmup cost on first call. The runner now exercises the
  full text path in `warmup_all()` so sequential prompt benchmarking reflects
  steady-state generation instead of cold delegate setup.
- Leading silence is automatically trimmed (`--trim_silence`, default on).
- Text-only `--text` mode now runs directly in `qwen3_tts_unified_runner` with
  dynamic prompt length, explicit prompt-budget checks, and assistant-wrapped
  prompt formatting aligned to `generate_codes.py`.
- Warm benchmark mode supports `--prompts_path`, `--repeat`, `--seed`, optional
  batch output writing via `--output_dir`, and `--disable_fused_cp_generate` for
  apples-to-apples comparisons.
- The recommended tokenizer path for local bring-up is
  `examples/models/qwen3-tts/qwen3-tts-12Hz-0.6B-Base/tokenizer.json`.
- Unified export manifests now record the text prompt contract and the current
  7-method surface, including `cp_generate`.
- Text mode still requires an external tokenizer path; tokenizer packaging is
  tracked in `TODO.md`.
- Metal/AOTI uses AOTInductor to compile graphs into `.so` with Metal kernels.
  Export takes ~8 min but runtime is GPU-accelerated.
- Voice clone / `ref_audio` / `ref_text`, full ICL prompting, and full sampling
  parity remain deferred. See `TODO.md` for the no-compromise backlog.
