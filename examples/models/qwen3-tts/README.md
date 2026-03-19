## Qwen3-TTS (XNNPACK)

ExecuTorch implementation of `Qwen/Qwen3-TTS-12Hz-0.6B-Base` with XNNPACK backend.

Two deployment modes:

1. **Unified single-PTE** (recommended for mobile): one `model.pte` with all
   pipeline stages (text encoding, talker, code predictor, decoder). Single file
   deployment with a C++ runner.
2. **Multi-file** (legacy): separate `.pte` files for decoder/talker/code predictor.

### Performance (Apple Silicon CPU, 8da4w quantized)

| Mode | Input | Decode time | Audio | Realtime factor |
|---|---|---|---|---|
| Unified (28 speech codes) | trimmed codes | **0.8s** | 2.2s | 2.8x RT |
| Unified (91 raw codes) | full codes | **2.0s** | 7.3s | 3.6x RT |

Model load + XNNPACK warmup: ~6s (one-time at app startup).

### Model sizes

| Config | Size | Notes |
|---|---|---|
| 8da4w + 4w embedding | **1,027 MB** | Recommended for mobile |
| 8da4w + 8w embedding | 1,176 MB | Better quality |
| 8da4w (no emb quant) | 2,065 MB | Full precision embeddings |

## Files

**Export:**
- `export_unified.py`: single-PTE multi-method export (recommended)
- `export_qwen3_tts.py`: decoder-only export (legacy bucketed)
- `export_talker.py`: talker/code predictor export (legacy)

**Runner:**
- `main_unified.cpp`, `qwen3_tts_unified_runner.*`: unified C++ runner
- `main.cpp`, `qwen3_tts_runner.*`: legacy decoder-only runner

**Model preparation:**
- `convert_weights.py`: converts HF snapshot into decoder/talker artifacts
- `convert_talker_weights.py`: converts talker weights to Meta/Llama format
- `generate_codes.py`: generates codec tokens from text (Python)
- `model.py`: decoder export wrapper and binary codec I/O

**Config:**
- `config/talker_config.json`: talker architecture (28L, dim=1024)
- `config/code_predictor_config.json`: code predictor architecture (5L, dim=1024)

## Prerequisites

```bash
conda activate executorch
pip install qwen-tts
```

Access to `Qwen/Qwen3-TTS-12Hz-0.6B-Base` on Hugging Face.

## Quick Start (Unified)

### 1) Convert weights

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

### 2) Export unified model

```bash
python examples/models/qwen3-tts/export_unified.py \
  --converted-dir examples/models/qwen3-tts/qwen3_tts_artifacts \
  --talker-dir examples/models/qwen3-tts/qwen3_tts_artifacts/talker_converted \
  --output-dir examples/models/qwen3-tts/qwen3_tts_exports_unified \
  --backend xnnpack --qlinear 8da4w --qembedding 4w
```

This produces a single `model.pte` (~1 GB) containing 6 methods:
`encode_text`, `talker`, `code_predictor`, `codec_embed`, `cp_head`, `decode_audio`.

### 3) Generate test codes

```bash
python examples/models/qwen3-tts/generate_codes.py \
  --model-id-or-path Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --text "Hello from ExecuTorch." \
  --output-codes /tmp/test_codes.bin \
  --trim-silence
```

### 4) Build runner

```bash
make qwen3-tts-cpu
```

### 5) Run

```bash
cmake-out/examples/models/qwen3-tts/qwen3_tts_unified_runner \
  --model_path examples/models/qwen3-tts/qwen3_tts_exports_unified/model.pte \
  --codes_path /tmp/test_codes.bin \
  --output_wav output.wav
```

The runner automatically trims leading silence and reports decode performance.

## Architecture

The unified `.pte` contains 6 named methods following the
[Parakeet multi-method pattern](../parakeet/):

```
text → tokenize → encode_text → projected embeddings
  → assemble composite prefill (codec control + text embeddings)
  → talker(prefill) → logits, hidden
  → loop until EOS:
      sample code_0, embed via codec_embed(group=0)
      code_predictor(prefill=[hidden, embed])
      for i in 1..15:
          cp_head(hidden, i-1) → sample code_i
          codec_embed(code_i, group=i) → embed
          code_predictor(step)
      sum all 16 embeds + text embed → next input
      talker(decode_step) → next logits, hidden
  → decode_audio(codes) → waveform → WAV
```

## Notes

- The decoder uses dynamic shapes (no bucketing needed). The `CausalConvNet`
  padding was patched to use integer ceiling division instead of `math.ceil`
  for `torch.export` compatibility.
- XNNPACK delegate initialization has a one-time ~5s cost per method on first
  call. The runner handles this via `warmup_decode()` during model loading.
- Leading silence in streaming mode codes is automatically trimmed by the
  runner (`--trim_silence`, default on).
- Full text-to-audio synthesis (`--text` mode) requires tiktoken C++ tokenizer
  integration (not yet implemented). Use `generate_codes.py` for now.
