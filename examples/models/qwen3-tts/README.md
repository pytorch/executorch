## Qwen3-TTS (XNNPACK-first Bring-up)

This directory adds an initial ExecuTorch bring-up for
`Qwen/Qwen3-TTS-12Hz-0.6B-Base` with an XNNPACK-first path.

The current implementation is split into two stages:

1. **Code generation (Python helper):**
   - Uses `qwen_tts` runtime to generate discrete acoustic codes from text.
   - Supports voice-clone prompt inputs (`ref_audio`, `ref_text`).
2. **Waveform decode (ExecuTorch .pte):**
   - Exports the speech-tokenizer decoder to `model.pte`.
   - Runs the decoder through ExecuTorch (XNNPACK / portable) and writes WAV.

### Why this split

The full Qwen3-TTS talker autoregressive generation stack is not yet exported in
this first bring-up. XNNPACK validation therefore focuses on the decode stage
that maps codebook tokens to waveform samples.

## Files

- `convert_weights.py`: converts HF snapshot into decoder/talker checkpoint artifacts.
- `export_qwen3_tts.py`: exports decoder path to ExecuTorch.
- `generate_codes.py`: generates codec tokens from text (and optional clone prompt).
- `main.cpp`, `qwen3_tts_runner.*`: C++ runner that can invoke helper + decode.

## Prerequisites

- ExecuTorch built from source.
- Conda env `executorch`.
- `qwen-tts` installed in that env.
- Access to `Qwen/Qwen3-TTS-12Hz-0.6B-Base` on Hugging Face
  (local snapshot or online download).

## 1) Convert HF weights

```bash
python examples/models/qwen3-tts/convert_weights.py \
  Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  examples/models/qwen3-tts/qwen3_tts_artifacts \
  --model-id-or-path Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --save-talker
```

## 2) Export decoder to ExecuTorch (XNNPACK)

```bash
python examples/models/qwen3-tts/export_qwen3_tts.py \
  --converted-dir examples/models/qwen3-tts/qwen3_tts_artifacts \
  --backend xnnpack \
  --fixed-codes-len 1200 \
  --output-dir examples/models/qwen3-tts/qwen3_tts_exports_fp32
```

Quantized experiment (example):

```bash
python examples/models/qwen3-tts/export_qwen3_tts.py \
  --converted-dir examples/models/qwen3-tts/qwen3_tts_artifacts \
  --backend xnnpack \
  --fixed-codes-len 1200 \
  --qlinear 8da4w \
  --output-dir examples/models/qwen3-tts/qwen3_tts_exports_8da4w
```

## 3) Build runner

```bash
make qwen3-tts-cpu
```

Runner binary:

```text
cmake-out/examples/models/qwen3-tts/qwen3_tts_runner
```

## 4) Run end-to-end text -> wav

```bash
conda run -n executorch cmake-out/examples/models/qwen3-tts/qwen3_tts_runner \
  --model_path examples/models/qwen3-tts/qwen3_tts_exports_fp32/model.pte \
  --text "Hello from ExecuTorch Qwen3 TTS." \
  --language English \
  --model_id_or_path Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --helper_script examples/models/qwen3-tts/generate_codes.py \
  --output_wav examples/models/qwen3-tts/output.wav
```

Voice clone example:

```bash
conda run -n executorch cmake-out/examples/models/qwen3-tts/qwen3_tts_runner \
  --model_path examples/models/qwen3-tts/qwen3_tts_exports_fp32/model.pte \
  --text "This is a voice clone test." \
  --language English \
  --model_id_or_path Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --ref_audio /path/to/ref.wav \
  --ref_text "reference transcript here" \
  --helper_script examples/models/qwen3-tts/generate_codes.py \
  --output_wav examples/models/qwen3-tts/output_clone.wav
```

## Notes

- Export currently uses static `--fixed-codes-len` due dynamic-shape guard issues.
- All experiment commands and outcomes are tracked in `PROGRESS.md`.
- Architecture and repository research context is tracked in `CONTEXT.md`.
