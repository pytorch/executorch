# Qwen3-TTS Bring-up Progress

This file records commands, outcomes, and observations for the bring-up.

## 2026-03-14

### Environment notes

- Conda env: `executorch`
- Installed package:
  - `qwen-tts==0.1.1`
- Noted dependency conflict warning:
  - `optimum-executorch 0.2.0.dev0 requires transformers==5.0.0rc1`
  - `qwen-tts` installation pulled `transformers==4.57.3`

### Status log

- [x] Scaffolded `examples/models/qwen3-tts` Python + C++ + CMake files.
- [x] Added conversion script for decoder/talker extraction from HF snapshots.
- [x] Added decoder export script for XNNPACK/portable.
- [x] Added helper for codec generation from text and optional clone prompt.
- [x] Added C++ runner to decode codec ids via exported `model.pte`.
- [x] Run conversion/export/build/runtime experiments.
- [x] Add non-quantized -> quantized experiment outcomes.

---

## Experiment log

### 1) Converter unit tests

Command:

```bash
conda run -n executorch python -m pytest examples/models/qwen3-tts/tests/test_convert_weights.py
```

Result: **PASS** (`3 passed`, ~1.8s)

### 2) Convert HF -> local artifacts

Command:

```bash
conda run -n executorch python examples/models/qwen3-tts/convert_weights.py \
  Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  examples/models/qwen3-tts/qwen3_tts_artifacts \
  --model-id-or-path Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --save-talker
```

Result: **PASS** (`elapsed ~83.5s`)

Artifacts:

- `qwen3_tts_decoder.pth`: `436M`
- `qwen3_tts_talker.pth`: `1.7G`
- `decoder_metadata.json`: `1.1K`

### 3) Export attempts (XNNPACK)

#### 3.1 Dynamic-shape export attempt

Command:

```bash
conda run -n executorch python examples/models/qwen3-tts/export_qwen3_tts.py \
  --converted-dir examples/models/qwen3-tts/qwen3_tts_artifacts \
  --backend xnnpack \
  --output-dir examples/models/qwen3-tts/qwen3_tts_exports_fp32
```

Result: **FAIL**

- Failure reason: `ConstraintViolationError` for dynamic `codes_len` guards in `torch.export`.
- Mitigation: switched to static `--fixed-codes-len` export.

#### 3.2 FP32 export (static length)

Command:

```bash
conda run -n executorch python examples/models/qwen3-tts/export_qwen3_tts.py \
  --converted-dir examples/models/qwen3-tts/qwen3_tts_artifacts \
  --backend xnnpack \
  --fixed-codes-len 1200 \
  --output-dir examples/models/qwen3-tts/qwen3_tts_exports_fp32
```

Result: **PASS** (`elapsed ~64.4s`)

Artifact:

- `qwen3_tts_exports_fp32/model.pte`: `440M`

#### 3.3 BF16 export (static length)

Command:

```bash
conda run -n executorch python examples/models/qwen3-tts/export_qwen3_tts.py \
  --converted-dir examples/models/qwen3-tts/qwen3_tts_artifacts \
  --backend xnnpack \
  --fixed-codes-len 1200 \
  --dtype bf16 \
  --output-dir examples/models/qwen3-tts/qwen3_tts_exports_bf16
```

Result: **PASS** (`elapsed ~46.1s`)

Artifact:

- `qwen3_tts_exports_bf16/model.pte`: `222M`

#### 3.4 8da4w quant export (static length)

Command:

```bash
conda run -n executorch python examples/models/qwen3-tts/export_qwen3_tts.py \
  --converted-dir examples/models/qwen3-tts/qwen3_tts_artifacts \
  --backend xnnpack \
  --fixed-codes-len 1200 \
  --qlinear 8da4w \
  --qlinear-group-size 32 \
  --output-dir examples/models/qwen3-tts/qwen3_tts_exports_8da4w
```

Result: **PASS** (`elapsed ~79.8s`)

Artifact:

- `qwen3_tts_exports_8da4w/model.pte`: `285M`

### 4) Build runner

Command:

```bash
make qwen3-tts-cpu
```

Result: **PASS** (`elapsed ~206.9s`)

Binary:

- `cmake-out/examples/models/qwen3-tts/qwen3_tts_runner`

### 5) Runtime checks

#### 5.1 FP32 text-only

Command:

```bash
conda run -n executorch cmake-out/examples/models/qwen3-tts/qwen3_tts_runner \
  --model_path examples/models/qwen3-tts/qwen3_tts_exports_fp32/model.pte \
  --text "Hello from ExecuTorch Qwen3 TTS." \
  --language English \
  --model_id_or_path Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --helper_script examples/models/qwen3-tts/generate_codes.py \
  --output_wav examples/models/qwen3-tts/output_text.wav
```

Result: **PASS** (`elapsed ~104.5s`)

- output: `output_text.wav`
- sample rate: `24000`
- frames: `76800`
- duration: `3.20s`
- file size: `150K`

#### 5.2 FP32 voice clone (`ref_audio` + `ref_text`)

Command:

```bash
conda run -n executorch cmake-out/examples/models/qwen3-tts/qwen3_tts_runner \
  --model_path examples/models/qwen3-tts/qwen3_tts_exports_fp32/model.pte \
  --text "This is a voice clone validation run." \
  --language English \
  --model_id_or_path Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --ref_audio poem.wav \
  --ref_text "This poem recording is the voice reference transcript." \
  --helper_script examples/models/qwen3-tts/generate_codes.py \
  --output_wav examples/models/qwen3-tts/output_clone.wav
```

Result: **PASS** (`elapsed ~100.6s`)

- output: `output_clone.wav`
- sample rate: `24000`
- frames: `88320`
- duration: `3.68s`
- file size: `173K`

#### 5.3 8da4w text-only

Command:

```bash
conda run -n executorch cmake-out/examples/models/qwen3-tts/qwen3_tts_runner \
  --model_path examples/models/qwen3-tts/qwen3_tts_exports_8da4w/model.pte \
  --text "Quantized decoder run." \
  --language English \
  --model_id_or_path Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --helper_script examples/models/qwen3-tts/generate_codes.py \
  --output_wav examples/models/qwen3-tts/output_text_8da4w.wav
```

Result: **PASS** (`elapsed ~71.1s`)

- output: `output_text_8da4w.wav`
- sample rate: `24000`
- frames: `53760`
- duration: `2.24s`
- file size: `105K`

#### 5.4 BF16 runtime

Command:

```bash
conda run -n executorch cmake-out/examples/models/qwen3-tts/qwen3_tts_runner \
  --model_path examples/models/qwen3-tts/qwen3_tts_exports_bf16/model.pte \
  --text "BF16 decoder run." \
  --language English \
  --model_id_or_path Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --helper_script examples/models/qwen3-tts/generate_codes.py \
  --output_wav examples/models/qwen3-tts/output_text_bf16.wav
```

Result: **FAIL / TIMEOUT**

- Process consumed CPU for ~8 minutes with no completed output artifact.
- Command was manually terminated.
- Follow-up needed to profile BF16 runtime behavior on this decoder graph.

#### 5.5 Decoder-only sanity runs from precomputed codec ids

FP32 command:

```bash
conda run -n executorch cmake-out/examples/models/qwen3-tts/qwen3_tts_runner \
  --model_path examples/models/qwen3-tts/qwen3_tts_exports_fp32/model.pte \
  --codes_path /var/folders/.../qwen3_tts_codegen_codes.bin \
  --output_wav examples/models/qwen3-tts/output_from_codes.wav
```

Result: **PASS** (`elapsed ~46.8s`, output `71K`)

8da4w command:

```bash
conda run -n executorch cmake-out/examples/models/qwen3-tts/qwen3_tts_runner \
  --model_path examples/models/qwen3-tts/qwen3_tts_exports_8da4w/model.pte \
  --codes_path /var/folders/.../qwen3_tts_codegen_codes.bin \
  --output_wav examples/models/qwen3-tts/output_from_codes_8da4w.wav
```

Result: **PASS** (`elapsed ~49.5s`, output `71K`)

BF16 command:

```bash
conda run -n executorch cmake-out/examples/models/qwen3-tts/qwen3_tts_runner \
  --model_path examples/models/qwen3-tts/qwen3_tts_exports_bf16/model.pte \
  --codes_path /var/folders/.../qwen3_tts_codegen_codes.bin \
  --output_wav examples/models/qwen3-tts/output_from_codes_bf16.wav
```

Result: **FAIL / TIMEOUT**

- Decoder-only BF16 path also stalled (>5 minutes at ~100% CPU) and was terminated.
- This indicates the issue is likely in BF16 decode execution itself, not helper code generation.
