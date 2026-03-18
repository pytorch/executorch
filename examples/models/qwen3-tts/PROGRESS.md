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

#### 5.5 Performance profiling: padding waste analysis

Profiling results for 91 real codes (metal_test_codes.bin):

| Stage | Actual (91 codes) | Padded (1200 codes) | Ratio |
|---|---|---|---|
| quantizer.decode | 0.003s | 0.005s | 1.7x |
| pre_transformer (8-layer attn) | 0.034s | 0.207s | 6x |
| upsample[0-1] (2x each) | 0.058s | 0.181s | 3x |
| **decoder[1-4] (vocoder convs)** | **0.94s** | **16.1s** | **17x** |
| **TOTAL** | **1.1s** | **17.2s** | **15.8x** |

Root cause: The decoder upsamples codes by 1920x through ConvTranspose1d layers.
Padding 91 codes to 1200 means processing 2.3M samples instead of 175K — a 13x
blowup in vocoder compute that dominates runtime.

Dynamic shape export fails due to CausalConvNet padding creating `math.ceil`
guard chains incompatible with `torch.export` symbolic shape constraints.

Solution: Multi-bucket export (`--bucket-sizes 75,150,300,600,1200`) with
nearest-bucket selection at runtime. For 91 codes, the 150 bucket is selected
instead of 1200, giving a proportional ~6-8x decode speedup.

#### 5.6 Decoder-only sanity runs from precomputed codec ids

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

## 2026-03-18

### 6) Decoder multi-bucket export (10.5x speedup)

Root cause of slow decoder: padding 91 codes to 1200 wastes 13x compute in
the vocoder's 1920x ConvTranspose1d upsample chain. Dynamic shapes fail due
to `math.ceil` guard chains in CausalConvNet.

Solution: export at multiple fixed `codes_len` values, pick the smallest
bucket >= actual length at runtime.

Command:

```bash
python examples/models/qwen3-tts/export_qwen3_tts.py \
  --converted-dir examples/models/qwen3-tts/qwen3_tts_artifacts \
  --backend xnnpack --qlinear 8da4w \
  --bucket-sizes 75,150,300,600,1200 \
  --output-dir examples/models/qwen3-tts/qwen3_tts_exports_8da4w_bucketed
```

Result: **PASS** — 5 `.pte` files produced (`model_75.pte` through `model_1200.pte`)

Benchmark (91 codes → 7.28s audio, 8da4w XNNPACK CPU):

| Bucket | Decode Time | Speedup vs 1200 |
|--------|-------------|-----------------|
| 75 | N/A | Too small (91 > 75) |
| **150 (selected)** | **3.1s** | **10.5x** |
| 300 | 6.4s | 5.1x |
| 600 | 15.2s | 2.1x |
| 1200 (old default) | 32.4s | 1.0x |

Scaling is near-linear with bucket size, confirming vocoder cost is
proportional to sequence length. Output quality is identical — 174720 samples
at 24000 Hz (7.28s) in both cases.

### 7) Talker export to ExecuTorch

The talker is architecturally identical to Qwen3 0.6B: 28-layer decoder-only
transformer with GQA, SiLU MLP, QK-norm, RoPE. Reused the existing
Llama/Qwen3 export infrastructure directly.

Actual architecture from weights (differs from HF config defaults):
- dim=1024, n_heads=16, n_kv_heads=8, head_dim=128, hidden_dim=3072
- Main talker: 28 layers, vocab_size=3072 (codec vocabulary)
- Code predictor: 5 layers, vocab_size=2048, 15 per-group embeddings/heads
- num_code_groups=16 (1 main + 15 sub, matching decoder's num_quantizers=16)

#### 7.1 Weight conversion

```bash
python examples/models/qwen3-tts/convert_talker_weights.py \
  --talker-checkpoint examples/models/qwen3-tts/qwen3_tts_artifacts/qwen3_tts_talker.pth \
  --output-dir examples/models/qwen3-tts/qwen3_tts_artifacts/talker_converted
```

Result: **PASS**

Artifacts:
- `talker_main.pth` (311 keys) — main backbone in Meta/Llama format
- `talker_code_predictor.pth` (56 keys) — code predictor backbone
- `talker_aux.pth` (37 keys) — text_projection, codec_head, per-group embeddings/heads

#### 7.2 Main talker export (8da4w, max_seq_len=256)

```bash
python examples/models/qwen3-tts/export_talker.py \
  --checkpoint examples/models/qwen3-tts/qwen3_tts_artifacts/talker_converted/talker_main.pth \
  --params examples/models/qwen3-tts/config/talker_config.json \
  --output-dir examples/models/qwen3-tts/qwen3_tts_exports_talker_8da4w_s256 \
  --backend xnnpack --qlinear 8da4w --max-seq-len 256
```

Result: **PASS** — `talker.pte` (259 MB)

#### 7.3 Code predictor export (8da4w, max_seq_len=32)

```bash
python examples/models/qwen3-tts/export_talker.py \
  --checkpoint examples/models/qwen3-tts/qwen3_tts_artifacts/talker_converted/talker_code_predictor.pth \
  --params examples/models/qwen3-tts/config/code_predictor_config.json \
  --output-dir examples/models/qwen3-tts/qwen3_tts_exports_talker_8da4w_s256 \
  --output-name code_predictor.pte \
  --backend xnnpack --qlinear 8da4w --max-seq-len 32
```

Result: **PASS** — `code_predictor.pte` (52 MB)

Note: `tok_embeddings.weight` and `output.weight` are missing (expected) —
the code predictor has 15 per-group embeddings/heads stored in `talker_aux.pth`.

#### 7.4 Talker benchmarks

max_seq_len has a large impact on KV cache attention cost:

| max_seq_len | Per-step latency | 91 steps total |
|-------------|------------------|----------------|
| 2048 | 269 ms/step | 24.5s |
| **256** | **64 ms/step** | **5.8s** |

Code predictor (max_seq_len=32): **7.2 ms/step**

#### 7.5 Projected end-to-end performance

All stages 8da4w XNNPACK on CPU, 91 codes (7.28s audio):

| Stage | Steps | Per-step | Total | % of time |
|-------|-------|----------|-------|-----------|
| Main talker | 91 | 64 ms | 5.8s | 31% |
| Code predictor | 1365 (91×15) | 7.2 ms | 9.8s | 53% |
| Decoder (bucket 150) | 1 | — | 3.1s | 16% |
| **Total** | | | **18.7s** | |

Comparison:

| Configuration | Total time | Speedup |
|---|---|---|
| Python baseline (all stages) | 58s | 1.0x |
| ExecuTorch 8da4w bucketed (all stages) | **18.7s** | **3.1x** |
| ExecuTorch decoder only (bucket 150 vs 1200) | 3.1s vs 32.4s | 10.5x |

### 8) Streaming decode (inspired by mlx-audio)

mlx-audio achieves realtime streaming by decoding audio incrementally every
~25 tokens instead of waiting for all codes. Applied the same approach:

```bash
python examples/models/qwen3-tts/streaming_generate.py \
  --decoder-dir examples/models/qwen3-tts/qwen3_tts_exports_8da4w_bucketed \
  --codes-path examples/models/qwen3-tts/metal_test_codes.bin \
  --chunk-size 25
```

| Mode | First audio | Total time | RTF |
|------|-------------|------------|-----|
| Streaming (25-code chunks, bucket 75) | **2.15s** | 6.68s | 1.09x RT |
| Non-streaming (all 91, bucket 150) | 3.97s | **3.97s** | 1.84x RT |
| Old baseline (all 91, bucket 1200) | 32.4s | 32.4s | 0.22x RT |

Streaming gives **2.15s first-audio latency** — user hears audio 1.8s sooner.
Non-streaming is faster total (less padding overhead from fewer decoder calls)
but has higher first-audio latency.

Key insight from mlx-audio: their streaming decoder maintains conv buffers
across chunks, avoiding redundant computation. Our chunked approach processes
each chunk independently (simpler but less efficient).

### Remaining work for 3s target

- [ ] C++ runner integration for talker prefill + decode orchestration
- [ ] Metal/GPU backend export (expected 3-5x speedup over CPU → ~4-6s)
- [ ] Code predictor optimization — currently 53% of total time (1365 steps).
  Options: batched/parallel inner loop, model distillation, or fewer code groups
- [ ] Text embedding + text_projection in C++ (currently requires Python)
- [ ] Prefill export (dynamic shape or bucketed) for prompt processing
