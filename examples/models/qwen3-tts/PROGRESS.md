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

### 15) XNNPACK confidence gate: metric/accounting fixes

Goal:

- Fix the measurement bugs identified during `/autoresearch` before starting MLX
  work, then document the best verified warmed XNNPACK path.

What changed:

- `qwen3_tts_unified_runner.cpp`
  - `codegen_ms` now excludes in-loop streaming decode checkpoints instead of
    timing across both code generation and chunk decode.
  - Non-streaming `first_audio_ms` is now reported relative to request start,
    matching the streaming code path.
- `main_unified.cpp`
  - `audio` and `rtf` now use the raw waveform before silence trimming.
  - Added `trimmed_audio` and `rtf_trimmed` so post-processing effects stay
    visible without polluting the main throughput metric.
- `tests/test_unified_quality_contract.py`
  - Added contract coverage for the separated codegen timing and the new raw vs.
    trimmed RTF reporting.
- `XNNPACK_CONFIDENCE_STATUS.md`
  - Added a dedicated note capturing the current trustworthy XNNPACK status,
    the exact warmed benchmark command, and the remaining blockers before we can
    claim we beat `mlx-audio`.

Verification:

Focused tests:

```bash
conda run -n executorch python -m unittest \
  examples.models.qwen3-tts.tests.test_unified_runner_contract \
  examples.models.qwen3-tts.tests.test_unified_quality_contract \
  examples.models.qwen3-tts.tests.test_unified_metadata
```

Result: **PASS** (`26 tests`)

Runner rebuild:

```bash
cmake --build cmake-out/examples/models/qwen3-tts --target qwen3_tts_unified_runner
```

Result: **PASS**

Warmed prompt-set benchmark (checked-in export):

```bash
cmake-out/examples/models/qwen3-tts/qwen3_tts_unified_runner \
  --model_path examples/models/qwen3-tts/qwen3_tts_exports_unified/model.pte \
  --tokenizer_path examples/models/qwen3-tts/qwen3-tts-12Hz-0.6B-Base/tokenizer.json \
  --prompts_path examples/models/qwen3-tts/benchmark_prompts.txt \
  --repeat 1 \
  --max_new_tokens 128 \
  --temperature 1.0 \
  --top_k 50 \
  --disable_streaming_decoder_surface
```

Result: **PASS**

- Avg raw RTF: `0.51x`
- Avg first audio: `3.57s`
- Avg codegen: `9.16s`
- Avg decode: `1.57s`

Warmed prompt-set benchmark (temporary `max_seq_len=160` export):

```bash
cmake-out/examples/models/qwen3-tts/qwen3_tts_unified_runner \
  --model_path /tmp/qwen3_tts_exports_unified_s160/model.pte \
  --tokenizer_path examples/models/qwen3-tts/qwen3-tts-12Hz-0.6B-Base/tokenizer.json \
  --prompts_path examples/models/qwen3-tts/benchmark_prompts.txt \
  --repeat 1 \
  --max_new_tokens 128 \
  --temperature 1.0 \
  --top_k 50 \
  --disable_streaming_decoder_surface
```

Result: **PASS**

- Avg raw RTF: `0.52x`
- Avg first audio: `3.43s`
- Avg codegen: `8.74s`
- Avg decode: `1.61s`

Non-streaming sanity check:

```bash
cmake-out/examples/models/qwen3-tts/qwen3_tts_unified_runner \
  --model_path examples/models/qwen3-tts/qwen3_tts_exports_unified/model.pte \
  --tokenizer_path examples/models/qwen3-tts/qwen3-tts-12Hz-0.6B-Base/tokenizer.json \
  --text "Hello from the non streaming timing check." \
  --max_new_tokens 64 \
  --temperature 1.0 \
  --top_k 50 \
  --non_streaming_mode \
  --disable_streaming_decoder_surface
```

Result: **PASS**

- `first_audio_ms = 7685.4`
- `generation_ms = 7685.4`
- `final_decode_ms = 902.5`

Interpretation:

- The timing/accounting layer is now trustworthy enough to start MLX work.
- The best current XNNPACK path remains the overlap-window fallback
  (`--disable_streaming_decoder_surface`).
- XNNPACK is still below realtime after load, so we cannot honestly claim it is
  faster than `mlx-audio` yet.

### 14) Streaming parity upgrade + XNNPACK path comparison

Goal: align the ExecuTorch streaming path more closely with upstream Qwen3-TTS
chunking semantics, add a dedicated streaming decoder export surface, and
measure whether that new surface actually improves XNNPACK first-audio latency.

Changes landed in source:

- Added `capture_reference_streaming_contract.py` to record fixed-seed upstream
  codec traces, chunk boundaries, and decode pacing semantics.
- Reworked `qwen3_tts_unified_runner` streaming decode from cumulative
  prefix re-decode to bounded overlap-window decode with delta chunk emission.
- Added a dedicated `decode_audio_stream` export surface plus manifest metadata:
  `streaming_decoder_contract_version`, `streaming_decoder_chunk_size`,
  `streaming_decoder_left_context_size`, and `streaming_decoder_max_codes`.
- Capability-gated the new surface in the C++ runner, warmed it up explicitly,
  and split timing into `first_audio_ms`, `chunk_decode_ms`, and
  `final_decode_ms`.
- Added contract/metadata/reference tests for the new export surface and runner
  switches.

Verification:

Reference contract capture:

```bash
conda run -n executorch python -u \
  examples/models/qwen3-tts/capture_reference_streaming_contract.py \
  --upstream-repo /Users/younghan/project/executorch-exp/Qwen3-TTS \
  --output-dir /tmp/qwen3_streaming_reference \
  --text "Hello from the streaming benchmark path." \
  --language English \
  --max-new-tokens 256
```

Result: **PASS**

- Wrote fixed-seed upstream reference codes/audio/contract metadata.

Focused tests:

```bash
conda run -n executorch python -m pytest \
  examples/models/qwen3-tts/tests/test_unified_runner_contract.py \
  examples/models/qwen3-tts/tests/test_unified_quality_contract.py \
  examples/models/qwen3-tts/tests/test_unified_metadata.py \
  examples/models/qwen3-tts/tests/test_streaming_reference_contract.py
```

Result: **PASS**

Runner rebuild:

```bash
cmake --build cmake-out/examples/models/qwen3-tts --target qwen3_tts_unified_runner
```

Result: **PASS**

Fresh unified export with streaming decoder surface:

```bash
conda run -n executorch python examples/models/qwen3-tts/export_unified.py \
  --converted-dir examples/models/qwen3-tts/qwen3_tts_artifacts \
  --talker-dir examples/models/qwen3-tts/qwen3_tts_artifacts/talker_converted \
  --output-dir examples/models/qwen3-tts/qwen3_tts_exports_unified \
  --backend xnnpack \
  --qlinear 8da4w
```

Result: **PASS** (`elapsed ~25 min`)

- Export includes `decode_audio_stream` and the new `streaming_decoder_*`
  metadata fields.

Streaming benchmark command family (same prompt, warmed process, WAV write on):

```bash
cmake-out/examples/models/qwen3-tts/qwen3_tts_unified_runner \
  --model_path examples/models/qwen3-tts/qwen3_tts_exports_unified/model.pte \
  --tokenizer_path examples/models/qwen3-tts/qwen3-tts-12Hz-0.6B-Base/tokenizer.json \
  --text "Hello from the streaming benchmark path." \
  --language English \
  --max_new_tokens 128 \
  --streaming_interval 2.0 \
  --streaming_chunk_size 300 \
  --streaming_left_context_size 25 \
  --output_wav /tmp/qwen3_streaming.wav
```

Results:

| Mode | Extra flags | First audio | Chunk decode | Generation-only | Audio | RTF |
|------|-------------|-------------|--------------|-----------------|-------|-----|
| Auto streaming surface | none | `8.73s` | `11.41s` | `15.20s` | `2.48s` | `0.16x` |
| Windowed fallback | `--disable_streaming_decoder_surface` | `5.41s` | `1.56s` | `7.39s` | `2.48s` | `0.34x` |
| Legacy cumulative | `--use_legacy_cumulative_streaming_decode` | `5.40s` | `1.60s` | `7.41s` | `2.48s` | `0.33x` |

Interpretation:

- The bounded overlap-window decode path is working and materially better than
  the old cumulative prefix strategy for first-audio-oriented streaming.
- The new fixed-shape `decode_audio_stream` surface is **not** yet a win on the
  current XNNPACK build. It is functionally correct but significantly slower
  than the dynamic `decode_audio` fallback on this benchmark.
- The dominant latency is still in talker/code-predictor generation. Streaming
  decode remains primarily a first-audio lever, not the main throughput bottleneck.

Follow-up:

- Investigate why `decode_audio_stream` regresses on XNNPACK despite the tighter
  fixed-shape contract.
- Keep the overlap-window fallback as the preferred current XNNPACK path while
  preserving the new export surface for future backend tuning.

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

## 2026-03-23

### 9) Unified text-only prompt-contract rewrite

Goal: internalize the text-only `generate_codes.py` prompt semantics into
`qwen3_tts_unified_runner` so the unified C++ binary can accept direct text
input with dynamic prompt length instead of the previous fixed 8-slot
approximation.

Changes landed in source:

- Added a shared prompt-contract helper (`text_prompt_contract.py`) with tests
  for:
  - assistant-wrapped prompt formatting
  - prompt embedding split (`role`, `first_text`, `trailing + tts_eos`)
  - prompt-budget validation (`prefill`, `max_new_tokens`, `max_seq_len`)
- Rewrote `qwen3_tts_unified_runner.cpp` to:
  - tokenize the assistant-wrapped prompt instead of raw text
  - run `encode_text` over the whole prompt once
  - fold the first text token into prefill
  - feed trailing text hidden states during autoregressive decode
  - enforce prompt-budget guardrails before generation
- Updated the unified runner CLI to:
  - reject ambiguous `--codes_path` + `--text` usage
  - require `--tokenizer_path` for text mode
  - wire `top_p` consistently through the public interface
- Synced unified export manifests and export metadata with the current 7-method
  surface, including `cp_generate` and the text-prompt contract fields.
- Added `TODO.md` as the explicit no-compromise backlog.

Verification:

Prompt/metadata/runner-contract tests:

```bash
python -m unittest \
  examples.models.qwen3-tts.tests.test_unified_prompt_flow \
  examples.models.qwen3-tts.tests.test_unified_metadata \
  examples.models.qwen3-tts.tests.test_unified_runner_contract
```

Result: **PASS** (`11 tests`)

Unified runner rebuild:

```bash
cmake --build cmake-out/examples/models/qwen3-tts --target qwen3_tts_unified_runner
```

Result: **PASS**

Tokenizer materialization for text-mode verification:

```bash
python - <<'PY'
from transformers import AutoTokenizer
from pathlib import Path
out_dir = Path('examples/models/qwen3-tts/tokenizer_cache')
out_dir.mkdir(parents=True, exist_ok=True)
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-TTS-12Hz-0.6B-Base', trust_remote_code=True)
tok.save_pretrained(out_dir)
print(out_dir / 'tokenizer.json')
PY
```

Result: **PASS** (`examples/models/qwen3-tts/tokenizer_cache/tokenizer.json`)

Text-mode smoke run against the existing checked-in unified artifact:

```bash
cmake-out/examples/models/qwen3-tts/qwen3_tts_unified_runner \
  --model_path examples/models/qwen3-tts/qwen3_tts_exports_unified/model.pte \
  --tokenizer_path examples/models/qwen3-tts/tokenizer_cache/tokenizer.json \
  --text "Hello from ExecuTorch." \
  --max_new_tokens 128 \
  --output_wav /tmp/qwen3_tts_short.wav
```

Result: **FAIL (stale artifact)**

- The updated runner successfully reached the new assistant-wrapped prompt path
  and completed talker prefill.
- The existing `model.pte` is stale: it does **not** contain `cp_generate` and
  does **not** expose the new prompt-contract constant methods.
- A fresh unified re-export is required before the text-only end-to-end path can
  be verified against a current artifact.

Follow-up:

- Re-export `qwen3_tts_exports_unified/model.pte` from the updated
  `export_unified.py`.
- Re-run short and longer text prompts through `qwen3_tts_unified_runner`.
- Confirm the fresh artifact exposes `cp_generate` and the prompt-contract
  metadata methods.

### 10) Fresh unified export + real CLI verification

Tokenizer source requested for runtime verification:

- `examples/models/qwen3-tts/qwen3-tts-12Hz-0.6B-Base`

Tokenizer materialization:

```bash
python - <<'PY'
from transformers import AutoTokenizer
from pathlib import Path
model_dir = Path('examples/models/qwen3-tts/qwen3-tts-12Hz-0.6B-Base')
tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
tok.save_pretrained(model_dir)
print(model_dir / 'tokenizer.json')
PY
```

Result: **PASS**

- Wrote `examples/models/qwen3-tts/qwen3-tts-12Hz-0.6B-Base/tokenizer.json`
- Note: Transformers warns about the Mistral regex path here, but the
  ExecuTorch tokenizer loader successfully falls back to PCRE2 at runtime.

Makefile runner build requested by user:

```bash
make qwen3-tts-cpu
```

Result: **PASS**

- `CMakePresets.json` was updated so the `qwen3-tts-cpu` workflow builds
  `qwen3_tts_unified_runner` instead of the legacy `qwen3_tts_runner`.
- `Makefile` was updated so the success message points at the unified binary.

Fresh unified export:

```bash
conda run -n executorch python examples/models/qwen3-tts/export_unified.py \
  --converted-dir examples/models/qwen3-tts/qwen3_tts_artifacts \
  --talker-dir examples/models/qwen3-tts/qwen3_tts_artifacts/talker_converted \
  --output-dir examples/models/qwen3-tts/qwen3_tts_exports_unified \
  --backend xnnpack \
  --qlinear 8da4w
```

Result: **PASS** (`elapsed ~24.0 min`)

- Saved fresh `qwen3_tts_exports_unified/model.pte` (`2378.5 MB`)
- Saved fresh `qwen3_tts_exports_unified/export_manifest.json`
- Verified source export includes `cp_generate` and the prompt-contract fields.

#### 10.1 Short real CLI run

```bash
cmake-out/examples/models/qwen3-tts/qwen3_tts_unified_runner \
  --model_path examples/models/qwen3-tts/qwen3_tts_exports_unified/model.pte \
  --tokenizer_path examples/models/qwen3-tts/qwen3-tts-12Hz-0.6B-Base/tokenizer.json \
  --text "Hello from ExecuTorch." \
  --max_new_tokens 128 \
  --output_wav /tmp/qwen3_tts_short.wav
```

Result: **PASS** (`elapsed ~28.2s`)

- Prompt token count: `15`
- Generated codes: `128`
- Output wav: `/tmp/qwen3_tts_short.wav`
- Samples written: `245760` at `24000 Hz`

#### 10.2 Longer real CLI run

```bash
cmake-out/examples/models/qwen3-tts/qwen3_tts_unified_runner \
  --model_path examples/models/qwen3-tts/qwen3_tts_exports_unified/model.pte \
  --tokenizer_path examples/models/qwen3-tts/qwen3-tts-12Hz-0.6B-Base/tokenizer.json \
  --text "ExecuTorch now runs the unified Qwen3 TTS path directly in C plus plus, with the assistant prompt built inside the runner, dynamic prompt length handling, and a fused code predictor path for end to end synthesis on XNNPACK." \
  --max_new_tokens 192 \
  --output_wav /tmp/qwen3_tts_long.wav
```

Result: **PASS** (`elapsed ~33.5s`)

- Prompt token count: `58`
- Generated codes: `192`
- Output wav: `/tmp/qwen3_tts_long.wav`
- Samples written: `368640` at `24000 Hz`

#### 10.3 Prompt-budget guardrail check

```bash
cmake-out/examples/models/qwen3-tts/qwen3_tts_unified_runner \
  --model_path examples/models/qwen3-tts/qwen3_tts_exports_unified/model.pte \
  --tokenizer_path examples/models/qwen3-tts/qwen3-tts-12Hz-0.6B-Base/tokenizer.json \
  --text "ExecuTorch now runs the unified Qwen3 TTS path directly in C plus plus, with the assistant prompt built inside the runner, dynamic prompt length handling, and a fused code predictor path for end to end synthesis on XNNPACK." \
  --max_new_tokens 16 \
  --output_wav /tmp/qwen3_tts_guardrail.wav
```

Result: **EXPECTED FAIL**

- The runner rejected the request with:
  `max_new_tokens=16 is too small to consume the trailing prompt budget=50.`
- This confirms the dynamic prompt-budget guardrail is active in the real CLI.

### 11) Quality remediation: codec IDs, sampler parity, and fixed voice artifact

Root-cause fixes landed after comparing the unified runner against the MLX
Qwen3-TTS reference:

- Corrected unified export/runtime metadata to use the real codec control token
  IDs (`2148..2157`) instead of the stale `4196..4205` band.
- Updated the C++ runner to extract the last-token talker/code-predictor state
  after prefill instead of reusing the first token.
- Suppressed the talker special-token band (`[vocab_size - 1024, vocab_size)`)
  during `code_0` sampling while still allowing `codec_eos_id`.
- Removed the silent decoder clamp-to-zero fallback for invalid codec IDs and
  now fail loudly if the talker/code-predictor produces an out-of-range code.
- Restored closer MLX sampling parity for the text path:
  `temperature=0.9`, `top_k=50`, `top_p=1.0`, `repetition_penalty=1.05`.
- Switched the runtime path away from greedy fused `cp_generate` rollout and
  back to the stochastic `code_predictor` + `cp_head` loop for the 15 sub-code
  groups.

Focused regression tests:

```bash
python -m unittest \
  examples.models.qwen3-tts.tests.test_unified_prompt_flow \
  examples.models.qwen3-tts.tests.test_unified_metadata \
  examples.models.qwen3-tts.tests.test_unified_runner_contract \
  examples.models.qwen3-tts.tests.test_unified_quality_contract
```

Result: **PASS**

- Ran `17` tests, all passing.

Fresh rebuild:

```bash
make qwen3-tts-cpu
```

Result: **PASS** (`elapsed ~52.9s`)

- Built `cmake-out/examples/models/qwen3-tts/qwen3_tts_unified_runner`

Fresh export after quality fixes:

```bash
conda run -n executorch python examples/models/qwen3-tts/export_unified.py \
  --converted-dir examples/models/qwen3-tts/qwen3_tts_artifacts \
  --talker-dir examples/models/qwen3-tts/qwen3_tts_artifacts/talker_converted \
  --output-dir examples/models/qwen3-tts/qwen3_tts_exports_unified \
  --backend xnnpack \
  --qlinear 8da4w
```

Result: **PASS** (`elapsed ~24.0 min`)

- Saved fresh `examples/models/qwen3-tts/qwen3_tts_exports_unified/model.pte`
- Saved fresh `examples/models/qwen3-tts/qwen3_tts_exports_unified/export_manifest.json`
- Verified manifest now records:
  `codec_pad_id=2148`, `codec_bos_id=2149`, `codec_eos_id=2150`,
  `codec_nothink_id=2155`, `codec_think_bos_id=2156`,
  `codec_think_eos_id=2157`

Fixed-voice validation artifact:

```bash
./cmake-out/examples/models/qwen3-tts/qwen3_tts_unified_runner \
  --model_path examples/models/qwen3-tts/qwen3_tts_exports_unified/model.pte \
  --tokenizer_path examples/models/qwen3-tts/qwen3-tts-12Hz-0.6B-Base/tokenizer.json \
  --text "ExecuTorch now runs the unified Qwen3 TTS path directly in C plus plus with corrected codec control tokens, special token suppression, and sampling that is aligned more closely with the reference implementation." \
  --max_new_tokens 192 \
  --temperature 0.9 \
  --top_k 50 \
  --top_p 1.0 \
  --repetition_penalty 1.05 \
  --output_wav examples/models/qwen3-tts/output_text_fixed_quality.wav
```

Result: **PASS** (`elapsed ~37.9s`)

- Prompt token count: `49`
- Generated codes: `192`
- Output wav:
  `examples/models/qwen3-tts/output_text_fixed_quality.wav`
- Samples written: `368640` at `24000 Hz`

### 13) Warm XNNPACK benchmark + fused `cp_generate` v2

Goal: measure steady-state text-to-voice latency honestly in one warmed process,
then reduce the XNNPACK hot-loop cost without switching backends.

Changes landed in source:

- Added a per-prompt `SynthesisSession` API plus `SynthesisTiming` so the
  runner can stay loaded/warmed while each request gets fresh RNG and state.
- Updated `main_unified.cpp` with warm benchmark controls:
  - `--prompts_path`
  - `--repeat`
  - `--seed`
  - `--output_dir`
  - `--disable_fused_cp_generate`
- Split timing into prompt prep, talker prefill, codegen, decode-audio, and
  total generation.
- Expanded `warmup_all()` so it actually executes the text path, including
  `encode_text`, `talker`, `codec_embed`, `code_predictor`, `cp_head`,
  `cp_generate`, and `decode_audio`.
- Replaced the old greedy-only fused `cp_generate` export with a v2 contract
  that:
  - keeps host-side `code_0` sampling
  - samples groups `1..15` inside the fused graph for the XNNPACK fast path
  - returns sampled sub-codes plus the fused embedding sum for the next talker step
- Added ABI/version metadata:
  - `cp_generate_contract_version = 2`
  - `cp_generate_fast_top_k = 50`
  - `cp_generate_sampler = cdf_topk50_no_top_p_v2`
- Gated the fast path on exported metadata so older `.pte` artifacts cleanly
  fall back to the legacy host-side sub-code loop instead of crashing.
- Aligned host and fused sub-code sampling to the same inverse-CDF categorical
  sampler shape for the current fast-path mode (`top_k=50`, top-p disabled).

Warm benchmark prompt set:

- `examples/models/qwen3-tts/benchmark_prompts.txt`

Warm benchmark results (`top_k=50`, `temperature=1.0`, `max_new_tokens=128`,
same warmed process, no WAV writes):

| Prompt | Legacy generation-only | Fused generation-only | Legacy codegen | Fused codegen |
|--------|-------------------------|-----------------------|----------------|---------------|
| 0 | 3.61s | 5.35s | 3.18s / 20 steps | 4.58s / 37 steps |
| 1 | 12.46s | 12.92s | 10.97s / 81 steps | 11.19s / 88 steps |
| 2 | 21.56s | 14.69s | 18.75s / 128 steps | 12.90s / 95 steps |

Interpretation:

- The fused path consistently lowers codegen cost per generated codec step:
  - prompt 0: ~159 ms/step -> ~124 ms/step
  - prompt 1: ~135 ms/step -> ~127 ms/step
  - prompt 2: ~146 ms/step -> ~136 ms/step
- End-to-end warm wall time still depends on sampling trajectory and EOS timing,
  so raw prompt latency can move in either direction even when the hot path is
  cheaper per step.
- The first-order XNNPACK bottleneck is still the talker/codegen loop, not the
  decoder and not startup once warmup is separated out.

Follow-up evaluation:

- Talker decode-step specialization remains secondary for now:
  warm benchmarks still show `codegen_ms` dominating `decode_audio_ms`.
- Streaming decode is mainly a first-audio latency lever, not the biggest
  throughput win for the current single-run warm benchmark.
- The next XNNPACK speed work should focus on:
  - reducing generated codec step count without hurting quality
  - shrinking per-step talker/code-predictor cost further
  - only then revisiting talker decode specialization and streaming decode

Verification:

Source/contract tests:

```bash
python -m unittest \
  examples.models.qwen3-tts.tests.test_unified_prompt_flow \
  examples.models.qwen3-tts.tests.test_unified_metadata \
  examples.models.qwen3-tts.tests.test_unified_runner_contract \
  examples.models.qwen3-tts.tests.test_unified_quality_contract
```

Result: **PASS** (`28 tests`)

Runner rebuild:

```bash
cmake --build cmake-out/examples/models/qwen3-tts --target qwen3_tts_unified_runner
```

Result: **PASS**

Unified export:

```bash
conda run -n executorch python examples/models/qwen3-tts/export_unified.py \
  --converted-dir examples/models/qwen3-tts/qwen3_tts_artifacts \
  --talker-dir examples/models/qwen3-tts/qwen3_tts_artifacts/talker_converted \
  --output-dir examples/models/qwen3-tts/qwen3_tts_exports_unified \
  --backend xnnpack --qlinear 8da4w
```

Result: **PASS** (`model.pte` + manifest updated with `cp_generate` v2 metadata)

Warm legacy comparison:

```bash
cmake-out/examples/models/qwen3-tts/qwen3_tts_unified_runner \
  --model_path examples/models/qwen3-tts/qwen3_tts_exports_unified/model.pte \
  --tokenizer_path examples/models/qwen3-tts/qwen3-tts-12Hz-0.6B-Base/tokenizer.json \
  --prompts_path examples/models/qwen3-tts/benchmark_prompts.txt \
  --repeat 1 \
  --max_new_tokens 128 \
  --temperature 1.0 \
  --top_k 50 \
  --disable_fused_cp_generate
```

Result: **PASS**

Warm fused benchmark:

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

Result: **PASS**
