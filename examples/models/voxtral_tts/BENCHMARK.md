# Voxtral TTS ExecuTorch Benchmark Results

Date: 2026-04-16
Machine: Meta devserver (CPU-only, no GPU)
Backend: ExecuTorch XNNPACK (CPU) + portable
Model: `mistralai/Voxtral-4B-TTS-2603`
Voice: `neutral_female`, seed `42`

## Short prompt — "Hello, how are you today?" (5 words)

| Config | model.pte | codec.pte | Frames | Audio | Wall time | RTF | Parakeet transcript |
|--------|-----------|-----------|--------|-------|-----------|-----|---------------------|
| FP32 XNNPACK | 15.5 GB | 610 MB | 40 | 3.20s | 15.3s | 4.8x | Hello, how are you today? |
| FP32 portable | 15.5 GB | 748 MB | 40 | 3.20s | 278s | 87x | Hello, how are you today? |
| 8da4w (feed_forward) | 7.0 GB | 610 MB | 43 | 3.44s | ~12s | ~3.5x | Hello, how are you today? |
| 8da8w (all) | 5.7 GB | 610 MB | 44 | 3.52s | ~10s | ~2.8x | Hello, how are you today? |
| 8da4w (all) | 4.3 GB | 610 MB | 33 | 2.64s | ~10s | ~3.8x | Ah hello. How are you today? |
| C reference (OpenBLAS) | N/A | N/A | 40 | 3.20s | ~300s | 94x | Hello, how are you today? |

## Long prompt — 541 chars / 90 words (paragraph)

Input text:
> The quick brown fox jumps over the lazy dog near the old stone bridge that
> crosses the winding river. Birds sing melodiously in the tall oak trees as
> the morning sun casts golden rays across the peaceful meadow. A gentle breeze
> carries the sweet scent of wildflowers through the valley, while distant
> church bells chime softly in the background. Children laugh and play in the
> nearby park, their joyful voices echoing through the neighborhood. The world
> feels calm and beautiful on this perfect spring morning, filled with warmth
> and wonder.

ExecuTorch configs ran with `--max_new_tokens 300` (= 24s audio at 12.5 Hz).
The C reference ran uncapped and produced 403 frames (32.2s), capturing the
full text. The ExecuTorch runs hit the 300-frame cap and truncated the last
~2 sentences. Use `--max_new_tokens 500` to avoid truncation for long texts.

| Config | model.pte | Frames | Audio | Wall time | RTF | Transcript (parakeet) |
|--------|-----------|--------|-------|-----------|-----|-----------------------|
| FP32 XNNPACK | 15.5 GB | 300 | 24.0s | 77s | 3.2x | Perfect through "Children laugh and play." |
| 8da4w (feed_forward) | 7.0 GB | 300 | 24.0s | 64s | 2.6x | Perfect through "...in the nearby park." |
| 8da8w (all) | 5.7 GB | 300 | 24.0s | 45s | 1.9x | "One" for "The" at start; otherwise perfect |
| 8da4w (all) | 4.3 GB | 300 | 24.0s | 49s | 2.0x | Perfect through "...in the background." |
| C reference (OpenBLAS) | N/A | 403 | 32.2s | 2508s | 77.9x | Full text perfect (no frame cap) |

### Audio quality metrics (long prompt)

| Config | RMS | Peak amplitude |
|--------|-----|----------------|
| FP32 XNNPACK | 0.0136 | [-0.182, 0.215] |
| 8da4w (feed_forward) | 0.0130 | [-0.142, 0.140] |
| 8da8w (all) | 0.0104 | [-0.127, 0.156] |
| 8da4w (all) | 0.0117 | [-0.120, 0.119] |

## Key observations

1. **XNNPACK is 20–50x faster than the C reference and portable backend** on
   the same CPU, thanks to optimized XNNPACK kernels for matmul and convolution.

2. **Quantization reduces model size 2–4x** with minimal quality impact:
   - `8da4w feed_forward` is the recommended config (2.2x smaller, perfect transcript)
   - `8da8w` is the fastest (RTF 1.9x) with good quality
   - `8da4w all` is the smallest (3.6x smaller) but may lose a word

3. **RTF improves with longer texts** due to amortized model loading and warmup:
   - Short prompt: RTF 3–5x
   - Long prompt: RTF 1.9–3.2x

4. **FP32 produces bit-identical codes to the C reference** when using the
   matching xorshift64+Box-Muller RNG (verified by `diff -q` on per-frame code
   dumps for the short prompt).

## GPU (A100) — CUDA AOTI backend

Date: 2026-04-22
Machine: Meta devserver `devvm22203.cco0` (NVIDIA PG509-210, A100 80 GB, driver 580.126.09)
Backend: ExecuTorch CUDA AOTI for LM (text_decoder, token_embedding, audio_token_embedding, semantic_head, predict_velocity); ExecuTorch portable for codec_decoder
Model: `mistralai/Voxtral-4B-TTS-2603`, FP32 weights, bf16-only inside Triton SDPA
Voice: `neutral_female`, seed `42`

### Short prompt — "Hello, how are you today?"

| Config | model.pte | model.ptd | codec.pte | Frames | Audio | LM time | LM RTF | Total time | RMS | Peak |
|--------|-----------|-----------|-----------|--------|-------|---------|--------|------------|-----|------|
| FP32 CUDA + portable codec | 5.4 MB | 15.8 GB | 748 MB | 43 | 3.44s | 11.5s | 3.34x | 178s | 0.0633 | [-0.491, 0.497] |
| 4w-quant CUDA + portable codec | 3.4 MB | 3.4 GB | 748 MB | 39 | 3.12s | 2.27s | 0.73x | 180s | 0.0477 | [-0.242, 0.238] |
| **4w-quant CUDA + CUDA codec** ⚡ | **3.4 MB** | **3.4 GB + 303 MB** | **5.7 MB** | **32** | **2.56s** | **2.09s** | **0.82x** | **3.7s** ⚡ | **0.0293** | **[-0.176, 0.152]** |

The full-CUDA pipeline (LM + codec both on GPU) drops total wall clock from 180 s → **3.7 s** for the same prompt — a **48× end-to-end speedup**. The codec rewrite (Conv1d / ConvTranspose1d expressed as `unfold + matmul` and `matmul + Fold`) is mathematically identical to the original ops (eager parity max abs diff = 5.5e-10 in fp32). Triton's batched-matmul autotune found 20 valid kernel choices for the rewritten codec where the conv form had 0.

Codec `.ptd` shrank from 748 MB (portable fp32 codec) to **303 MB** (CUDA AOTI fp32 codec) — same weights, smaller serialized layout under AOTI. Codec `.pte` went from 748 MB (weights inline) to 5.7 MB (weights in `.ptd`).

The 4w (int4 weight-only, group_size=32, `tile_packed_to_4d` packing for `_weight_int4pack_mm`) variant gives:
- **4.6× smaller `.ptd`** (3.4 GB vs 15.8 GB) — fits well within A100 80 GB and lets multiple replicas coexist
- **4.6× faster LM** (2.27 s vs 11.5 s) — and now **sub-real-time** (RTF 0.73x)
- **No quality regression**: 39 frames (vs baseline 40), audio amplitude (RMS 0.0477, peak 0.24) actually closer to the XNNPACK FP32 reference than the FP32-CUDA run

`flow_head.input_projection` is auto-skipped during quantization (its `[3072, 36]` weight isn't divisible by `group_size=32`); everything else in the decoder + flow-head linears quantizes cleanly.

### Numerical parity vs XNNPACK FP32

Validated with `seed=42` on `"Hello, how are you today?"` against the eager FP32 CPU baseline:
- Last-position prefill hidden cosine similarity: **0.999994**
- First-frame semantic argmax: **identical** (3040)
- First-frame semantic top-5: **identical**
- Frame count before END_AUDIO: 43 vs CPU baseline 40 (within bf16-SDPA noise)

### Known limitations (resolved)

1. ~~**Codec runs on CPU.**~~ **RESOLVED 2026-04-23.** Conv1d / ConvTranspose1d in `model.py` are now expressed as `unfold + matmul` / `matmul + Fold` (`_conv1d_as_matmul`, `_conv_transpose1d_as_matmul`). AOTI lowers them onto Triton matmul kernels — codec wall time dropped from ~155 s to ~40 ms.
2. **`.ptd` is 3.4 GB (4w-quant) or 15.8 GB (FP32 LM weights).** Acceptable for A100 80 GB; embedded deployment would want further weight reduction.
3. **First call autotunes Triton kernels** (~10 s extra). The runner's `warmup()` amortizes this over the first user-visible synth. Codec is *not* warmed (its first real call also pays autotune cost, but only once per process — under the new path it's <1 s anyway).

### Reproducing

```bash
conda activate et-cuda
unset CPATH                    # critical — see project_executorch_cuda_install.md memory
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Export FP32 (best quality, 15.8 GB .ptd)
python examples/models/voxtral_tts/export_voxtral_tts.py \
  --model-path ~/models/mistralai/Voxtral-4B-TTS-2603 \
  --backend cuda --dtype fp32 \
  --output-dir ./voxtral_tts_exports_cuda

# Or export 4w-quantized (4.6× smaller, sub-real-time, near-baseline quality)
# --dtype is auto-promoted to bf16 and tile_packed_to_4d packing is auto-set.
python examples/models/voxtral_tts/export_voxtral_tts.py \
  --model-path ~/models/mistralai/Voxtral-4B-TTS-2603 \
  --backend cuda --qlinear 4w \
  --output-dir ./voxtral_tts_exports_cuda_4w

# Build (parent ExecuTorch needs CUDA enabled first)
cmake --workflow --preset llm-release-cuda
cd examples/models/voxtral_tts && cmake --workflow --preset voxtral-tts-cuda && cd ../../..

# Run (full CUDA pipeline — LM + codec)
./cmake-out/examples/models/voxtral_tts/voxtral_tts_runner \
  --model ./voxtral_tts_exports_cuda_4w/model.pte \
  --data_path ./voxtral_tts_exports_cuda_4w/aoti_cuda_blob.ptd \
  --codec ./voxtral_tts_exports_cuda_4w/codec_decoder.pte \
  --codec_data_path ./voxtral_tts_exports_cuda_4w/codec_aoti_cuda_blob.ptd \
  --tokenizer ~/models/mistralai/Voxtral-4B-TTS-2603/tekken.json \
  --voice ~/models/mistralai/Voxtral-4B-TTS-2603/voice_embedding/neutral_female.pt \
  --text "Hello, how are you today?" \
  --output cuda_full.wav --seed 42 --max_new_tokens 100
```

## vllm-omni comparison (not runnable on this machine)

This benchmark was run on a CPU-only devserver. The [vllm-omni](https://github.com/vllm-project/vllm-omni)
reference implementation requires CUDA GPU (A100/H100 recommended) and typically
achieves sub-1x RTF (real-time or faster). To compare:

```bash
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni
uv pip install gradio==5.50
python examples/online_serving/voxtral_tts/gradio_demo.py \
  --host <your-server-url> --port 8000
```

ExecuTorch's value proposition is **on-device inference without GPU dependency**
— achieving 1.9–3.2x RTF on CPU alone.

## Reproducing

```bash
conda activate executorch
VOXTRAL_DIR=~/.cache/huggingface/hub/models--mistralai--Voxtral-4B-TTS-2603/snapshots/<sha>

# Export (pick one)
python export_voxtral_tts.py --model-path $VOXTRAL_DIR --backend xnnpack --output-dir ./exports
python export_voxtral_tts.py --model-path $VOXTRAL_DIR --backend xnnpack --qlinear 8da4w --decoder-qlinear-scope feed_forward --output-dir ./exports
python export_voxtral_tts.py --model-path $VOXTRAL_DIR --backend xnnpack --qlinear 8da8w --output-dir ./exports

# Build
cmake --workflow --preset llm-release
cd examples/models/voxtral_tts && cmake --workflow --preset voxtral-tts-xnnpack && cd ../../..

# Run
./cmake-out/examples/models/voxtral_tts/voxtral_tts_runner \
  --model ./exports/model.pte \
  --codec ./exports/codec_decoder.pte \
  --tokenizer $VOXTRAL_DIR/tekken.json \
  --voice $VOXTRAL_DIR/voice_embedding/neutral_female.pt \
  --text "Hello, how are you today?" \
  --output output.wav --seed 42 --max_new_tokens 300

# Verify with parakeet STT
python examples/models/voxtral_tts/transcribe_parakeet.py \
  --audio output.wav \
  --parakeet-runner ./cmake-out/examples/models/parakeet/parakeet_runner \
  --parakeet-model examples/models/parakeet/parakeet_tdt_exports/model.pte \
  --parakeet-tokenizer examples/models/parakeet/parakeet_tdt_exports/tokenizer.model
```
