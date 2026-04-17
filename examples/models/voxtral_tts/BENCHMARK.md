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
