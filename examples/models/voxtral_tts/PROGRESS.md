# Voxtral TTS Progress Handoff

Single-source handoff for `examples/models/voxtral_tts`. Written so work can
be resumed on another machine without prior chat history.

Last updated: 2026-04-16

## Current state: WORKING (CPU portable + XNNPACK, FP32 + quantized)

End-to-end ExecuTorch runner produces intelligible speech verified by parakeet
STT. Offline, streaming, and live-playback (`--speaker`) modes all work.

| Backend | Quant | model.pte | RTF (short) | RTF (long) | Transcript |
|---------|-------|-----------|-------------|------------|------------|
| XNNPACK | fp32 | 15.5 GB | 4.8x | 3.2x | Hello, how are you today? |
| XNNPACK | 8da4w ff | 7.0 GB | ~3.5x | 2.6x | Hello, how are you today? |
| XNNPACK | 8da8w | 5.7 GB | ~2.8x | 1.9x | Hello, how are you today? |
| XNNPACK | 8da4w all | 4.3 GB | ~3.8x | 2.0x | Ah hello. How are you today? |
| Portable | fp32 | 15.5 GB | 87x | — | Hello, how are you today? |

FP32 frame codes are **bit-identical** to the C reference (`voxtral-tts.c`)
for all 40 frames. Waveform correlation with C ref is 0.9995.

## Bugs fixed (vs prior handoff)

1. **Codec reshape order** (`model.py:1150`) — `waveform.reshape(B, 1, P*T)`
   was patch-outer/frame-inner. Fixed to `waveform.transpose(1, 2).reshape(B,
   1, T * P)` (frame-outer/patch-inner matching C ref). This was the root
   cause of unintelligible audio.

2. **Flow-matching RNG** (`voxtral_tts_runner.cpp`) — replaced
   `std::normal_distribution` with xorshift64+Box-Muller matching the C
   reference. Without this, acoustic codes diverge by frame 1.

3. **ALiBi slopes** (`model.py:794`) — `_get_alibi_slopes` used `r**i`
   (starting at 1.0); fixed to `r**(i+1)` (starting at 0.5, matching ALiBi
   paper and C ref). Improved codec correlation from 0.998 to 0.9995.

4. **Runner stdout** (`voxtral_tts_runner.cpp`, `main.cpp`) — all info
   messages moved to stderr so `--speaker` mode outputs clean PCM to stdout.

5. **STT gate** (`verify_xnnpack_transcript.py`) — replaced Apple STT (macOS
   only) with parakeet runner (`transcribe_parakeet.py`) for cross-platform
   validation.

## Files changed

| File | Change |
|------|--------|
| `model.py` | Codec reshape fix + ALiBi slope fix |
| `voxtral_tts_runner.cpp` | xorshift64 RNG, stderr logging, VOXTRAL_DUMP_CODES env var, streaming RNG fix |
| `voxtral_tts_runner.h` | Added `flow_rng_state_` field |
| `main.cpp` | Added `--speaker` flag, stderr logging for speaker mode |
| `export_voxtral_tts.py` | Codec export comment clarification |
| `verify_xnnpack_transcript.py` | Parakeet STT, `--qlinear none` support |
| `transcribe_parakeet.py` | New: resample + parakeet runner helper |
| `BENCHMARK.md` | New: quantization + long-text benchmark results |
| `README.md` | Updated: quantization docs, streaming, live playback, runner options |

## Next steps: Metal and CUDA backends

The streaming architecture is backend-agnostic — `model_->execute()` calls are
the same regardless of backend. Adding Metal/CUDA requires:

1. **Export**: add `--backend metal` / `--backend cuda` paths to
   `export_voxtral_tts.py`, following `voxtral_realtime/export_voxtral_rt.py`.
2. **Build**: add CMake presets for `voxtral-tts-metal` / `voxtral-tts-cuda`
   in `CMakePresets.json`, and Makefile targets.
3. **Test**: re-run the acceptance gate with the new backend's .pte files.

No runner C++ changes needed — the runner is backend-transparent.

## Quick start on a new machine

```bash
conda activate executorch

# Download model (if not cached)
huggingface-cli download mistralai/Voxtral-4B-TTS-2603

# Export
VOXTRAL_DIR=~/.cache/huggingface/hub/models--mistralai--Voxtral-4B-TTS-2603/snapshots/<sha>
python export_voxtral_tts.py --model-path $VOXTRAL_DIR --backend xnnpack \
  --qlinear 8da4w --decoder-qlinear-scope feed_forward \
  --output-dir ./voxtral_tts_exports

# Build
cmake --workflow --preset llm-release
cd examples/models/voxtral_tts && cmake --workflow --preset voxtral-tts-xnnpack && cd ../../..

# Run
./cmake-out/examples/models/voxtral_tts/voxtral_tts_runner \
  --model ./voxtral_tts_exports/model.pte \
  --codec ./voxtral_tts_exports/codec_decoder.pte \
  --tokenizer $VOXTRAL_DIR/tekken.json \
  --voice $VOXTRAL_DIR/voice_embedding/neutral_female.pt \
  --text "Hello, how are you today?" \
  --output output.wav --seed 42

# Verify (requires parakeet exports built separately — see examples/models/parakeet/)
python examples/models/voxtral_tts/transcribe_parakeet.py \
  --audio output.wav \
  --parakeet-runner ./cmake-out/examples/models/parakeet/parakeet_runner \
  --parakeet-model examples/models/parakeet/parakeet_tdt_exports/model.pte \
  --parakeet-tokenizer examples/models/parakeet/parakeet_tdt_exports/tokenizer.model
```
