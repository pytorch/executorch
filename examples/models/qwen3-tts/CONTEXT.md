# Qwen3-TTS Bring-up Context

## Scope

- Target model: `Qwen/Qwen3-TTS-12Hz-0.6B-Base`
- Target path: `examples/models/qwen3-tts`
- Backend: XNNPACK (CPU)

## Reference patterns used

### 1) Qwen conversion/export patterns

- `examples/models/qwen3/convert_weights.py`
  - HF checkpoint conversion style with shard handling.
- `examples/models/qwen3_5/convert_weights.py`
  - strict key mapping behavior and defensive conversion logic.
- `examples/models/qwen3_5/tests/test_convert_weights.py`
  - focused conversion unit tests for mapping and unknown keys.

### 2) Speech model export/runtime patterns

- `examples/models/voxtral_realtime/export_voxtral_rt.py`
  - multi-method export wrappers.
  - backend split and metadata in `constant_methods`.
- `examples/models/voxtral_realtime/voxtral_realtime_runner.cpp`
  - custom C++ runner using `executorch::extension::Module`.
- `examples/models/whisper/main.cpp`
  - ASR runtime ergonomics and preprocessor handoff.

### 3) Build integration patterns

- `examples/models/whisper/CMakeLists.txt`
- `examples/models/whisper/CMakePresets.json`
- top-level `Makefile`

### 4) Backend support references

- `examples/models/MODEL_BACKEND_SUPPORT.md`
  - confirms XNNPACK as the practical first backend target for CPU bring-up.
  - speech model examples currently emphasize CUDA/Metal; this bring-up closes a
    gap for CPU-oriented TTS decode execution.

## Repository observations (examples/models survey)

- Existing audio examples are STT-focused (`whisper`, `parakeet`, `voxtral_realtime`).
- No first-class generic TTS runner existed before this bring-up.
- Existing reusable primitive for speech output generation is closest in
  tokenizer/codec decoder stacks (not yet standardized as a shared TTS runtime).

## Qwen3-TTS package observations

- `Qwen3TTSModel.generate_voice_clone(...)` performs:
  - text/ref prompt packing,
  - talker generation of codec tokens,
  - speech tokenizer decode into waveform.
- Speech tokenizer decode path for 12Hz variant is represented by
  `Qwen3TTSTokenizerV2Decoder` and can run from codebook tokens.
- Full talker generation export to ExecuTorch is significantly larger in scope
  (autoregressive + sub-talker generation path and cache/state flow).

## Bring-up design choice

To get XNNPACK validation first:

- Export the **speech-tokenizer decoder** into ExecuTorch.
- Keep **codec generation** in Python helper using upstream `qwen_tts`.
- Add a C++ runner that:
  - optionally invokes helper (`text -> codec ids`)
  - then decodes codec ids through exported `model.pte` (`codec ids -> wav`).

This keeps the path runnable and measurable while preserving room to move
talker generation into ExecuTorch in a follow-up phase.

## Implemented architecture map

### Conversion layer

- `convert_weights.py`
  - pulls local or remote HF snapshots.
  - reads safetensor shards and extracts:
    - speech decoder weights (`decoder.*` from `speech_tokenizer/`)
    - optional talker weights (`talker.*` from root model)
  - writes `decoder_metadata.json` for export/runtime contracts.

### Export layer

- `model.py`
  - defines `Qwen3TTSSpeechDecoderExport` wrapper.
  - computes output lengths from codec tokens and runs decoder forward.
- `export_qwen3_tts.py`
  - lowers wrapper to ExecuTorch.
  - attaches `constant_methods` metadata:
    - `output_sample_rate`
    - `decode_upsample_rate`
    - `num_quantizers`
    - `codebook_size`
    - `fixed_codes_len`
  - supports fp32/bf16 and optional 8da4w quant for linear layers.

### Runtime layer

- `generate_codes.py`
  - uses upstream `Qwen3TTSModel` for text->codec generation.
  - supports:
    - text-only mode (fallback x-vector prompt from generated silence)
    - voice clone mode (`ref_audio` + optional `ref_text`)
  - emits compact binary codec file consumed by C++ runner.
- `qwen3_tts_runner.cpp`
  - loads exported decoder `.pte`.
  - optionally invokes helper script for codec generation.
  - pads codec sequence to `fixed_codes_len` and decodes waveform.
  - writes PCM16 WAV output.

## Why fixed-length export is used

- Initial dynamic-shape export failed with `torch.export` constraint violations
  on `codes_len` for decoder internals.
- Static export (`fixed_codes_len=1200`) was adopted to unblock XNNPACK
  execution.
- Runner-side padding with sentinel `-1` preserves true output trimming through
  decoder length metadata.

## Follow-up work suggested by this bring-up

1. Move talker autoregressive generation into ExecuTorch methods
   (prefill/decode-step style).
2. Investigate BF16 decode runtime stall observed in current experiments.
3. Add Metal backend support for the speech decoder.
4. Replace helper-script dependency with fully in-runner ExecuTorch graph path.
