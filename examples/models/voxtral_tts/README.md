# Voxtral-4B-TTS-2603 on ExecuTorch

Text-to-speech with [Voxtral-4B-TTS-2603](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603) running on ExecuTorch.

## Architecture

Three-component pipeline generating 24kHz audio from text:

1. **Mistral LLM** (~4B params) — autoregressive text-to-hidden-states
2. **Flow Matching Head** (3-layer transformer) — hidden states to 37 audio codebook tokens per frame via 7-step Euler ODE
3. **Codec Decoder** (Conv1d/ConvTranspose1d + 8 transformer layers) — codebook tokens to waveform

## Quick Start

### 1. Export

```bash
# Download model
huggingface-cli download mistralai/Voxtral-4B-TTS-2603 --local-dir ~/models/Voxtral-4B-TTS-2603

# FP32 XNNPACK (best quality)
python export_voxtral_tts.py \
    --model-path ~/models/Voxtral-4B-TTS-2603 \
    --backend xnnpack \
    --output-dir ./voxtral_tts_exports

# FP32 portable (CPU only)
python export_voxtral_tts.py \
    --model-path ~/models/Voxtral-4B-TTS-2603 \
    --backend portable \
    --output-dir ./voxtral_tts_exports
```

### CUDA backend (NVIDIA GPU)

Sub-real-time TTS on A100. The full pipeline (LM + codec) runs on GPU via
ExecuTorch's AOTI CUDA backend. End-to-end ~3.7 s wall clock for
`"Hello, how are you today?"` with `--qlinear 4w`.

```bash
# Pre-flight (one-time per shell):
unset CPATH                                   # critical, see "CUDA gotchas" below
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Export FP32 (best quality, 15.8 GB .ptd)
python export_voxtral_tts.py \
    --model-path ~/models/Voxtral-4B-TTS-2603 \
    --backend cuda --dtype fp32 \
    --output-dir ./voxtral_tts_exports_cuda

# Export 4w-quantized (RECOMMENDED — 4.6× smaller .ptd, sub-real-time)
# --dtype is auto-promoted to bf16; --qlinear-packing-format auto-set to tile_packed_to_4d.
python export_voxtral_tts.py \
    --model-path ~/models/Voxtral-4B-TTS-2603 \
    --backend cuda --qlinear 4w \
    --output-dir ./voxtral_tts_exports_cuda_4w

# Build (parent ExecuTorch needs CUDA enabled — use llm-release-cuda, not llm-release)
cmake --workflow --preset llm-release-cuda
cd examples/models/voxtral_tts && cmake --workflow --preset voxtral-tts-cuda && cd ../../..

# Run (full CUDA pipeline)
./cmake-out/examples/models/voxtral_tts/voxtral_tts_runner \
    --model ./voxtral_tts_exports_cuda_4w/model.pte \
    --data_path ./voxtral_tts_exports_cuda_4w/aoti_cuda_blob.ptd \
    --codec ./voxtral_tts_exports_cuda_4w/codec_decoder.pte \
    --codec_data_path ./voxtral_tts_exports_cuda_4w/codec_aoti_cuda_blob.ptd \
    --tokenizer ~/models/Voxtral-4B-TTS-2603/tekken.json \
    --voice ~/models/Voxtral-4B-TTS-2603/voice_embedding/neutral_female.pt \
    --text "Hello, how are you today?" \
    --output output.wav --seed 42 --max_new_tokens 100
```

Or use the one-shot script:

```bash
bash examples/models/voxtral_tts/run_cuda_e2e.sh ~/models/Voxtral-4B-TTS-2603
```

#### CUDA performance vs other backends

Headlines on A100 80 GB for `"Hello, how are you today?"` (`seed=42`):

| Backend | model.ptd | LM time | Total | E2E RTF |
|---|---|---|---|---|
| XNNPACK fp32 (CPU) | — | 3.2 s | 15.3 s | 4.8x |
| CUDA fp32 | 15.8 GB | 11.5 s | 178 s* | 51x* |
| **CUDA 4w + CUDA codec** | **3.4 GB** | **2.1 s** | **3.7 s** | **0.88x** ⚡ |

\* Pre conv-as-matmul codec rewrite; codec ran on portable CPU.

#### CUDA gotchas

1. **`unset CPATH` is mandatory.** If `CPATH` contains `/usr/local/cuda-13.0/...`, gcc picks CUDA 13's `crt/host_runtime.h` which has a 2-arg `__cudaLaunch` macro incompatible with nvcc 12.8's stub generation. Manifests as `__cudaLaunch was not declared` during the build. Verify with `echo $CPATH` (should be empty or only contain cuda-12.8).
2. **Use CUDA 12.8, not 13.0.** ExecuTorch's CUDA backend (`backends/cuda/runtime/shims/sort.cu`) was written against CUB 2.x; CUDA 13's CUB 3.0 breaks it.
3. **Set `LD_LIBRARY_PATH=$CONDA_PREFIX/lib`** before launching the runner. The AOTI `.so` files require GLIBCXX 3.4.30+ which conda's libstdc++ provides but `/lib64/libstdc++.so.6` does not.
4. **`pip install -e . --no-build-isolation`** after pulling source changes. The default `install_executorch.sh` does `pip install .` — repo edits to `examples/models/voxtral_tts/` won't take effect until you reinstall as editable.
5. **Use `llm-release-cuda` preset** for the parent build (not `llm-release`). The default preset doesn't enable `EXECUTORCH_BUILD_CUDA`, so `aoti_cuda_backend` won't exist when the runner CMake tries to link it.

### Quantization (XNNPACK)

Dynamic quantization reduces model size with minimal quality loss.

```bash
# 8da4w: feed_forward only (recommended — best quality/size tradeoff)
python export_voxtral_tts.py \
    --model-path ~/models/Voxtral-4B-TTS-2603 \
    --backend xnnpack \
    --qlinear 8da4w \
    --decoder-qlinear-scope feed_forward \
    --output-dir ./voxtral_tts_8da4w_ff

# 8da8w: all decoder layers
python export_voxtral_tts.py \
    --model-path ~/models/Voxtral-4B-TTS-2603 \
    --backend xnnpack \
    --qlinear 8da8w \
    --output-dir ./voxtral_tts_8da8w

# 8da4w: all decoder layers (most aggressive, smaller model)
python export_voxtral_tts.py \
    --model-path ~/models/Voxtral-4B-TTS-2603 \
    --backend xnnpack \
    --qlinear 8da4w \
    --output-dir ./voxtral_tts_8da4w
```

#### Quantization configs

| Config | Scope | model.pte | Quality |
|--------|-------|-----------|---------|
| fp32 | — | 15.5 GB | Best (reference) |
| `8da4w` | `feed_forward` | 7.0 GB | Excellent |
| `8da8w` | `all` | 5.7 GB | Excellent |
| `8da4w` | `all` | 4.3 GB | Good |

#### Quantization options

| Flag | Description |
|------|-------------|
| `--qlinear` | Quantize LLM decoder + flow head linear layers: `4w`, `8w`, `8da4w`, `8da8w` |
| `--qlinear-group-size` | Group size for linear quantization (default: auto) |
| `--decoder-qlinear-scope` | Scope decoder quantization: `all`, `attention`, `feed_forward`, `none` (default: `all`) |
| `--qlinear-codec` | Quantize codec decoder linear layers: `4w`, `8w` |
| `--qembedding` | Quantize embedding layers: `4w`, `8w` (XNNPACK: not yet supported) |

### 2. Build

```bash
# Build ExecuTorch core + XNNPACK
cmake --workflow --preset llm-release

# Build the runner (XNNPACK)
cd examples/models/voxtral_tts
cmake --workflow --preset voxtral-tts-xnnpack
cd ../../..

# Or portable (CPU only)
cd examples/models/voxtral_tts
cmake --workflow --preset voxtral-tts-cpu
cd ../../..
```

### 3. Run

```bash
# Offline (full generation then decode)
./cmake-out/examples/models/voxtral_tts/voxtral_tts_runner \
    --model voxtral_tts_exports/model.pte \
    --codec voxtral_tts_exports/codec_decoder.pte \
    --tokenizer ~/models/Voxtral-4B-TTS-2603/tekken.json \
    --voice ~/models/Voxtral-4B-TTS-2603/voice_embedding/neutral_female.pt \
    --text "Hello, how are you today?" \
    --output output.wav \
    --seed 42

# Streaming (incremental codec decoding, emits audio chunks as frames are generated)
./cmake-out/examples/models/voxtral_tts/voxtral_tts_runner \
    --model voxtral_tts_exports/model.pte \
    --codec voxtral_tts_exports/codec_decoder.pte \
    --tokenizer ~/models/Voxtral-4B-TTS-2603/tekken.json \
    --voice ~/models/Voxtral-4B-TTS-2603/voice_embedding/neutral_female.pt \
    --text "Hello, how are you today?" \
    --output output.wav \
    --streaming --seed 42
```

### Live playback

Use `--speaker` to write raw f32le PCM to stdout for real-time playback.
All log messages go to stderr so stdout is pure audio data.

```bash
# Linux: pipe to aplay
./cmake-out/examples/models/voxtral_tts/voxtral_tts_runner \
    --model voxtral_tts_exports/model.pte \
    --codec voxtral_tts_exports/codec_decoder.pte \
    --tokenizer ~/models/Voxtral-4B-TTS-2603/tekken.json \
    --voice ~/models/Voxtral-4B-TTS-2603/voice_embedding/neutral_female.pt \
    --text "Hello, how are you today?" \
    --output output.wav \
    --speaker --seed 42 | aplay -f FLOAT_LE -r 24000 -c 1

# macOS: pipe to ffplay
./cmake-out/examples/models/voxtral_tts/voxtral_tts_runner \
    ... --speaker | ffplay -f f32le -ar 24000 -nodisp -autoexit -

# Save raw PCM to file (convert later with ffmpeg)
./cmake-out/examples/models/voxtral_tts/voxtral_tts_runner \
    ... --speaker > output.raw 2>log.txt
ffmpeg -f f32le -ar 24000 -ac 1 -i output.raw output.wav
```

Streaming emits audio in chunks (first chunk ~0.4s, subsequent ~2s) as frames
are generated, enabling low-latency playback while generation continues.

### Runner options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `model.pte` | Path to LLM + acoustic head `.pte` |
| `--codec` | `codec_decoder.pte` | Path to codec decoder `.pte` |
| `--tokenizer` | `tekken.json` | Path to Tekken tokenizer |
| `--voice` | (neutral_female) | Voice preset name or path to `.pt` embedding |
| `--text` | (required) | Text to synthesize |
| `--output` | `output.wav` | Output WAV file path |
| `--seed` | `42` | Random seed for flow-matching noise |
| `--temperature` | `0.0` | Semantic sampling temperature (0 = greedy) |
| `--max_new_tokens` | `2048` | Max audio frames to generate |
| `--streaming` | off | Streaming mode with chunked codec decoding |
| `--speaker` | off | Write raw f32le PCM to stdout for live playback |

## Backend Support

| Backend | Status | Quantization |
|---------|--------|-------------|
| CPU (portable) | Supported | fp32 |
| XNNPACK | Supported | fp32, 8da4w, 8da8w, 4w, 8w |

## Exported Artifacts

Two `.pte` files:

- **model.pte** — Multi-method: `token_embedding`, `text_decoder`, `semantic_head`, `predict_velocity`, `audio_token_embedding`
- **codec_decoder.pte** — Audio codec decoder (Conv1d/ConvTranspose1d + transformers)

## Audio Parameters

- Sample rate: 24,000 Hz
- Frame rate: 12.5 Hz (1 codebook frame = 80ms audio)
- Codebooks: 37 per frame (1 semantic VQ-8192 + 36 acoustic FSQ-21)
- Flow matching: 7-step Euler ODE with classifier-free guidance (alpha=1.2)
