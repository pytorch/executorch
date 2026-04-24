# Voxtral TTS

Self-contained ExecuTorch implementation of
[Voxtral-4B-TTS-2603](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603),
a ~4B parameter text-to-speech model that produces 24 kHz mono audio from
text. Weights are loaded directly from the HuggingFace safetensors
checkpoint. Supports CPU (portable + XNNPACK) and CUDA backends.

## Overview

The pipeline has two stages: **export** (Python, once) and **inference**
(C++ runner, repeated). Export converts the HuggingFace checkpoint into a
`model.pte` (LM + flow head, 5 methods) plus a `codec_decoder.pte`. At
inference time, the C++ runner loads both `.pte` files, the tokenizer, and a
voice embedding, then synthesizes a `.wav` file.

The model has three components:
1. **Mistral 4B LLM decoder** — autoregressive text-to-hidden-states
2. **Flow Matching Head** (3-layer transformer) — hidden states to 37
   audio codebook tokens per frame via a 7-step Euler ODE
3. **Codec Decoder** (Conv1d / ConvTranspose1d stack + 8 transformer
   layers) — codebook tokens to 24 kHz waveform

## Prerequisites

- ExecuTorch installed from source (see [building from source](../../../docs/source/using-executorch-building-from-source.md))
- Model weights downloaded from HuggingFace. The directory should contain
  `params.json`, `consolidated.safetensors`, `tekken.json`, and
  `voice_embedding/` with one or more `.pt` voice files.
  ```bash
  huggingface-cli download mistralai/Voxtral-4B-TTS-2603 \
      --local-dir ~/models/Voxtral-4B-TTS-2603
  ```
- For CUDA: NVIDIA GPU with CUDA 12.8 or 12.9 toolkit (tested on A100 80GB
  / sm_80 and RTX 5080 / sm_120).
  Note: CUDA 13 is not supported (CUB 3.0 incompatibility in
  `backends/cuda/runtime/shims/sort.cu`).

## Export

Export produces `model.pte` and `codec_decoder.pte`. For CUDA, it also
produces `aoti_cuda_blob.ptd` and `codec_aoti_cuda_blob.ptd` containing
the compiled CUDA kernels and weights.

```bash
# CPU (XNNPACK, FP32)
python export_voxtral_tts.py \
    --model-path ~/models/Voxtral-4B-TTS-2603 \
    --backend xnnpack \
    --output-dir ./voxtral_tts_exports

# CUDA, 4-bit weight-only quant (recommended — sub-real-time on A100)
python export_voxtral_tts.py \
    --model-path ~/models/Voxtral-4B-TTS-2603 \
    --backend cuda \
    --qlinear 4w \
    --output-dir ./voxtral_tts_exports_cuda_4w
```

`--dtype` is auto-promoted to `bf16` and `--qlinear-packing-format` is
auto-set to `tile_packed_to_4d` when `--backend cuda --qlinear 4w` is
selected (required by AOTI's `_weight_int4pack_mm` kernel).

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path` | (required) | Local directory with `params.json` + `consolidated.safetensors` |
| `--backend` | `xnnpack` | `portable`, `xnnpack`, `cuda`, `cuda-windows` |
| `--dtype` | `fp32` | Model dtype: `fp32` or `bf16` (auto-promoted to bf16 when CUDA + `--qlinear`) |
| `--output-dir` | `./voxtral_tts_exports` | Output directory |
| `--max-seq-len` | `4096` | KV cache length |
| `--qlinear` | (none) | Linear layer quantization: `4w`, `8w`, `8da4w`, `8da8w` |
| `--qlinear-group-size` | `32` | Group size for linear quantization |
| `--qlinear-packing-format` | (auto) | `tile_packed_to_4d` (auto-set for CUDA + 4w) |
| `--decoder-qlinear-scope` | `all` | Scope decoder quant to `all`, `attention`, `feed_forward`, or `none` |
| `--qlinear-codec` | (none) | Quantize codec decoder linears: `4w`, `8w` |
| `--qembedding` | (none) | Embedding quantization: `4w`, `8w` (XNNPACK: not yet supported) |
| `--streaming` | off | Enable streaming codec chunking metadata |

### CUDA quantization configs

Validated on A100, `seed=42`, `"Hello, how are you today?"`:

| Config | model.ptd | LM time | Total wall | E2E RTF | Notes |
|---|---|---|---|---|---|
| `--backend cuda` | 15.8 GB | 11.5 s | 178 s | 51x | FP32 weights, codec on portable CPU |
| **`--backend cuda --qlinear 4w`** | **3.4 GB** | **2.1 s** | **3.7 s** | **0.88x** ⚡ | int4 weights, codec on CUDA |

### XNNPACK quantization configs

| Config | Scope | model.pte | RTF (long prompt) |
|---|---|---|---|
| `--qlinear 8da4w --decoder-qlinear-scope feed_forward` | FFN only | 7.0 GB | 2.6x |
| `--qlinear 8da8w` | all decoder | 5.7 GB | 1.9x |
| `--qlinear 8da4w` | all decoder | 4.3 GB | 2.0x |

## Build

ExecuTorch must be installed from source first (see
[Prerequisites](#prerequisites)). The `make` target handles building the
core libraries and the runner binary.

```bash
# CUDA (recommended)
make voxtral_tts-cuda

# CPU
make voxtral_tts-cpu
```

This builds ExecuTorch with the requested backend, then the runner binary
at `cmake-out/examples/models/voxtral_tts/voxtral_tts_runner`.

## Run

The runner requires:
- `model.pte` — exported LM + flow head (see [Export](#export))
- `codec_decoder.pte` — exported codec
- `tekken.json` — tokenizer from the model weights directory
- A `.pt` voice embedding from the model's `voice_embedding/` directory

For CUDA also pass `--data_path` and `--codec_data_path` for the AOTI
delegate `.ptd` files.

```bash
# CUDA, full pipeline
unset CPATH                                       # see Troubleshooting
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cmake-out/examples/models/voxtral_tts/voxtral_tts_runner \
    --model voxtral_tts_exports_cuda_4w/model.pte \
    --data_path voxtral_tts_exports_cuda_4w/aoti_cuda_blob.ptd \
    --codec voxtral_tts_exports_cuda_4w/codec_decoder.pte \
    --codec_data_path voxtral_tts_exports_cuda_4w/codec_aoti_cuda_blob.ptd \
    --tokenizer ~/models/Voxtral-4B-TTS-2603/tekken.json \
    --voice ~/models/Voxtral-4B-TTS-2603/voice_embedding/neutral_female.pt \
    --text "Hello, how are you today?" \
    --output output.wav \
    --seed 42 \
    --max_new_tokens 200
```

Output is **24 kHz mono 16-bit PCM**. Listen with `ffplay output.wav` or
`aplay output.wav`.

Or use the one-shot script that does export + build + run end to end:

```bash
bash examples/models/voxtral_tts/run_cuda_e2e.sh ~/models/Voxtral-4B-TTS-2603
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `model.pte` | Path to exported `model.pte` (LM + flow head) |
| `--data_path` | (none) | Path to LM `.ptd` (required for CUDA) |
| `--codec` | `codec_decoder.pte` | Path to exported codec `.pte` |
| `--codec_data_path` | (none) | Path to codec `.ptd` (required for CUDA codec export) |
| `--tokenizer` | `tekken.json` | Path to tokenizer JSON from the base model |
| `--voice` | (required) | Path to voice embedding `.pt` |
| `--text` | (required) | Prompt text to synthesize |
| `--output` | `output.wav` | Output WAV file path |
| `--seed` | `42` | RNG seed (semantic sampling + flow noise) |
| `--temperature` | `0.0` | Sampling temperature (0 = greedy) |
| `--max_new_tokens` | `2048` | Max audio frames to generate (~12.5 frames/sec) |
| `--streaming` | off | Chunked codec emission for lower per-chunk latency |
| `--speaker` | off | Pipe raw f32le PCM to stdout (e.g. `... --speaker \| aplay -f FLOAT_LE -r 24000 -c 1`) |

### Available voices

`neutral_female`, `neutral_male`, `casual_female`, `casual_male`,
`cheerful_female`, `ar_male`, `de_female`, `de_male`, `es_female`,
`es_male`, `fr_female`, `fr_male` (under `voice_embedding/` in the model
directory).

## License

The ExecuTorch example code in this directory is licensed under the
BSD-style license found in the repository root.

The [Voxtral-4B-TTS-2603](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603)
model weights and bundled voice embeddings are licensed by Mistral AI under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
(Creative Commons Attribution-NonCommercial 4.0 International). This means:

- **Attribution required**: credit Mistral AI and the voice data sources
  (EARS, CML-TTS, IndicVoices-R, Arabic Natural Audio datasets).
- **NonCommercial only**: the model weights may not be used for commercial
  purposes.

The NC restriction originates from the voice reference datasets used to train
the model. See the
[model card](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603) for
full details.

## Troubleshooting

- **`__cudaLaunch was not declared` during build**: `CPATH` is polluted with
  CUDA 13's include path. `unset CPATH` and rebuild. CUDA 13's
  `crt/host_runtime.h` has a 2-arg `__cudaLaunch` macro incompatible with
  nvcc 12.8's stub generation.
- **`GLIBCXX_3.4.30 not found` at runner load time**: AOTI `.so` files
  require a newer libstdc++ than `/lib64/libstdc++.so.6`. Set
  `LD_LIBRARY_PATH=$CONDA_PREFIX/lib` before launching the runner.
- **`aoti_cuda_backend` target not found at link time**: the parent
  ExecuTorch was built without CUDA. Use `make voxtral_tts-cuda` (which
  builds with `EXECUTORCH_BUILD_CUDA=ON`) instead of running cmake by hand.
- **`cannot find -lcuda` during `pip install -e .` or export (WSL2)**: the
  CUDA toolkit doesn't ship `libcuda.so` — on WSL2 the driver lib lives at
  `/usr/lib/wsl/lib/`. Prepend it (or `/usr/local/cuda/lib64/stubs`) to
  `LIBRARY_PATH` before invoking pip / the export script.
- **First call takes ~30–50 s**: Triton autotunes the LM matmul kernels on
  first run, then caches per-process. The runner's `warmup()` amortizes
  this so the first user-visible synth pays the cost once.
- **`pip install -e .` after pulling source changes**: the default
  `install_executorch.sh` does `pip install .`. Repo edits to
  `examples/models/voxtral_tts/` won't take effect until you reinstall as
  editable.

## Pre-exported artifacts

For users who want to skip the export step, ready-to-run CUDA artifacts
are available on the HuggingFace hub at
[`younghan-meta/Voxtral-4B-TTS-2603-ExecuTorch-CUDA`](https://huggingface.co/younghan-meta/Voxtral-4B-TTS-2603-ExecuTorch-CUDA).

ExecuTorch's CUDA backend uses AOTInductor, which bakes pre-compiled
cubins for the export-time GPU's compute capability into `*.ptd`. Cubins
are not compatible across architectures, so the repo ships per-arch
subfolders:

| Folder | Compute capability | Example GPUs |
|---|---|---|
| `sm80/` | `sm_80` (Ampere) | A100, A30 |
| `sm120/` | `sm_120` (Blackwell) | RTX 5080, RTX 5090 |

Find your GPU's arch with `nvidia-smi --query-gpu=compute_cap --format=csv`,
then `hf download ... --include 'sm80/*'` (or `sm120`). If your arch isn't
shipped, re-export on the target GPU with the command above — the AOTI
compile step writes cubins for the local arch.
