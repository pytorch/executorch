# Voxtral Realtime

Self-contained ExecuTorch implementation of Mistral's
[Voxtral-Mini-4B-Realtime-2602](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602),
a ~4B parameter streaming speech-to-text model. No HuggingFace Transformers
dependency — weights are loaded directly from the Mistral checkpoint.
See [model.md](model.md) for architecture and implementation details.

## Overview

The pipeline has two stages: **export** (Python, once) and **inference**
(C++ runner, repeated). Export converts the Mistral checkpoint into a
`model.pte` file. A separate `preprocessor.pte` handles audio-to-mel
conversion. At inference time, the C++ runner loads both `.pte` files
and the Tekken tokenizer, then transcribes audio to text.

Two modes are supported: **streaming** (process 80ms chunks in real time,
including live microphone input, with unlimited duration) and **offline**
(encode full audio, then decode, bounded by `--max-seq-len`). The examples
below use streaming mode. Omit `--streaming` from export and run commands
for offline mode.

## Demo: streaming on Metal backend with microphone input

https://github.com/user-attachments/assets/44717dc5-777f-4710-ad55-5ec4fa04b9c4

Also, try a sample [standalone macOS app](https://github.com/meta-pytorch/executorch-examples/tree/main/voxtral_realtime/macos) to do real time transcription.

https://github.com/user-attachments/assets/6d6089fc-5feb-458b-a60b-08379855976a

## Prerequisites

- ExecuTorch installed from source (see [building from source](../../../docs/source/using-executorch-building-from-source.md))
- [safetensors](https://pypi.org/project/safetensors/) (`pip install safetensors`)
- Model weights downloaded from [HuggingFace](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602).
  The directory should contain `params.json`, `consolidated.safetensors`,
  and `tekken.json`.

## Preprocessor

Export a preprocessor `.pte` to convert raw audio into the format the
model expects:

```bash
python -m executorch.extension.audio.mel_spectrogram \
    --feature_size 128 \
    --streaming \
    --output_file ./voxtral_rt_exports/preprocessor.pte
```

For offline mode:

```bash
python -m executorch.extension.audio.mel_spectrogram \
    --feature_size 128 \
    --max_audio_len 300 \
    --output_file ./voxtral_rt_exports/preprocessor.pte
```

For MLX backend, use `--backend mlx`:

```bash
python -m executorch.extension.audio.mel_spectrogram \
    --feature_size 128 \
    --max_audio_len 300 \
    --backend mlx \
    --output_file ./voxtral_rt_exports/preprocessor.pte
```

For streaming, use a separate preprocessor with `--streaming` (no audio
length limit):

```bash
python -m executorch.extension.audio.mel_spectrogram \
    --feature_size 128 \
    --streaming \
    --output_file ./voxtral_streaming_exports/preprocessor.pte
```

For streaming with MLX backend:

```bash
python -m executorch.extension.audio.mel_spectrogram \
    --feature_size 128 \
    --streaming \
    --backend mlx \
    --output_file ./voxtral_streaming_exports/preprocessor.pte
```

## Export

Export produces a single `.pte` containing the audio encoder, text decoder,
and token embedding.

> [!TIP]
> Mistral has already published pre-exported `.pte` files for select backends, including macOS Metal, on their [HuggingFace Hub](https://huggingface.co/mistral-labs/Voxtral-Mini-4B-Realtime-2602-Executorch).

### XNNPACK (default)

```bash
python export_voxtral_rt.py \
    --model-path ~/models/Voxtral-Mini-4B-Realtime-2602 \
    --backend xnnpack \
    --streaming \
    --sliding-window 2048 \
    --output-dir ./voxtral_rt_exports \
    --qlinear-encoder 8da4w \
    --qlinear 8da4w \
    --qembedding 8w
```

### Backend support

| Backend | Offline | Streaming | Quantization |
|---------|---------|-----------|--------------|
| `xnnpack` | ✓ | ✓ | `4w`, `8w`, `8da4w`, `8da8w` |
| `metal` | ✓ | ✓ | none (fp32) or `fpa4w` (Metal-specific 4-bit) |
| `mlx` | ✓ | ✓ | `4w`, `8w`, `nvfp4` (NVIDIA FP4 dtype) |
| `cuda` | ✓ | ✓ | `4w`, `8w` |
| `cuda-windows` | ✓ | ✓ | `4w`, `8w` |
| `rocm` | ✓ | ✓ | BF16; packed linear `4w` and embedding `8w` |


MLX and Metal backends provide Apple GPU acceleration. CUDA provides NVIDIA GPU
acceleration, and experimental ROCm support provides AMD GPU acceleration, both
through AOTInductor.

#### CUDA export examples

Offline with int4 quantization:

```bash
python export_voxtral_rt.py \
    --model-path ~/models/Voxtral-Mini-4B-Realtime-2602 \
    --backend cuda \
    --dtype bf16 \
    --output-dir ./voxtral_rt_exports \
    --qlinear-encoder 4w \
    --qlinear-encoder-packing-format tile_packed_to_4d \
    --qlinear 4w \
    --qlinear-packing-format tile_packed_to_4d \
    --qembedding 8w
```

Streaming with int4 quantization:

```bash
python export_voxtral_rt.py \
    --model-path ~/models/Voxtral-Mini-4B-Realtime-2602 \
    --backend cuda \
    --dtype bf16 \
    --streaming \
    --output-dir ./voxtral_rt_exports \
    --qlinear-encoder 4w \
    --qlinear-encoder-packing-format tile_packed_to_4d \
    --qlinear 4w \
    --qlinear-packing-format tile_packed_to_4d \
    --qembedding 8w
```

#### ROCm export examples

ROCm support is experimental. Manual validation currently covers MI300X
(`gfx942`); CI canaries exercise `gfx950` and `gfx1100`. ROCm is never enabled
automatically. Use a ROCm PyTorch build with its matching Triton AMD backend;
do not run `install_executorch.sh`, because its dependency setup can replace
ROCm PyTorch with a CPU build.

The validated ROCm configurations use BF16 and optionally packed `4w` linear
weights with an `8w` embedding. The exporter rejects other ROCm dtype and
quantization combinations before loading the model.

Start with the BF16 streaming baseline:

```bash
python export_voxtral_rt.py \
    --model-path ~/models/Voxtral-Mini-4B-Realtime-2602 \
    --backend rocm \
    --dtype bf16 \
    --streaming \
    --sliding-window 2048 \
    --output-dir ./voxtral_rt_rocm_bf16
```

The preferred W4/BF16 setup nibble-packs TorchAO weight-only INT4 tensors and
runs the ExecuTorch Triton W4A16 kernel while keeping activations in BF16:

```bash
python export_voxtral_rt.py \
    --model-path ~/models/Voxtral-Mini-4B-Realtime-2602 \
    --backend rocm \
    --dtype bf16 \
    --streaming \
    --sliding-window 2048 \
    --output-dir ./voxtral_rt_rocm_w4_bf16 \
    --qlinear-encoder 4w \
    --qlinear 4w \
    --qembedding 8w
```

Do not use CUDA's `tile_packed_to_4d` option on ROCm. That format requires the
CUDA-only `_weight_int4pack_mm` fallback shim, which is intentionally not built
or advertised by the ROCm backend. The exporter rejects that combination
before model loading.

The packed path performs dequantization inside the GPU kernel and does not
materialize a full BF16 weight for each invocation.

The default packed path retains the existing dynamic decoder export and uses
the packed INT4 matmul kernel. CUDA and other non-ROCm exports are unchanged.
Encoder linears also use packed INT4 matmul.

An experimental ROCm-only matvec export is available for performance testing:

```bash
python export_voxtral_rt.py \
    --model-path ~/models/Voxtral-Mini-4B-Realtime-2602 \
    --backend rocm \
    --dtype bf16 \
    --streaming \
    --sliding-window 2048 \
    --rocm-packed-matvec \
    --output-dir ./voxtral_rt_rocm_w4_bf16_matvec \
    --qlinear-encoder 4w \
    --qlinear 4w \
    --qembedding 8w
```

This specializes the decoder to its actual one-token runner input and uses a
BF16-rounded packed matvec. On MI300X it roughly doubled decode throughput for
the 30-second test clip, but greedy output differed from the dynamic matmul
baseline. It is off by default; verify transcript quality and performance on
the target GPU before enabling it. Kernel and export-graph tests cover this
option, but CI does not run a full-model transcript check with it.

#### Metal export examples

Offline:

```bash
python export_voxtral_rt.py \
    --model-path ~/models/Voxtral-Mini-4B-Realtime-2602 \
    --backend metal \
    --output-dir ./voxtral_rt_exports \
    --qlinear-encoder fpa4w \
    --qlinear fpa4w
```

Streaming:

```bash
python export_voxtral_rt.py \
    --model-path ~/models/Voxtral-Mini-4B-Realtime-2602 \
    --backend metal \
    --dtype bf16 \
    --streaming \
    --sliding-window 2048 \
    --output-dir ./voxtral_rt_exports \
    --qlinear-encoder fpa4w \
    --qlinear fpa4w
```

Metal 4-bit quantization (`fpa4w`) requires torchao built with experimental MPS ops:

```bash
# From the ao repo (third-party/ao/)
USE_CPP=1 TORCHAO_BUILD_EXPERIMENTAL_MPS=1 pip install . --no-build-isolation

# Or while installing ExecuTorch from source
EXECUTORCH_BUILD_KERNELS_TORCHAO=1 TORCHAO_BUILD_EXPERIMENTAL_MPS=1 ./install_executorch.sh
```

#### MLX export examples

MLX backend uses the MLX delegate for Apple Silicon GPU acceleration.
NVFP4 quantizes weights using NVIDIA's FP4 data type.

Offline (NVFP4):

```bash
python export_voxtral_rt.py \
    --model-path ~/models/Voxtral-Mini-4B-Realtime-2602 \
    --backend mlx \
    --output-dir ./voxtral_rt_exports \
    --qlinear-encoder nvfp4 \
    --qlinear nvfp4 \
    --qembedding nvfp4
```

Streaming (NVFP4):

```bash
python export_voxtral_rt.py \
    --model-path ~/models/Voxtral-Mini-4B-Realtime-2602 \
    --backend mlx \
    --streaming \
    --output-dir ./voxtral_rt_exports \
    --qlinear-encoder nvfp4 \
    --qlinear nvfp4 \
    --qembedding nvfp4
```

Offline (int4 linear + int8 embedding):

```bash
python export_voxtral_rt.py \
    --model-path ~/models/Voxtral-Mini-4B-Realtime-2602 \
    --backend mlx \
    --output-dir ./voxtral_rt_exports \
    --qlinear-encoder 4w \
    --qlinear 4w \
    --qembedding 8w \
    --qembedding-group-size 128
```

Streaming (int4 linear + int8 embedding):

```bash
python export_voxtral_rt.py \
    --model-path ~/models/Voxtral-Mini-4B-Realtime-2602 \
    --backend mlx \
    --streaming \
    --sliding-window 2048 \
    --output-dir ./voxtral_rt_exports \
    --qlinear-encoder 4w \
    --qlinear 4w \
    --qembedding 8w \
    --qembedding-group-size 128
```

#### CUDA-Windows export examples

Requires `x86_64-w64-mingw32-g++` on `PATH` (mingw-w64 cross-compiler) and
`WINDOWS_CUDA_HOME` pointing to the extracted Windows CUDA package directory.
See [Parakeet README](../parakeet/README.md#cuda-windows-export) for detailed extraction steps.

```bash
export WINDOWS_CUDA_HOME=/opt/cuda-windows/extracted/cuda_cudart/cudart

python export_voxtral_rt.py \
    --model-path ~/models/Voxtral-Mini-4B-Realtime-2602 \
    --backend cuda-windows \
    --dtype bf16 \
    --streaming \
    --sliding-window 2048 \
    --output-dir ./voxtral_rt_exports \
    --qlinear-encoder 4w \
    --qlinear-encoder-packing-format tile_packed_to_4d \
    --qlinear 4w \
    --qlinear-packing-format tile_packed_to_4d \
    --qembedding 8w
```

> [!NOTE]
> Omit `--streaming` from any export command above for offline mode.
> CUDA, CUDA-Windows, and ROCm exports also produce an `aoti_cuda_blob.ptd` file alongside `model.pte`.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path` | (required) | Directory with `params.json` + `consolidated.safetensors` |
| `--backend` | `xnnpack` | `xnnpack`, `mlx`, `metal`, `cuda`, `cuda-windows`, `rocm`, or `portable` |
| `--dtype` | `fp32` | Model dtype: `fp32` or `bf16` |
| `--output-dir` | `./voxtral_rt_exports` | Output directory |
| `--max-seq-len` | `4096` | KV cache length (offline mode only; ignored with `--streaming`) |
| `--delay-tokens` | `6` | Transcription delay in tokens (6 = 480ms) |

| `--qlinear` | (none) | Decoder linear layer quantization (`4w`, `8w`, `8da4w`, `8da8w`, `fpa4w`, `nvfp4`) |
| `--qlinear-group-size` | auto | Group size for decoder linear quantization |
| `--qlinear-packing-format` | (none) | Packing format for decoder 4w quantization (`tile_packed_to_4d` for CUDA) |
| `--qlinear-encoder` | (none) | Encoder linear layer quantization (`4w`, `8w`, `8da4w`, `8da8w`, `fpa4w`, `nvfp4`) |
| `--qlinear-encoder-group-size` | auto | Group size for encoder linear quantization |
| `--qlinear-encoder-packing-format` | (none) | Packing format for encoder 4w quantization (`tile_packed_to_4d` for CUDA) |
| `--qembedding` | (none) | Embedding layer quantization (`4w`, `8w`, `nvfp4`) |
| `--qembedding-group-size` | auto | Group size for embedding quantization |
| `--streaming` | off | Export streaming model with ring buffer KV caches (unlimited duration) |
| `--max-enc-len` | `750` | Encoder sliding window size (streaming only) |
| `--sliding-window` | from `params.json` | Decoder sliding window size (streaming only; ignored in offline mode). Smaller values reduce memory and improve decode speed but limit context |
| `--rocm-packed-matvec` | off | Experimental fixed-shape packed INT4 decoder matvec; requires ROCm and decoder `4w` |

**Notes:**
- `fpa4w` quantization requires `--backend metal`.
- The model was trained with `--delay-tokens 6`. Other values may degrade accuracy.
- The decoder sliding window controls how far back the decoder can attend. At 80ms/step: 2048 = ~2.7 min, 4096 = ~5.5 min, 8192 = ~10.9 min.

## Build

ExecuTorch must be installed from source first (see
[Prerequisites](#prerequisites)). The `make` targets below handle
building core libraries and the runner binary.

```bash
make voxtral_realtime-cpu      # XNNPACK (CPU)
make voxtral_realtime-metal    # Metal (Apple GPU)
make voxtral_realtime-cuda     # CUDA (NVIDIA GPU)
make voxtral_realtime-rocm     # ROCm (AMD GPU, experimental)
```

The CPU, CUDA, Metal, and MLX targets produce the runner at
`cmake-out/examples/models/voxtral_realtime/voxtral_realtime_runner`. ROCm uses
the isolated path below.

### ROCm (experimental)

From the ExecuTorch repository root, the explicit ROCm target builds and
installs the LLM runtime and the model runner into a separate directory:

```bash
make voxtral_realtime-rocm
```

The equivalent workflows are:

```bash
cmake --workflow --preset llm-release-rocm
cd examples/models/voxtral_realtime
cmake --workflow --preset voxtral-realtime-rocm
```

The runner is written to
`cmake-out-rocm-llm/examples/models/voxtral_realtime/voxtral_realtime_runner`.
The separate directory keeps ROCm configuration out of default and CUDA build
caches.

For an end-to-end BF16 or W4/BF16 example:

```bash
examples/models/voxtral_realtime/run_rocm_e2e.sh \
    ~/models/Voxtral-Mini-4B-Realtime-2602 \
    /path/to/input-16khz-mono.wav \
    w4-bf16 \
    both \
    /path/to/output
```

The third argument selects `bf16`, `w4-bf16`, or both precision modes. The
fourth selects `streaming`, `offline`, or both execution modes. Set
`ROCM_PACKED_MATVEC=1` to opt into the experimental fixed-shape decoder matvec.
Set `ROCM_PATH` if ROCm is installed outside `/opt/rocm`.
The script reports model export time, PTE/PTD sizes, and RTF computed as runner
inference time divided by WAV duration.

The AOTInductor-generated shared objects use the C++ runtime from the active
Python environment. Add that runtime before launching the runner directly:

```bash
PYTHON_PREFIX="$(python -c 'import sys; print(sys.prefix)')"
export LD_LIBRARY_PATH="$PYTHON_PREFIX/lib:${LD_LIBRARY_PATH:-}"
```

Without it, systems with an older `/lib64/libstdc++.so.6` can fail while loading
the extracted delegate library with a missing `GLIBCXX` version.

### CUDA-Windows

On Windows (PowerShell), use CMake workflow presets from the executorch root
directory. If you exported with 4-bit quantization, specify your GPU's compute
capability to avoid "invalid device function" errors (the `int4mm` kernels
require SM 80+).

```powershell
cmake --workflow --preset llm-release-cuda
Push-Location examples/models/voxtral_realtime
cmake --workflow --preset voxtral-realtime-cuda
Pop-Location
```

This builds ExecuTorch with CUDA backend support. The runner binary is at
the same path as above. Requires NVIDIA GPU with CUDA toolkit installed.

### Metal (Apple GPU)

```bash
make voxtral_realtime-metal
```

This builds ExecuTorch with Metal backend support. The runner binary is at
the same path as above. Metal exports can only run on macOS with Apple Silicon.

### MLX (Apple GPU)

```bash
make voxtral_realtime-mlx
```

This builds ExecuTorch with MLX backend support. MLX provides GPU acceleration
on Apple Silicon via the MLX delegate.

## Run

The runner requires:
- `model.pte` — exported model (see [Export](#export))
- `tekken.json` — tokenizer from the model weights directory
- `preprocessor.pte` — mel spectrogram preprocessor (see [Preprocessor](#preprocessor))
- A 16kHz mono WAV audio file (or live audio via `--mic`)

### Basic usage

```bash
cmake-out/examples/models/voxtral_realtime/voxtral_realtime_runner \
    --model_path voxtral_rt_exports/model.pte \
    --tokenizer_path ~/models/Voxtral-Mini-4B-Realtime-2602/tekken.json \
    --preprocessor_path voxtral_rt_exports/preprocessor.pte \
    --audio_path input.wav \
    --streaming
```

Omit `--streaming` for offline transcription (requires an offline-exported
model and offline preprocessor).

For CUDA and ROCm, add
`--data_path voxtral_rt_exports/aoti_cuda_blob.ptd`.

**Windows (PowerShell):**

```powershell
.\cmake-out\examples\models\voxtral_realtime\Release\voxtral_realtime_runner.exe `
    --model_path voxtral_rt_exports\model.pte `
    --data_path voxtral_rt_exports\aoti_cuda_blob.ptd `
    --tokenizer_path C:\path\to\tekken.json `
    --preprocessor_path voxtral_rt_exports\preprocessor.pte `
    --audio_path input.wav `
    --streaming
```

### Live microphone input

Use `--mic` to read raw 16kHz float32 PCM from stdin. Requires a
streaming-exported model and streaming preprocessor. Pipe from any audio
capture tool:

```bash
# macOS
ffmpeg -f avfoundation -i ":0" -ar 16000 -ac 1 -f f32le -nostats -loglevel error pipe:1 | \
  cmake-out/examples/models/voxtral_realtime/voxtral_realtime_runner \
    --model_path voxtral_rt_exports/model.pte \
    --tokenizer_path ~/models/Voxtral-Mini-4B-Realtime-2602/tekken.json \
    --preprocessor_path voxtral_rt_exports/preprocessor.pte \
    --mic
```

Ctrl+C stops recording and flushes remaining text.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model_path` | `model.pte` | Path to exported model |
| `--data_path` | (none) | Path to delegate data file (`.ptd`, required for CUDA and ROCm) |
| `--tokenizer_path` | `tekken.json` | Path to Tekken tokenizer |
| `--preprocessor_path` | (none) | Path to mel preprocessor `.pte` |
| `--audio_path` | (none) | Path to 16kHz mono WAV file |
| `--temperature` | `0.0` | Sampling temperature (0 = greedy) |
| `--offline_max_new_tokens` | `500` | Offline-only: maximum extra tokens after audio embeddings are exhausted |
| `--streaming` | off | Use streaming transcription (from WAV file) |
| `--mic` | off | Live microphone mode (reads raw f32le PCM from stdin) |
| `--mic_chunk_ms` | `80` | Mic read chunk size in ms (multiples of 80 recommended) |
| `--color` | (none) | Output text color: `green` or `red` |

## Troubleshooting

- **Audio format**: Input must be 16kHz mono WAV. Convert with
  `ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav`.
- **OOM during export**: Reduce `--max-seq-len` (offline mode) or skip
  encoder quantization (`--qlinear-encoder`).
- **"Model was not exported with --streaming"**: Re-export with the
  `--streaming` flag. Both `--streaming` and `--mic` runner modes
  require a streaming-exported model.
- **`fpa4w` error**: This quantization requires `--backend metal`.
- **Metal runner fails with `Library not loaded: @rpath/libc++.1.dylib`**:
  The AOTInductor-compiled `.so` inside the `.pte` references `libc++` via
  `@rpath`, which can't be resolved when extracted to a temp directory.
  Add `/usr/lib` to `DYLD_LIBRARY_PATH` so dyld finds it in the shared cache:
  ```bash
  DYLD_LIBRARY_PATH=/usr/lib \
      cmake-out/examples/models/voxtral_realtime/voxtral_realtime_runner ...
  ```
- **Metal runner fails with `Library not loaded: libomp.dylib`**:
  The AOTInductor-compiled `.so` links against OpenMP. Install it via
  Homebrew and add it to `DYLD_LIBRARY_PATH`:
  ```bash
  brew install libomp
  DYLD_LIBRARY_PATH=/usr/lib:$(brew --prefix libomp)/lib \
      cmake-out/examples/models/voxtral_realtime/voxtral_realtime_runner ...
  ```
