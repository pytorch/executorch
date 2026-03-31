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
including live microphone input) and **offline** (encode full audio, then
decode). The examples below use streaming mode. Omit `--streaming` from
export and run commands for offline mode.

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
    --output-dir ./voxtral_rt_exports \
    --qlinear-encoder 8da4w \
    --qlinear 8da4w \
    --qembedding 8w
```

<details>
<summary><strong>Metal</strong></summary>

```bash
python export_voxtral_rt.py \
    --model-path ~/models/Voxtral-Mini-4B-Realtime-2602 \
    --backend metal \
    --dtype bf16 \
    --streaming \
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

</details>

<details>
<summary><strong>CUDA</strong></summary>

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

</details>

<details>
<summary><strong>CUDA-Windows</strong></summary>

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
    --output-dir ./voxtral_rt_exports \
    --qlinear-encoder 4w \
    --qlinear-encoder-packing-format tile_packed_to_4d \
    --qlinear 4w \
    --qlinear-packing-format tile_packed_to_4d \
    --qembedding 8w
```

</details>

> [!NOTE]
> Omit `--streaming` from any export command above for offline mode.
> CUDA and CUDA-Windows exports also produce an `aoti_cuda_blob.ptd` file alongside `model.pte`.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path` | (required) | Directory with `params.json` + `consolidated.safetensors` |
| `--backend` | `xnnpack` | `xnnpack`, `metal`, `cuda`, `cuda-windows`, or `portable` |
| `--dtype` | `fp32` | Model dtype: `fp32` or `bf16` |
| `--output-dir` | `./voxtral_rt_exports` | Output directory |
| `--max-seq-len` | `4096` | KV cache length |
| `--delay-tokens` | `6` | Transcription delay in tokens (6 = 480ms) |
| `--qlinear` | (none) | Decoder linear layer quantization (`4w`, `8w`, `8da4w`, `8da8w`, `fpa4w`) |
| `--qlinear-group-size` | `32` | Group size for decoder linear quantization |
| `--qlinear-packing-format` | (none) | Packing format for decoder 4w quantization (`tile_packed_to_4d` for CUDA) |
| `--qlinear-encoder` | (none) | Encoder linear layer quantization (`4w`, `8w`, `8da4w`, `8da8w`, `fpa4w`) |
| `--qlinear-encoder-group-size` | `32` | Group size for encoder linear quantization |
| `--qlinear-encoder-packing-format` | (none) | Packing format for encoder 4w quantization (`tile_packed_to_4d` for CUDA) |
| `--qembedding` | (none) | Embedding layer quantization (`8w`) |
| `--streaming` | off | Export streaming encoder with KV cache |
| `--max-enc-len` | `750` | Encoder sliding window size (streaming only) |

**Notes:**
- `fpa4w` quantization requires `--backend metal`.
- The model was trained with `--delay-tokens 6`. Other values may degrade accuracy.

## Build

ExecuTorch must be installed from source first (see
[Prerequisites](#prerequisites)). The `make` targets below handle
building core libraries and the runner binary.

```bash
make voxtral_realtime-cpu      # XNNPACK (CPU)
make voxtral_realtime-metal    # Metal (Apple GPU)
make voxtral_realtime-cuda     # CUDA (NVIDIA GPU)
```

All targets produce the runner binary at
`cmake-out/examples/models/voxtral_realtime/voxtral_realtime_runner`.

### CUDA-Windows

On Windows (PowerShell), use CMake workflow presets from the executorch root
directory. If you exported with 4-bit quantization, specify your GPU's compute
capability to avoid "invalid device function" errors (the `int4mm` kernels
require SM 80+).

```powershell
$env:CMAKE_CUDA_ARCHITECTURES="80;86;89;90;120"
cmake --workflow --preset llm-release-cuda
Push-Location examples/models/voxtral_realtime
cmake --workflow --preset voxtral-realtime-cuda
Pop-Location
```

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

For CUDA backends (Linux and Windows), add `--data_path voxtral_rt_exports/aoti_cuda_blob.ptd`.

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
| `--data_path` | (none) | Path to delegate data file (`.ptd`, required for CUDA) |
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
- **OOM during export**: Reduce `--max-seq-len` or skip encoder
  quantization (`--qlinear-encoder`).
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
