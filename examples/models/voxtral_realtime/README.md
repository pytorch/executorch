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

Two modes are supported: **offline** (encode full audio, then decode)
and **streaming** (process 80ms chunks in real time, including live
microphone input).

## Prerequisites

- ExecuTorch installed from source (see [building from source](../../../docs/source/using-executorch-building-from-source.md))
- [safetensors](https://pypi.org/project/safetensors/) (`pip install safetensors`)
- Model weights downloaded from [HuggingFace](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602).
  The directory should contain `params.json`, `consolidated.safetensors`,
  and `tekken.json`.

## Preprocessor

Export a preprocessor `.pte` to convert raw audio into the format the
model expects. `--max_audio_len 300` supports audio up to 5 minutes
(300 seconds):

```bash
python -m executorch.extension.audio.mel_spectrogram \
    --feature_size 128 \
    --max_audio_len 300 \
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

## Export

Export produces a single `.pte` containing the audio encoder, text decoder,
and token embedding.

```bash
python export_voxtral_rt.py \
    --model-path ~/models/Voxtral-Mini-4B-Realtime-2602 \
    --backend xnnpack \
    --output-dir ./voxtral_rt_exports \
    --qlinear-encoder 8da4w \
    --qlinear 8da4w \
    --qembedding 8w
```

For streaming, add `--streaming` to export the encoder for incremental
processing (80ms audio chunks):

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

### Backend support

| Backend | Offline | Streaming | Quantization |
|---------|---------|-----------|--------------|
| `xnnpack` | ✓ | ✓ | `4w`, `8w`, `8da4w`, `8da8w` |
| `metal` | ✓ | ✗ | none (fp32) or `fpa4w` (Metal-specific 4-bit) |

Metal backend provides Apple GPU acceleration. It does not yet support
streaming mode.

#### Metal export example

```bash
python export_voxtral_rt.py \
    --model-path ~/models/Voxtral-Mini-4B-Realtime-2602 \
    --backend metal \
    --output-dir ./voxtral_rt_exports \
    --qlinear-encoder fpa4w \
    --qlinear fpa4w
```

**Note:** Metal 4-bit quantization requires torchao built with experimental MPS (Metal) ops.

You can install torchao with Metal support from the `ao` repo (in executorch/third-party/ao/)
```bash
USE_CPP=1 TORCHAO_BUILD_EXPERIMENTAL_MPS=1 pip install . --no-build-isolation
```

Alternatively, you can build torchao with Metal support while installing ExecuTorch from source
```bash
EXECUTORCH_BUILD_KERNELS_TORCHAO=1 TORCHAO_BUILD_EXPERIMENTAL_MPS=1 ./install_executorch.sh
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path` | (required) | Directory with `params.json` + `consolidated.safetensors` |
| `--backend` | `xnnpack` | `xnnpack`, `metal`, or `portable` |
| `--output-dir` | `./voxtral_rt_exports` | Output directory |
| `--max-seq-len` | `4096` | KV cache length |
| `--delay-tokens` | `6` | Transcription delay in tokens (6 = 480ms) |
| `--qlinear` | (none) | Decoder linear layer quantization (`4w`, `8w`, `8da4w`, `8da8w`, `fpa4w`) |
| `--qlinear-group-size` | `32` | Group size for decoder linear quantization |
| `--qlinear-encoder` | (none) | Encoder linear layer quantization (`4w`, `8w`, `8da4w`, `8da8w`, `fpa4w`) |
| `--qlinear-encoder-group-size` | `32` | Group size for encoder linear quantization |
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

### XNNPACK (CPU)

```bash
make voxtral_realtime-cpu
```

This builds ExecuTorch core libraries with XNNPACK, then the runner binary
at `cmake-out/examples/models/voxtral_realtime/voxtral_realtime_runner`.

### Metal (Apple GPU)

```bash
make voxtral_realtime-metal
```

This builds ExecuTorch with Metal backend support. The runner binary is at
the same path as above. Metal exports can only run on macOS with Apple Silicon.

## Run

The runner requires:
- `model.pte` — exported model (see [Export](#export))
- `tekken.json` — tokenizer from the model weights directory
- `preprocessor.pte` — mel spectrogram preprocessor (see [Preprocessor](#preprocessor))
- A 16kHz mono WAV audio file (or live audio via `--mic`)

```bash
cmake-out/examples/models/voxtral_realtime/voxtral_realtime_runner \
    --model_path voxtral_rt_exports/model.pte \
    --tokenizer_path ~/models/Voxtral-Mini-4B-Realtime-2602/tekken.json \
    --preprocessor_path voxtral_rt_exports/preprocessor.pte \
    --audio_path input.wav
```

For streaming, add `--streaming`. This requires a model exported with
`--streaming`. The runner processes audio in 80ms steps, computing mel
and running the encoder+decoder incrementally.

```bash
cmake-out/examples/models/voxtral_realtime/voxtral_realtime_runner \
    --model_path voxtral_rt_exports/model.pte \
    --tokenizer_path ~/models/Voxtral-Mini-4B-Realtime-2602/tekken.json \
    --preprocessor_path voxtral_rt_exports/preprocessor.pte \
    --audio_path input.wav \
    --streaming
```

For live microphone input, use `--mic` to read raw 16kHz float32 PCM from
stdin. This requires a model exported with `--streaming` and a streaming
preprocessor. Pipe from any audio capture tool:

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

| Flag | Default | Description |
|------|---------|-------------|
| `--model_path` | `model.pte` | Path to exported model |
| `--tokenizer_path` | `tekken.json` | Path to Tekken tokenizer |
| `--preprocessor_path` | (none) | Path to mel preprocessor `.pte` |
| `--audio_path` | (none) | Path to 16kHz mono WAV file |
| `--temperature` | `0.0` | Sampling temperature (0 = greedy) |
| `--max_new_tokens` | `500` | Maximum tokens to generate |
| `--streaming` | off | Use streaming transcription (from WAV file) |
| `--mic` | off | Live microphone mode (reads raw f32le PCM from stdin) |

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
