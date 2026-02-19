# Voxtral Realtime

Self-contained ExecuTorch implementation of Mistral's
[Voxtral-Mini-4B-Realtime-2602](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602),
a ~4B parameter streaming speech-to-text model. No HuggingFace Transformers
dependency — weights are loaded directly from the Mistral checkpoint.

## Prerequisites

- ExecuTorch installed from source (see [building from source](../../../docs/source/using-executorch-building-from-source.md))
- [safetensors](https://pypi.org/project/safetensors/) (`pip install safetensors`)
- [torchao](https://github.com/pytorch/ao) (`pip install torchao`)
- Model weights downloaded from [HuggingFace](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602)
  (requires access approval). The directory should contain `params.json`,
  `consolidated.safetensors`, and `tekken.json`.

## Preprocessor

The model takes mel spectrogram input. Export a preprocessor `.pte` to
convert raw 16kHz mono audio into the mel tensor expected by the audio
encoder (128 bins, hop=160, n_fft=400):

```bash
python -m executorch.extension.audio.mel_spectrogram \
    --feature_size 128 \
    --max_audio_len 300 \
    --output_file ./voxtral_rt_exports/preprocessor.pte
```

This produces `preprocessor.pte` which takes a 1-D waveform tensor
`(num_samples,)` and outputs a mel spectrogram `(1, 128, T_mel)`. The
`--max_audio_len 300` flag supports audio up to 5 minutes.

For streaming, add `--streaming` to skip the 30-second chunk padding so
that 1280 samples (80ms) produces exactly 8 mel frames:

```bash
python -m executorch.extension.audio.mel_spectrogram \
    --feature_size 128 \
    --streaming \
    --output_file ./voxtral_rt_exports/preprocessor.pte
```

## Export

Export produces a single `.pte` file with three methods:

| Method | Input | Output |
|--------|-------|--------|
| `audio_encoder` | mel spectrogram `(1, 128, T_mel)` | audio embeddings `(1, T_mel//8, 3072)` |
| `text_decoder` | embeddings `(1, seq_len, 3072)` + positions `(seq_len,)` | logits `(1, seq_len, 131072)` |
| `token_embedding` | token IDs `(1, seq_len)` | embeddings `(1, seq_len, 3072)` |

```bash
python export_voxtral_rt.py \
    --model-path ~/models/Voxtral-Mini-4B-Realtime-2602 \
    --backend xnnpack \
    --output-dir ./voxtral_rt_exports \
    --qlinear-encoder 8da4w \
    --qlinear 8da4w \
    --qembedding 8w
```

For streaming, add `--streaming` to export the encoder with KV caches for
incremental processing. This replaces `audio_encoder` with
`encode_audio_chunk` which processes 8 mel frames at a time:

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

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path` | (required) | Directory with `params.json` + `consolidated.safetensors` |
| `--backend` | `xnnpack` | `xnnpack` or `portable` |
| `--output-dir` | `./voxtral_rt_exports` | Output directory |
| `--max-seq-len` | `4096` | KV cache length |
| `--delay-tokens` | `6` | Transcription delay in tokens (6 = 480ms) |
| `--qlinear` | (none) | Decoder linear layer quantization (`4w`, `8w`, `8da4w`, `8da8w`) |
| `--qlinear-group-size` | `32` | Group size for decoder linear quantization |
| `--qlinear-encoder` | (none) | Encoder linear layer quantization (`4w`, `8w`, `8da4w`, `8da8w`) |
| `--qlinear-encoder-group-size` | `32` | Group size for encoder linear quantization |
| `--qembedding` | (none) | Embedding layer quantization (`8w`) |
| `--streaming` | off | Export streaming encoder with KV cache |
| `--max-enc-len` | `750` | Encoder sliding window size (streaming only) |

## Build

```bash
make voxtral_realtime-cpu
```

This builds ExecuTorch core libraries with XNNPACK, then the runner binary
at `cmake-out/examples/models/voxtral_realtime/voxtral_realtime_runner`.

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

For streaming, add `--streaming`. The runner processes audio in 80ms steps
(one audio token per step), computing mel and running the encoder+decoder
incrementally.

```bash
cmake-out/examples/models/voxtral_realtime/voxtral_realtime_runner \
    --model_path voxtral_rt_exports/model.pte \
    --tokenizer_path ~/models/Voxtral-Mini-4B-Realtime-2602/tekken.json \
    --preprocessor_path voxtral_rt_exports/preprocessor.pte \
    --audio_path input.wav \
    --streaming
```

For live microphone input, use `--mic` to read raw 16kHz float32 PCM from
stdin. Pipe from any audio capture tool:

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

### Example output

```
$ cmake-out/examples/models/voxtral_realtime/voxtral_realtime_runner \
    --model_path voxtral_rt_exports/model.pte \
    --tokenizer_path tekken.json \
    --preprocessor_path voxtral_rt_exports/preprocessor.pte \
    --audio_path output.wav

Mr. Quilter is the apostle of the middle classes, and we are glad to
welcome his gospel. Nor is Mr. Quilter's manner less interesting than
his matter. He tells us that at this festive season of the year, with
Christmas and roast beef looming before us, similes drawn from eating
and its results occur most readily to the mind.
```

## Architecture

See [model.md](model.md) for architecture details, design choices, and
checkpoint format.
