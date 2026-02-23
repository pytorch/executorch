# Silero VAD for ExecuTorch

Export and run [Silero VAD](https://github.com/snakers4/silero-vad) (16kHz, ~2MB, MIT license) on ExecuTorch for native C++ voice activity detection.

Voice activity detection answers "when is someone speaking" — the model outputs a speech probability (0.0–1.0) for each 32ms audio chunk.

## Quick Start

```bash
# Export to .pte
cd examples/models/silero_vad
python export_silero_vad.py --jit-model /path/to/silero_vad.jit

# Build the C++ runner (from repo root)
make silero-vad-cpu

# Run VAD
./cmake-out/examples/models/silero_vad/silero_vad_runner \
    --model_path examples/models/silero_vad/silero_vad_exports/silero_vad.pte \
    --audio_path /path/to/audio.wav
```

Output:

```
  12.512 12.672 speech
  12.736 13.216 speech
  13.248 13.696 speech
  13.728 13.952 speech
  13.984 14.144 speech
  14.176 14.240 speech
  14.272 14.432 speech
  18.144 18.176 speech
  19.168 19.200 speech
  22.624 23.776 speech
  25.216 25.312 speech
  25.344 26.240 speech
  26.304 26.592 speech
  29.376 30.464 speech
  ...

502 segments, 32793 frames, 1049.4s
Speech: 19489/32793 frames (59.4%)
```

Each line is `start_seconds end_seconds speech`. Each output frame covers 32ms of audio (512 samples at 16kHz).

## Getting the Model

The export script requires the `silero_vad.jit` file from the Silero VAD repository:

```bash
git clone https://github.com/snakers4/silero-vad.git
```

The JIT model is at `silero-vad/src/silero_vad/data/silero_vad.jit`.

## Export

```bash
python export_silero_vad.py --jit-model /path/to/silero-vad/src/silero_vad/data/silero_vad.jit
```

| Argument | Description |
|----------|-------------|
| `--jit-model` | Path to `silero_vad.jit` file (required) |
| `--backend` | `portable` or `xnnpack` (default: `xnnpack`) |
| `--output-dir` | Output directory (default: `./silero_vad_exports`) |

Output: `silero_vad_exports/silero_vad.pte` (~2 MB).

## C++ Runner

### Build

From the repository root:

```bash
make silero-vad-cpu
```

Binary: `cmake-out/examples/models/silero_vad/silero_vad_runner`

### Arguments

| Argument | Description |
|----------|-------------|
| `--model_path` | Path to `.pte` file (default: `silero_vad.pte`) |
| `--audio_path` | Path to input WAV file (16kHz mono, required) |
| `--threshold` | Speech probability threshold, 0.0–1.0 (default: `0.5`) |

### How It Works

The model processes audio in 512-sample chunks (32ms at 16kHz). Each chunk is prepended with 64 samples of context from the previous chunk, forming a 576-sample input. The model carries an LSTM hidden state across chunks and outputs a single speech probability per chunk.

The runner applies a simple threshold to produce speech segments: consecutive frames above the threshold are merged into a single segment.

### Limitations

- Input must be 16kHz mono WAV.
- No smoothing or minimum segment duration filtering — raw threshold output.

## Architecture

```
Input: audio (1, 576), LSTM state (2, 1, 128)

Learned STFT:  Conv1d(1, 258, 256, stride=128)  →  magnitude spectrum (1, 129, 4)
CNN encoder:   Conv1d(129→128→64→64→128, kernel=3) →  (1, 128)
LSTM:          LSTMCell(128, 128)                 →  h, c
Decoder:       Conv1d(128, 1, 1) → Sigmoid        →  probability (1, 1)

Output: speech probability (1, 1), new LSTM state (2, 1, 128)
```

## Exported Method

The `.pte` contains a single method:

| Method | Backend | Input | Output |
|--------|---------|-------|--------|
| `forward` | XNNPACK | `x` (1, 576) float, `state` (2, 1, 128) float | `prob` (1, 1) float, `new_state` (2, 1, 128) float |

**Metadata** via `constant_methods`: `sample_rate` (16000), `window_size` (512), `context_size` (64).
