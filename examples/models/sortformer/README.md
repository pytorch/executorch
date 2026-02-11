# Streaming Sortformer Diarizer for ExecuTorch

Export and run [nvidia/diar_streaming_sortformer_4spk-v2](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2) (117M params) on ExecuTorch for native C++ speaker diarization. Tested with both [v2](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2) and [v2.1](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1).

Speaker diarization answers "who spoke when" — the model outputs per-frame activity probabilities for up to 4 speakers. This is not ASR; there is no text output.

## Quick Start

```bash
# Install Python dependencies
pip install -r install_requirements.txt

# Export to .pte
cd examples/models/sortformer
python export_sortformer.py --nemo-path /path/to/model.nemo --backend xnnpack

# Build the C++ runner (from repo root)
make sortformer-cpu

# Run diarization
./cmake-out/examples/models/sortformer/sortformer_runner \
    --model_path examples/models/sortformer/sortformer_exports/sortformer.pte \
    --audio_path /path/to/audio.wav
```

Output:

```
Diarization results (440 segments, 13117 frames, 1049.4s):
  0.480 1.760 speaker_0
  10.960 11.760 speaker_1
  12.000 14.640 speaker_1
  ...

Speaker 0: 3649/13117 frames active (27.8%)
Speaker 1: 2040/13117 frames active (15.6%)
Speaker 2: 2597/13117 frames active (19.8%)
Speaker 3: 606/13117 frames active (4.6%)
```

Each line is `start_seconds end_seconds speaker_N`. Each output frame covers 80ms of audio (10ms stride x 8x subsampling).

## Export

```bash
python export_sortformer.py --nemo-path /path/to/model.nemo --backend xnnpack
```

| Argument | Description |
|----------|-------------|
| `--nemo-path` | Path to `.nemo` model file |
| `--hf-model` | HuggingFace model ID (default: `nvidia/diar_streaming_sortformer_4spk-v2`) |
| `--backend` | `portable` or `xnnpack` (default: `xnnpack`) |
| `--output-dir` | Output directory (default: `./sortformer_exports`) |

Output: `sortformer_exports/sortformer.pte` (~470 MB unquantized). The preprocessor is always lowered with the portable backend regardless of `--backend`.

## Validate

Compare `.pte` output against NeMo's `diarize()` reference on real audio:

```bash
python validate_pte.py \
    --nemo-path /path/to/model.nemo \
    --pte-path ./sortformer_exports/sortformer.pte \
    --wav /path/to/audio.wav
```

A passing result shows `decision agreement > 95%`.

## C++ Runner

### Build

From the repository root:

```bash
make sortformer-cpu
```

Binary: `cmake-out/examples/models/sortformer/sortformer_runner`

### Arguments

| Argument | Description |
|----------|-------------|
| `--model_path` | Path to `.pte` file (default: `sortformer.pte`) |
| `--audio_path` | Path to input WAV file (16kHz mono, required) |
| `--threshold` | Speaker activity threshold, 0.0–1.0 (default: `0.5`) |
| `--chunk_len` | Encode chunk size in 80ms frames (default: `124`) |
| `--fifo_len` | FIFO buffer size in 80ms frames (default: `124`) |

### How Streaming Works

The runner handles audio of any length through a three-stage pipeline:

1. **Preprocessor**: converts raw audio to mel spectrograms (up to 120s per call, chunked automatically for longer audio).
2. **Pre_encode**: 8x convolutional downsampling from mel frames to 512-dim embeddings (4000 mel frames per call, ~40s each).
3. **Encode**: conformer + transformer that produces per-frame speaker probabilities.

The encode method has a 1000-frame input limit, so long audio must be processed in chunks. Each encode step sees three concatenated buffers:

```
[speaker_cache | FIFO | current_chunk]  →  encode  →  predictions
```

- **Current chunk** (`--chunk_len`): the new embedding frames being processed.
- **FIFO** (`--fifo_len`): a sliding window of recent embeddings providing short-term context. As new chunks arrive, the oldest FIFO frames are pushed to the speaker cache.
- **Speaker cache** (188 frames max): long-term memory. When it overflows, the least speaker-discriminative frames are dropped (ranked by max speaker probability — a simplification of NeMo's log-odds scoring).

Only the current chunk's predictions are kept as output. The cache and FIFO exist purely to give the model context from earlier in the audio.

### Streaming Configurations

The defaults match the model's "High" config. The model was trained and evaluated with these specific configs, so matching one gives best results:

| Config | `--chunk_len` | `--fifo_len` | Context per step |
|--------|---------------|--------------|------------------|
| Very high | 340 | 40 | 188 + 40 + 340 = 568 |
| **High (default)** | **124** | **124** | **188 + 124 + 124 = 436** |
| Low | 6 | 188 | 188 + 188 + 6 = 382 |
| Ultra low | 3 | 188 | 188 + 188 + 3 = 379 |

For offline processing, larger chunks generally give better predictions since the model sees more new audio per step. The "High" config is a good default.

### Limitations

- Input must be 16kHz mono WAV.
- Memory scales linearly with audio length (~1.5 MB per minute) since all embeddings are collected before the streaming encode loop.

## Exported Methods

The `.pte` contains three methods, split along streaming boundaries so the caller manages chunking and cache state:

| Method | Backend | Input | Output |
|--------|---------|-------|--------|
| `preprocessor` | portable | `audio` (N,) float, `length` (1,) int64 | `mel` (1, 128, T) float, `mel_len` (1,) int64 |
| `pre_encode` | XNNPACK | `chunk` (1, 4000, 128) float, `chunk_len` (1,) int64 | `embs` (1, 500, 512) float, `emb_len` (1,) int64 |
| `encode` | XNNPACK | `embs` (1, T, 512) float, `emb_len` (1,) int64 | `preds` (1, T, 4) float |

- `preprocessor`: dynamic audio length (min=1600, max=1,920,000 samples).
- `pre_encode`: static shapes (4000 mel frames).
- `encode`: dynamic sequence length (min=2, max=1000 frames).

**Metadata** via `constant_methods`: `sample_rate`, `window_stride`, `subsampling_factor`, `fc_d_model`, `tf_d_model`, `max_num_of_spks`, `spkcache_len`.

The mel output from `preprocessor` is `(1, 128, T)` channels-first. `pre_encode` expects `(1, T, 128)` channels-last. The caller must transpose and pad to 4000 frames between stages. See `model.md` for architecture details.
