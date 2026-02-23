# Streaming Sortformer Diarizer

Speaker diarization model (117M params) that outputs per-frame speaker activity
probabilities for up to 4 speakers. Based on NVIDIA NeMo's
`SortformerEncLabelModel`. Not ASR — no text output.

Source: `nvidia/diar_streaming_sortformer_4spk-v2` (HuggingFace, CC-BY-4.0).
Tested with both [v2](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2) and [v2.1](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1) (same architecture, v2.1 adds speaker cache scoring parameters).

Paper: [Sortformer](https://arxiv.org/abs/2409.06656),
[Streaming Sortformer](https://arxiv.org/abs/2507.18446)

## Pipeline

```
Raw audio (B, num_samples) @ 16kHz
  → Preprocessor (mel spectrogram: 128 bins, 25ms window, 10ms stride)
(B, 128, T)
  → ConformerEncoder (FastConformer, 17 layers, d_model=512)
    └─ pre_encode: ConvSubsampling 8x → (B, T/8, 512)
    └─ 17× ConformerLayer (rel-pos self-attn + conv + FFN)
(B, T/8, 512)
  → encoder_proj: Linear(512 → 192)
(B, T/8, 192)
  → TransformerEncoder (18 layers, hidden=192, 8 heads)
(B, T/8, 192)
  → Speaker head: Linear(192→192) → ReLU → Linear(192→4) → Sigmoid
(B, T/8, 4)
```

Each output frame = 80ms of audio (10ms stride × 8x subsampling).

## Streaming Architecture

Audio is processed in chunks. Before running the conformer + transformer,
three sequences are concatenated:

```
[speaker_cache | FIFO | current_chunk]  →  encoder  →  transformer  →  head
                                                        ↓
                                              extract chunk predictions
```

- **Speaker cache** (188 frames max): Long-term memory of the most
  speaker-discriminative frames — frames where a single speaker is clearly
  dominant (non-overlapped speech), selected via log-odds importance scoring.
  Compressed when it overflows by keeping the highest-scoring frames per
  speaker plus silence placeholders.

- **FIFO** (configurable): Short-term sliding window of recent embeddings.
  Oldest frames are popped into the speaker cache when the FIFO fills up.

- **Current chunk**: Fresh audio. Only this chunk's `pre_encode`
  (ConvSubsampling) runs; cache and FIFO reuse previously computed embeddings
  via `bypass_pre_encode=True`.

### Streaming Configurations

| Config           | Latency | chunk_len | right_context | fifo_len | update_period | spkcache_len |
|------------------|---------|-----------|---------------|----------|---------------|--------------|
| Very high        | 30.4s   | 340       | 40            | 40       | 300           | 188          |
| High             | 10.0s   | 124       | 1             | 124      | 124           | 188          |
| Low              | 1.04s   | 6         | 7             | 188      | 144           | 188          |
| Ultra low        | 0.32s   | 3         | 1             | 188      | 144           | 188          |

All values in 80ms frames. Latency = (chunk_len + right_context) × 80ms.

## Model Parameters

| Parameter              | Value |
|------------------------|-------|
| sample_rate            | 16000 |
| mel features           | 128   |
| window_size            | 0.025s (25ms) |
| window_stride          | 0.01s (10ms) |
| subsampling_factor     | 8     |
| fc_d_model (conformer) | 512   |
| tf_d_model (transformer)| 192  |
| conformer_layers       | 17    |
| transformer_layers     | 18    |
| attention_heads        | 8     |
| max_num_of_spks        | 4     |
| spkcache_len           | 188   |

## NeMo Classes

| Class | File | Role |
|-------|------|------|
| `SortformerEncLabelModel` | `sortformer_diar_models.py` | Top-level model, forward/diarize |
| `SortformerModules` | `sortformer_modules.py` | Streaming state, encoder_proj, speaker head, cache compression |
| `ConformerEncoder` | `conformer_encoder.py` | FastConformer with pre_encode subsampling |
| `TransformerEncoder` | `transformer_encoders.py` | Standard transformer encoder |
| `AudioToMelSpectrogramPreprocessor` | ASR modules | Mel spectrogram extraction |

## Output

Raw output is `(B, T/8, 4)` sigmoid probabilities. The `diarize()` method
post-processes with VAD-style thresholding and smoothing to produce segments:
`[start_seconds, end_seconds, speaker_index]`.

## Loading

```python
from nemo.collections.asr.models import SortformerEncLabelModel

# From HuggingFace
model = SortformerEncLabelModel.from_pretrained("nvidia/diar_streaming_sortformer_4spk-v2")

# From local .nemo file
model = SortformerEncLabelModel.restore_from("path/to/model.nemo", map_location="cpu", strict=False)

model.eval()
```

## ExecuTorch Export

### Usage

```bash
cd examples/models/sortformer

# Export
python export_sortformer.py --nemo-path /path/to/model.nemo --backend xnnpack

# Validate
python validate_pte.py \
    --nemo-path /path/to/model.nemo \
    --pte-path ./sortformer_exports/sortformer.pte \
    --wav /path/to/audio.wav
```

Output: `sortformer_exports/sortformer.pte` (~470 MB unquantized).

### Exported Methods

Three methods, splitting along streaming boundaries so the C++ runner
manages chunking and cache state:

**`preprocessor`** — portable (no XNNPACK)
- Input: `audio` (N,) float, `length` (1,) int64
- Output: `mel` (1, 128, T_mel) float, `mel_len` (1,) int64
- Dynamic shapes: `audio` dim 0, min=1600, max=1,920,000 (120s)

**`pre_encode`** — XNNPACK
- Input: `chunk` (1, 4000, 128) float, `chunk_len` (1,) int64
- Output: `embs` (1, 500, 512) float, `emb_len` (1,) int64
- Static shapes: conv-derived symbolic expression `1+((L-1)//8)` creates
  an unsolvable guard with `nn.Linear`, so chunk size is fixed at 4000 mel
  frames (~40s). Streaming chunk sizes are fixed per config, so this is not
  a practical limitation.

**`encode`** — XNNPACK
- Input: `embs` (1, T_total, 512) float, `emb_len` (1,) int64
- Output: `preds` (1, T_total, 4) float
- Dynamic shapes: `embs` dim 1, min=2, max=1000

**Metadata** (constant_methods): `sample_rate`, `window_stride`,
`subsampling_factor`, `fc_d_model`, `tf_d_model`, `max_num_of_spks`,
`spkcache_len`.

### Caller Responsibilities

The mel output from `preprocessor` is `(1, 128, T)` (channels-first).
`pre_encode` expects `(1, T, 128)` (channels-last). The caller must
transpose and pad/truncate to 4000 frames between stages. See
`validate_pte.py` for the exact sequence.

## torch.export Fixes

All fixes are in `prepare_for_export()` and the wrapper classes.

### 1. Preprocessor `pad_to=16`
Pads frame count to multiples of 16, creating a data-dependent branch.
Fix: `model.preprocessor.featurizer.pad_to = 0`.

### 2. ConformerEncoder `update_max_seq_length`
Conditionally extends positional encoding buffers. Involves distributed
calls and buffer mutation.
Fix: pre-allocate to max length, then no-op:
```python
model.encoder.set_max_audio_length(5000)
model.encoder.update_max_seq_length = lambda seq_length, device: None
```

### 3. ConvSubsampling MaskedConvSequential
`MaskedConvSequential` uses data-dependent masking that creates export
guards. Masking is unnecessary for single-sample inference.
Fix: `PreEncodeWrapper` runs conv layers directly via `nn.ModuleList`,
bypassing `MaskedConvSequential`.

### 4. `reshape(b, t, -1)` on conv output
`reshape` with `-1` generates an unsolvable guard when the time dim is
symbolic.
Fix: `x.permute(0, 2, 1, 3).flatten(2)` — flattens static dims C and F
without generating guards on the dynamic time dimension.

### 5. Conv-derived expression + Linear
After strided convolutions, the time dim becomes `1+((L-1)//8)`. Applying
`nn.Linear` triggers a guard the solver can't prove. No code-level fix
exists.
Fix: use static shapes for `pre_encode`.

### 6. RelPositionMultiHeadAttention `rel_shift`
The conformer's `rel_shift` uses `pad → view(b,h,-1,T) → slice → view`
to shift relative position scores. The `view` mixes two T-dependent
dimensions, creating a `(2*T*T)//T` guard the solver can't simplify.
Fix: `_rel_shift_export` replaces pad+view with `torch.gather` using
explicitly computed indices. Monkey-patched onto all conformer attention
layers in `prepare_for_export`.
