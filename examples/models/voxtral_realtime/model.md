# Voxtral Realtime — Architecture & Design Notes

Developer reference for `model.py`. For export/usage instructions see
[README.md](README.md).

Source: [mistralai/Voxtral-Mini-4B-Realtime-2602](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602)

Reference implementation: [vLLM voxtral_realtime.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/voxtral_realtime.py)

## Architecture

```
Audio waveform @ 16kHz
  -> Mel spectrogram (128 bins, hop=160, window=400)
(B, 128, T_mel)
  -> CausalConv1d (128 -> 1280, k=3, s=1) + GELU
  -> CausalConv1d (1280 -> 1280, k=3, s=2) + GELU
(B, 1280, T_mel//2) -> transpose -> (B, T_mel//2, 1280)
  -> 32x CausalEncoderLayer (RMSNorm -> RoPE attention -> RMSNorm -> SwiGLU)
  -> RMSNorm
(B, T_mel//2, 1280)
  -> Reshape: concat downsample_factor=4 consecutive frames
(B, T_mel//8, 5120)
  -> AudioLanguageAdapter: Linear(5120, 3072) -> GELU -> Linear(3072, 3072)
(B, T_audio, 3072) = audio_embeds

audio_embeds + token_embedding(prev_token) = combined_embeds

combined_embeds
  -> 26x MistralDecoderLayer (RMSNorm -> GQA attention -> adaptive RMSNorm(t_cond) -> SwiGLU)
  -> RMSNorm -> Linear(3072, 131072)
(B, seq_len, 131072) = logits
```

Audio and text embeddings are **summed** at each position (not concatenated
or masked-scatter like the original non-realtime Voxtral).

### Adaptive RMSNorm

Each decoder layer has a time-conditioned FFN norm unique to this model.
After the standard RMSNorm on the FFN input, a learned scale is applied:

```
scale = 1 + Sequential(Linear(3072->32), GELU, Linear(32->3072))(t_cond)
ffn_input = rms_norm(x) * scale
```

The `t_cond` is a sinusoidal embedding of `n_delay_tokens` (default 6 = 480ms),
precomputed once and passed to each decoder layer as a constant.

### Differences from original Voxtral (non-realtime)

| Aspect | Voxtral (3B) | Voxtral Realtime (4B) |
|--------|-------------|----------------------|
| Encoder | Bidirectional Whisper, LayerNorm, sinusoidal pos | Causal Whisper, RMSNorm, RoPE |
| FFN | Standard FFN | SwiGLU |
| Audio + text | masked_scatter (replace placeholders) | element-wise sum |
| Decoder norm | Standard RMSNorm | Adaptive RMSNorm with t_cond |
| Streaming | No (30s chunks) | Yes (frame-by-frame) |
| Embeddings | Separate lm_head | Tied (output = tok_embeddings) |

## Model Parameters

| Parameter | Encoder | LM Decoder |
|-----------|---------|------------|
| dim | 1280 | 3072 |
| layers | 32 | 26 |
| heads | 32 | 32 (8 KV, GQA 4:1) |
| head_dim | 64 | 128 |
| hidden_dim (FFN) | 5120 | 9216 |
| rope_theta | 1,000,000 | 1,000,000 |
| biases (attn) | wq, wv, wo yes; wk no | none |
| biases (FFN) | w2 yes; w1, w3 no | none |
| vocab_size | — | 131,072 |
| total params | ~1B | ~3.4B |

Audio parameters: 16kHz sample rate, 128 mel bins, hop_length=160,
window_size=400, downsample_factor=4, frame_rate=12.5 fps.

## ExecuTorch Design Choices

The model is written directly with ExecuTorch custom ops rather than using
source transformations. The patterns come from `examples/models/llama/`,
`extension/llm/export/builder.py`, and `optimum-executorch`.

### KV cache: `[B, S, H, D]` layout + `update_cache` custom op

Cache shape is `(1, max_seq_len, n_kv_heads, head_dim)`. Uses
`torch.ops.llama.update_cache(value, cache, start_pos)` which mutates the
cache in-place. This avoids the `index_put_` + `copy_` pattern that triggers
a `requires_grad` bug in `SpecPropPass` during `to_executorch()`.

The `[B, S, H, D]` layout matches what `update_cache` and `custom_sdpa`
expect, so there are no transposes between cache update and attention.

Reference: `examples/models/llama/source_transformation/custom_kv_cache.py`
(`CustomKVCache`).

### SDPA: backend-specific attention implementations

`SDPA` is its own module (not inline code), making it swappable for
backend-specific implementations.

#### XNNPACK/Portable: `SDPA` with `custom_sdpa` fused kernel

Uses `torch.ops.llama.custom_sdpa` — a custom op that fuses attention
computation with causal masking. Two modes (mutually exclusive):

- **Causal** (decoder): `custom_sdpa(q, k, v, start_pos, None, 0, True)`.
  Causal masking via `start_pos` + `is_causal=True`.
- **Explicit mask** (streaming encoder): `custom_sdpa(q, k, v, start_pos, mask, 0, False)`.
  Sliding window mask from `EncoderRingKVCache.create_causal_mask()`.

Both modes handle GQA expansion internally and upcast to float32.

Reference: `examples/models/llama/source_transformation/sdpa.py`
(`SDPACustom`).

#### Metal: `StandardSDPA` with `F.scaled_dot_product_attention`

Metal backend uses AOTInductor (AOTI) which has compatibility issues with
the `custom_sdpa` custom op. For the text decoder, `StandardSDPA` uses the
standard PyTorch `F.scaled_dot_product_attention` with explicit attention masks.
The model's `use_standard_attention` parameter controls which SDPA
implementation is used. The export script sets this based on `--backend`:

```python
use_standard_attention = (args.backend == "metal")
model = load_model(
    args.model_path,
    use_standard_attention=use_standard_attention,
)
```

### Attention layout: `[B, T, H, D]` throughout

Q/K/V projections produce `[B, T, H, D]` via `.view()`. RoPE operates on
`[B, T, H, D]`. Cache stores `[B, S, H, D]`. SDPA receives both in this
layout. No `transpose(1, 2)` pairs in the decoder attention hot path.

This eliminates the need for `RemoveRedundantTransposes` post-export pass
that Llama/optimum-executorch require when using `[B, H, S, D]` attention
with `[B, S, H, D]` cache.

### RoPE: `reshape+unbind` with float32 upcast

```python
q_r, q_i = q.float().reshape(q.shape[:-1] + (-1, 2)).unbind(-1)
```

- `reshape+unbind` instead of stride-2 slicing (`x[..., ::2]`) — avoids
  strided access patterns that produce complex index expressions during export.
- `.float()` upcast before rotation, `.type_as()` downcast after — prevents
  precision loss in fp16/bf16 inference.

Reference: `examples/models/llama/rope.py` (`apply_rotary_emb`).

### RMSNorm: `F.rms_norm` with `self.dim`

Uses `F.rms_norm(x, (self.dim,), self.weight, self.eps)` with a stored
`self.dim` attribute for compatibility with Llama's
`replace_rms_norm_with_native_rms_norm()` source transformation.

### Encoder attention: backend-specific implementations

#### Offline encoder

The offline encoder has no KV cache (processes full mel at once) and
no GQA (n_heads == n_kv_heads). Uses `F.scaled_dot_product_attention` with
`is_causal=True` and standard `[B, H, T, D]` layout. No custom ops needed.
This works on all backends (XNNPACK, Metal, Portable).

The offline encoder uses full causal attention (no sliding window).
The model's `params.json` specifies `sliding_window: 750` but this is
only enforced in the streaming encoder (via KV cache). For audio shorter
than 750 encoder frames (~15s), full causal is equivalent.

#### Streaming encoder

**XNNPACK/Portable only.** The streaming encoder uses `SDPA` (custom_sdpa op)
with explicit sliding window masks from `EncoderRingKVCache`.

**Metal does not yet support streaming mode.** The custom ops used by
`StreamingAudioEncoderExport` (`update_cache_with_indices`, `custom_sdpa`)
are incompatible with AOTI. Future work may add AOTI-compatible streaming
encoder using `index_copy_` and `F.scaled_dot_product_attention`.

## Streaming Encoder (`StreamingAudioEncoderExport`)

For streaming/live transcription, the encoder processes audio incrementally
(8 mel frames = 80ms per step) instead of the full mel at once.

### Architecture

`StreamingAudioEncoderExport` shares all weights with the offline encoder
but uses a different forward path:

```
mel_chunk (1, 128, 8)
  + conv1_state (1, 128, 2) + conv2_state (1, 1280, 2)
  -> cat(state, chunk) -> raw Conv1d (no CausalConv1d padding) -> GELU
  -> cat(state, conv1_out) -> raw Conv1d -> GELU
(1, 1280, 4) -> transpose -> (1, 4, 1280)
  -> 32x streaming encoder layer (EncoderRingKVCache + custom_sdpa)
  -> RMSNorm
(1, 4, 1280)
  -> Reshape downsample (1, 1, 5120) -> Adapter (1, 1, 3072)
-> audio_embeds, new_conv1_state, new_conv2_state
```

### Conv state management

The causal convolutions need left context across chunk boundaries.
Instead of zero-padding (offline) or recompute-with-overlap (vLLM),
explicit conv state carries the tail of the previous chunk:

- **Conv1** (kernel=3, stride=1): state = last 2 mel frames from previous
  chunk. `cat(state, chunk)` → (1, 128, 10) → Conv1d → (1, 1280, 8).
- **Conv2** (kernel=3, stride=2): state = last 2 conv1 GELU output frames.
  `cat(state, conv1_out)` → (1, 1280, 10) → Conv1d → (1, 1280, 4).

The raw `nn.Conv1d` is called directly (bypassing `CausalConv1d.forward`
which would zero-pad). This produces identical results to the offline
encoder — verified to within fp32 precision (max diff < 2e-5).

### Encoder KV cache

Each of the 32 encoder transformer layers gets its own `EncoderRingKVCache`
instance — a ring buffer that overwrites old entries when the window is
exceeded, enabling streaming of arbitrary length audio.

- Cache shape: `(1, 2*max_enc_len, 32, 64)` per layer. The buffer is 2x the
  window size because writes happen *before* attention. With a 1x buffer
  (size = window), writing `seq_len` new entries evicts that many old ones —
  but the current queries still need those old entries. Example with
  `window=4, seq_len=4, start_pos=5`: a 1x buffer would overwrite positions
  1-4 with 5-8, so query at position 5 can only attend to itself instead of
  positions 2-4. A 2x buffer (size 8) keeps positions 1-4 alive alongside
  5-8, giving query 5 full access to its window.
- Default `max_enc_len=750` (matching the model's trained
  sliding window). Configurable via `--max-enc-len`.
- Memory: 32 layers × 2 × 1500 × 32 × 64 × 4 bytes ≈ 786 MB (fp32)
- Duration: unlimited (ring buffer overwrites old entries, RoPE computed on-the-fly)

Cache writes use `torch.ops.llama.update_cache_with_indices` (a custom op
that scatter-writes via an indices tensor). Write indices are computed
analytically: `(arange(seq_len) + start_pos) % buf_size`. No mutable
position state is needed.

Position tracking is analytic — no mutable state buffer. For buffer
slot `j` after `total_written` frames have been stored:

```
abs_pos[j] = j + ((total_written - 1 - j) // buf_size) * buf_size
```

Negative results indicate unwritten slots. The sliding window mask
is computed from these positions each step:

```python
valid = (cache_pos >= 0) & (delta >= 0) & (delta < window_size)
mask = torch.where(valid, 0.0, float("-inf"))
```

The mask is identical for all 32 layers (same `input_pos`), so it
is computed once in `forward()` and reused.

### STFT overlap for streaming mel

The streaming preprocessor (`WhisperAudioProcessor(streaming=True)`)
computes mel without 30-second chunk padding. To match offline mel values
at chunk boundaries, the C++ runner uses overlapping audio windows:

- **Left overlap**: 320 samples (2 × hop_length, ≥ n_fft/2 = 200)
- **Right look-ahead**: 40 samples (2.5ms, matches vLLM's
  `streaming_look_ahead_ms`)
- **Total window**: 320 + 1280 + 40 = 1640 samples → 10 mel frames
- **Frame extraction**: skip first 2 frames (overlap region), take
  frames 2–9 (the 8 that align with offline mel frame positions)

For the first step, the left overlap is zero-padded (matching the
offline encoder's `center=True` STFT edge behavior). The 2.5ms
look-ahead introduces negligible latency.

### Per-component quantization

Quantization is applied per-component after wrapping (following the
Parakeet pattern), allowing different configs for encoder vs decoder:

```bash
# XNNPACK/Portable
--qlinear-encoder 8w      # encoder linear layers
--qlinear 8da4w           # decoder linear layers
--qembedding 8w           # embedding layer

# Metal
--qlinear-encoder fpa4w   # encoder linear layers
--qlinear fpa4w           # decoder linear layers
```

The streaming encoder references the same module objects that
`quantize_model_()` mutates in-place, so quantized weights are
used transparently. Conv1d layers are not quantized (not targeted
by `quantize_model_`). KV caches and SDPA have no trainable weights.

#### Metal-specific quantization

Metal backend uses `fpa4w` (floating-point activation, 4-bit weight)
quantization from TorchAO's experimental MPS ops. This uses
`UIntxWeightOnlyConfig` with HQQ-based quantization parameter selection:

```python
from torchao.experimental.quant_api import UIntxWeightOnlyConfig

config = UIntxWeightOnlyConfig(
    group_size=qlinear_group_size,
    bitwidth=4,
    uintx_choose_qparams_algorithm="hqq",
)
```

### Export: `strict=True` and `.item()` pattern

All exports use `torch.export.export(..., strict=True)`, matching the
Llama builder (`extension/llm/export/builder.py`) and optimum-executorch.

`strict=True` is required because the model uses `.item()` to extract
scalar positions from input tensors (for `update_cache`, `custom_sdpa`,
and ring buffer index computation). With `strict=True`, `.item()` produces
an unbacked `SymInt` — a symbolic integer that remains dynamic at runtime.
With `strict=False`, `.item()` returns the concrete sample value which
gets baked into the graph as a constant, making all cache positions
and attention masks static (the model has no temporal memory).

Each `.item()` call is guarded with `torch._check_is_size(start_pos)`
(non-negative constraint) and optionally `torch._check(start_pos < max)`
(upper bound for bounded caches like the decoder KV cache). The encoder
ring buffer has no upper bound since positions are unlimited.

This is the same pattern used by:
- `examples/models/llama/attention.py` (`KVCache.update`)
- `optimum-executorch` (`custom_sdpa_with_start_pos_forward`)

## Checkpoint Format

Mistral format: `params.json` + `consolidated.safetensors` (bf16, 8.3 GB).

### Memory-efficient loading

`load_model()` uses the Llama pattern to halve peak memory (~17 GB instead
of ~34 GB for the full-size model):

1. **Meta device construction** — `with torch.device("meta"):` builds the
   model with zero-storage parameter tensors (shape/dtype metadata only).
2. **safetensors lazy access** — `safe_open` loads tensors on demand, cast
   to float32 (the default; bf16 is rejected by the XNNPACK partitioner).
3. **`assign=True` state dict loading** — replaces meta tensors by reference
   instead of copying into pre-allocated storage. No duplication.
4. **Post-load fixups** — re-tie `output.weight = tok_embeddings.weight`
   (broken by assign), materialize remaining meta buffers (KV caches as
   zeros), recompute RoPE frequency tables.

Reference: `examples/models/llama/model.py` (`load_model` function).

### Weight mapping

| Checkpoint prefix | Model prefix |
|------------------|-------------|
| `mm_streams_embeddings.embedding_module.whisper_encoder.conv_layers.*` | `encoder.conv_layers.*` |
| `mm_streams_embeddings.embedding_module.whisper_encoder.transformer.layers.*` | `encoder.layers.*` |
| `mm_streams_embeddings.embedding_module.whisper_encoder.transformer.norm.*` | `encoder.norm.*` |
| `mm_streams_embeddings.embedding_module.audio_language_projection.{0,2}.weight` | `adapter.w_{in,out}.weight` |
| `mm_streams_embeddings.embedding_module.tok_embeddings.weight` | `decoder.tok_embeddings.weight` |
| `layers.*` | `decoder.layers.*` |
| `norm.weight` | `decoder.norm.weight` |

Weights are cast to float32 during loading. `decoder.output.weight` is tied
to `decoder.tok_embeddings.weight` (not in checkpoint). KV cache, RoPE
frequency buffers are runtime-initialized.

Tokenizer: Mistral Tekken format (`tekken.json`, 131K vocab).

## Class Hierarchy

```
VoxtralRealtimeModel
  encoder: CausalWhisperEncoder
    conv_layers: [CausalConv1d, CausalConv1d]
    layers: 32x CausalEncoderLayer
      attention_norm: RMSNorm
      attention: EncoderAttention (wq/wk/wv/wo, F.scaled_dot_product_attention)
      ffn_norm: RMSNorm
      feed_forward: EncoderSwiGLU (w1/w2/w3)
    norm: RMSNorm
  adapter: AudioLanguageAdapter (w_in/w_out)
  decoder: MistralDecoder
    tok_embeddings: Embedding
    layers: 26x MistralDecoderLayer
      attention_norm: RMSNorm
      attention: LMAttention
        wq/wk/wv/wo: Linear (no bias)
        kv_cache: KVCache (XNNPACK) or StaticKVCache (Metal)
        sdpa: SDPA (XNNPACK) or StandardSDPA (Metal)
      ffn_norm: RMSNorm
      ada_rms_norm_t_cond: Sequential(Linear, GELU, Linear)
      feed_forward: LMMLP (w1/w2/w3)
    norm: RMSNorm
    output: Linear (tied to tok_embeddings)

StreamingAudioEncoderExport (XNNPACK/Portable only - Metal not yet supported)
  conv1: nn.Conv1d (shared from encoder.conv_layers[0].conv)
  conv2: nn.Conv1d (shared from encoder.conv_layers[1].conv)
  layers: 32x CausalEncoderLayer (shared from encoder.layers)
  enc_norm: RMSNorm (shared from encoder.norm)
  adapter: AudioLanguageAdapter (shared from model.adapter)
  kv_caches: 32x EncoderRingKVCache (ring buffer for sliding window attention)
  sdpa: SDPA (for streaming attention with custom_sdpa op)
  inv_freq: RoPE inverse frequencies (owned, on-the-fly computation)
```

### Backend-specific components

| Component | XNNPACK/Portable | Metal |
|-----------|------------------|-------|
| Decoder KV cache | `KVCache` (update_cache op) | `StaticKVCache` (index_copy_) |
| Decoder SDPA | `SDPA` (custom_sdpa op) | `StandardSDPA` (F.scaled_dot_product_attention) |
| Streaming encoder | ✓ Supported | ✗ Not yet supported |

Metal backend uses standard PyTorch ops (`F.scaled_dot_product_attention`,
`index_copy_`) instead of custom ops for AOTInductor compatibility. Only
offline mode is currently supported on Metal.
