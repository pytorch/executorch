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
source transformations. The patterns come from `examples/models/llama/` and
`optimum-executorch`.

### KV cache: `[B, S, H, D]` layout + `update_cache` custom op

Cache shape is `(1, max_seq_len, n_kv_heads, head_dim)`. Uses
`torch.ops.llama.update_cache(value, cache, start_pos)` which mutates the
cache in-place. This avoids the `index_put_` + `copy_` pattern that triggers
a `requires_grad` bug in `SpecPropPass` during `to_executorch()`.

The `[B, S, H, D]` layout matches what `update_cache` and `custom_sdpa`
expect, so there are no transposes between cache update and attention.

Reference: `examples/models/llama/source_transformation/custom_kv_cache.py`
(`CustomKVCache`).

### SDPA: separate `nn.Module` + `custom_sdpa` fused kernel

`SDPA` is its own module (not inline code), making it swappable for
alternative implementations (quantized, CoreML, manual matmul).

Uses `torch.ops.llama.custom_sdpa(q, k, v, start_pos, None, 0, True)`
which handles:
- **GQA expansion** internally (no `repeat_interleave` needed)
- **Causal masking** via `start_pos` + `is_causal=True` (no pre-built
  mask buffer)
- **float32 upcast** for numerical stability

Reference: `examples/models/llama/source_transformation/sdpa.py`
(`SDPACustom`).

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

### Encoder attention: standard `F.scaled_dot_product_attention`

The encoder has no KV cache (processes full mel at once in offline mode) and
no GQA (n_heads == n_kv_heads). Uses `F.scaled_dot_product_attention` with
`is_causal=True` and standard `[B, H, T, D]` layout. No custom ops needed.

Uses full causal attention (no sliding window of 750) — acceptable for
offline mode and simpler for export. Sliding window would be added for
streaming (Phase 3).

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
        kv_cache: KVCache (update_cache custom op)
        sdpa: SDPA (custom_sdpa custom op)
      ffn_norm: RMSNorm
      ada_rms_norm_t_cond: Sequential(Linear, GELU, Linear)
      feed_forward: LMMLP (w1/w2/w3)
    norm: RMSNorm
    output: Linear (tied to tok_embeddings)
```
