# Voxtral TTS — Architecture & Design Notes

Developer reference for `model.py` and `codec.py`. For export/usage instructions
see [README.md](README.md).

The model is written directly with ExecuTorch custom ops rather than using
source transformations. Patterns are reused from `examples/models/voxtral_realtime/`
and `examples/models/llama/`.

## Architecture

```
Text tokens
  -> tok_embeddings: Embedding(vocab_size, D)
(1, seq, D) = text_embeds

text_embeds
  -> 26x MistralDecoderLayer (RMSNorm -> GQA attention -> RMSNorm -> SwiGLU)
  -> RMSNorm
(1, seq, D) = hidden_states

hidden_states[:, -1, :]
  -> FlowMatchingAudioTransformer (every step, no lm_head):
     -> semantic_codebook_output(hidden) -> argmax -> semantic_code
     -> 7-step Euler ODE with CFG:
        each step: [input_proj(x_t), time_proj(t_emb), llm_proj(hidden)]
          -> 3x AcousticTransformerBlock (RMSNorm -> bidirectional GQA -> RMSNorm -> SwiGLU)
          -> RMSNorm -> acoustic_codebook_output -> velocity
     -> quantize to discrete codes
(1, 1 + n_acoustic_codebook) = audio_codes

audio_codes
  -> AudioTokenEmbedding: multi-codebook lookup + sum
(1, 1, D) = audio_embeds   (fed back to decoder for next step)

After generation complete:
  all audio_codes -> AudioTokenizerDecoder (codec.pte):
    -> quantizer.decode -> (B, D_latent, T)
    -> [CausalConv1d + Transformer(ALiBi, sliding window) + CausalConvTranspose1d] × N
    -> output_proj -> reshape
(1, 1, T * downsample_factor) = waveform
```

The model exports five methods in `model.pte` and one in `codec.pte`:

| File | Method | Input | Output |
|------|--------|-------|--------|
| model.pte | `token_embedding` | token IDs `(1, seq)` | embeddings `(1, seq, D)` |
| model.pte | `text_decoder` | embeddings `(1, seq, D)` + positions `(seq,)` | hidden states `(1, seq, D)` |
| model.pte | `lm_head` | hidden `(1, 1, D)` | logits `(1, 1, vocab_size)` |
| model.pte | `decode_audio_frame` | hidden `(1, D)` + noise `(1, C)` | audio codes `(1, 1+C)` |
| model.pte | `audio_token_embedding` | codes `(1, K, seq)` | embeddings `(1, seq, D)` |
| codec.pte | `audio_decoder` | codes `(1, K, T)` | waveform `(1, 1, T*upsample)` |

## Model Parameters

| Parameter | LM Decoder | Acoustic Transformer |
|-----------|------------|---------------------|
| dim | D (from config) | D (same as LM) |
| layers | 26 | 3 |
| heads | 32 (8 KV, GQA 4:1) | 6 (2 KV, GQA 3:1) |
| head_dim | 128 | 128 |
| hidden_dim (FFN) | 9216 | from config (default 2048) |
| rope_theta | 1,000,000 | — (no RoPE) |
| biases (attn) | none | none (default) |
| biases (FFN) | none | none (default) |
| vocab_size | 131,072 | — |
| positional encoding | RoPE (split-half) | none (seq_len=3, fixed) |
| attention type | causal (with KV cache) | bidirectional (no cache) |

| Audio Parameter | Value |
|-----------------|-------|
| semantic_codebook_size | 8,192 |
| acoustic_codebook_size | 21 |
| n_acoustic_codebook | 36 |
| n_audio_special_tokens | 2 ([EMPTY_AUDIO]=0, [END_AUDIO]=1) |
| acoustic_decode_iters | 8 (7 Euler steps) |
| cfg_alpha | 1.2 |

| Codec Parameter | Value |
|-----------------|-------|
| sampling_rate | 24,000 Hz |
| patch_size | 240 |
| latent_dim | 292 (256 semantic + 36 acoustic) |
| dim | 1,024 |
| decoder stages | 4 (transformer lengths: 2,2,2,2) |
| decoder convs | kernels: 3,4,4,4; strides: 1,2,2,2 |
| attention | ALiBi + causal sliding window |
| downsample_factor | 1,920 (240 × 8) |

## Memory Footprint

Decoder KV cache: 26 layers × 2 (K, V) × 4096 × 8 × 128 × bytes_per_elem.
fp32: ≈ 832 MB, bf16: ≈ 416 MB.

Runtime memory = model weights + KV cache + working memory. Weight
sizes depend on quantization: ~16 GB (fp32), ~8 GB (bf16), ~4 GB (8w),
~2 GB (4w/fpa4w). Metal backend should use bf16 (`--dtype bf16`) when
quantization is enabled.

## Class Hierarchy

```
VoxtralTTSModel
  decoder: MistralDecoder
    tok_embeddings: Embedding
    layers: 26x MistralDecoderLayer
      attention_norm: RMSNorm
      attention: LMAttention
        wq/wk/wv/wo: Linear (no bias)
        kv_cache: KVCache (XNNPACK) or StaticKVCache (Metal)
        sdpa: SDPA (XNNPACK) or MetalSDPA (Metal)
      ffn_norm: RMSNorm
      feed_forward: LMMLP (w1/w2/w3)
    norm: RMSNorm
    output: Linear (tied to tok_embeddings)
  acoustic_transformer: FlowMatchingAudioTransformer
    time_embedding: TimeEmbedding (sinusoidal, inv_freq buffer)
    input_projection: Linear(n_acoustic_codebook, at_dim)
    time_projection: Linear(at_dim, at_dim)
    llm_projection: Linear(at_input_dim, at_dim)
    layers: 3x AcousticTransformerBlock
      attention_norm: RMSNorm
      attention: BidirectionalAttention (wq/wk/wv/wo, manual attention)
      ffn_norm: RMSNorm
      feed_forward: LMMLP (w1/w2/w3)
    norm: RMSNorm
    semantic_codebook_output: Linear(at_input_dim, padded_semantic_vocab)
    acoustic_codebook_output: Linear(at_dim, n_acoustic_codebook)
  audio_token_embedding: AudioTokenEmbedding
    embeddings: Embedding (shared multi-codebook with offsets)

AudioTokenizerDecoder (codec.pte, separate)
  quantizer: AudioCodebook
    semantic: SemanticCodebook (Euclidean, embedding_sum / cluster_usage)
    acoustic: AcousticCodebook (FSQ, n_levels=21)
  decoder_blocks: ModuleList
    CodecCausalConv1d (latent_dim -> dim, first projection)
    CodecTransformerBlock × 2 (window=2)
    CodecCausalConvTranspose1d (dim, dim, k=4, s=2)
    CodecTransformerBlock × 2 (window=4)
    CodecCausalConvTranspose1d (dim, dim, k=4, s=2)
    CodecTransformerBlock × 2 (window=8)
    CodecCausalConvTranspose1d (dim, dim, k=4, s=2)
    CodecTransformerBlock × 2 (window=16)
  output_proj: CodecCausalConv1d (dim -> patch_size)
```

## Text Decoder

The text decoder (`MistralDecoder`) is a standard Mistral decoder with
GQA. Unlike voxtral_realtime, there is no adaptive RMSNorm (`ada_rms_norm_t_cond`)
since TTS has no transcription delay conditioning.

`forward()` returns **normed hidden states** (after `self.norm(x)`) rather
than logits. This is because hidden states are consumed by both the LM head
(for text token sampling) and the acoustic transformer (for audio code
generation). The LM head is exported as a separate method.

### RoPE convention

The LM decoder uses **split-half** (rotate_half) RoPE, matching the
HuggingFace/Mistral convention. This is different from the Llama
**interleaved** (reshape+unbind pairs) convention used by
`examples/models/llama/`. The two conventions are NOT interchangeable —
using the wrong one produces completely different hidden states.

Split-half: splits head_dim into first half and second half, swaps and
negates. Interleaved: groups consecutive pairs `(x[0],x[1]), (x[2],x[3])`.

### KV cache

Same as voxtral_realtime:

**XNNPACK/Portable:** `KVCache` with `[B, S, H, D]` layout, `torch.ops.llama.update_cache`.

**Metal:** `StaticKVCache` with `[B, H, S, D]` layout, `index_copy_`.

### SDPA

Same as voxtral_realtime:

**XNNPACK/Portable:** `SDPA` — `torch.ops.llama.custom_sdpa`.

**Metal:** `MetalSDPA` — `_scaled_dot_product_attention_math_for_mps`. The
MPS SDPA meta kernel requires `seq_len > 2`, so the Metal export uses
`Dim("seq_len", min=3, ...)`. Single-token decode must pad to length 3.


### Attention mask

The Metal attention mask uses an additive float mask matching Q/K/V dtype.
The branch-free implementation avoids `if seqlen > 1:` to prevent unprovable
shape guards during export:

```python
diff = input_pos.unsqueeze(1) - k_pos.unsqueeze(0) + 1
valid = torch.clamp(diff, min=0, max=1)
mask = (valid.to(dtype) - 1.0) * 1e9
```

## Acoustic Transformer

`FlowMatchingAudioTransformer` generates audio codes from LLM hidden states
via a flow-matching ODE with classifier-free guidance.

### Flow matching decode

`decode_one_frame(hidden_states, noise)`:

1. **Semantic code**: `semantic_codebook_output(hidden_states)` → argmax.
   The semantic head operates on the raw LLM hidden states (`at_input_dim`),
   not the transformer's internal dim. In real models, `at_dim == at_input_dim`.

2. **Acoustic codes via Euler ODE** (7 steps, unrolled by `torch.export`):
   ```
   x = noise * noise_scale
   for each timestep pair (t, t+dt):
     t_emb = time_embedding(t)
     # Batch conditional + unconditional for CFG
     v_all = predict_velocity([x, x], [hidden, zeros], [t_emb, t_emb])
     v = cfg_alpha * v_cond + (1 - cfg_alpha) * v_uncond
     x = x + v * dt
   ```

3. **Quantize**: `clamp(-1, 1)` → scale to `[0, levels-1]` → round → offset
   by `_N_AUDIO_SPECIAL_TOKENS`.

### Noise as input

The runner generates `torch.randn(1, n_acoustic_codebook)` and passes it to
`decode_audio_frame`. This avoids baking random state into the exported
graph (same pattern as the vLLM CUDA graph wrapper which uses pre-allocated
noise buffers).

### Bidirectional attention

`BidirectionalAttention` is non-causal with no positional encoding. The
sequence length is always 3 tokens: `[input_projection(x_t), time_emb, llm_hidden]`.
Uses manual attention (`q @ k.T * scale → softmax → @ v`) since the fixed
seq_len=3 makes SDPA kernel overhead unnecessary. GQA expansion via
`repeat_interleave` (3:1 ratio with default 6 heads / 2 KV heads).

### `_predict_velocity`

```
x_t: (B, n_acoustic_codebook) → input_projection → (B, 1, at_dim)
t_emb: (B, at_dim) → time_projection → (B, 1, at_dim)
llm_output: (B, at_input_dim) → llm_projection → (B, 1, at_dim)
  cat → (B, 3, at_dim) → 3x AcousticTransformerBlock → norm
  → acoustic_codebook_output(token_0) → (B, n_acoustic_codebook)
```

## Audio Token Embedding

`AudioTokenEmbedding` maps multi-codebook audio codes back to the LLM's
input space. Each codebook has its own offset into a shared embedding
table (pre-computed as a registered buffer). Forward:

```python
offset_codes = codes + offsets[None, :, None]   # per-codebook offset
emb = embeddings(offset_codes)                  # (B, K, seq, D)
return emb.sum(dim=1)                           # (B, seq, D)
```

## Audio Codec Decoder

The codec decoder (`codec.py`) converts discrete audio codes to a waveform.
It reimplements the vLLM `VoxtralTTSAudioTokenizer` decoder path with
export-safe modifications.

### Quantizer decode

`AudioCodebook.decode(codes)` splits codes into semantic (codebook 0) and
acoustic (codebooks 1..K) parts:
- **Semantic**: Euclidean codebook lookup (`embedding_sum / cluster_usage`)
- **Acoustic**: FSQ inverse (`(codes * 2 / (levels - 1)) - 1`)

Concatenated to produce `(B, D_latent, T)` continuous embeddings.

### Decoder network

The decoder alternates between transformer blocks and transposed convolutions:

```
CausalConv1d (latent_dim → dim, k=3, s=1)
  → 2x TransformerBlock (ALiBi, window=2)
  → CausalConvTranspose1d (dim, dim, k=4, s=2)    ← 2x upsample
  → 2x TransformerBlock (ALiBi, window=4)
  → CausalConvTranspose1d (dim, dim, k=4, s=2)    ← 2x upsample
  → 2x TransformerBlock (ALiBi, window=8)
  → CausalConvTranspose1d (dim, dim, k=4, s=2)    ← 2x upsample
  → 2x TransformerBlock (ALiBi, window=16)
CausalConv1d output_proj (dim → patch_size, k=7)
  → reshape (B, patch_size, T) → (B, 1, T * patch_size)
```

The sliding window size doubles at each upsample stage (2 → 4 → 8 → 16),
starting from the window size after the encoder's downsampling stages.

### Export-safe modifications

| vLLM pattern | ExecuTorch replacement |
|-------------|----------------------|
| `einops.rearrange` | Explicit `permute`/`reshape` |
| `flash_attn_func` | `F.scaled_dot_product_attention` with ALiBi mask |
| `weight_norm` parametrizations | Folded via `remove_parametrizations` before export |
| Dynamic padding (`math.ceil`) | Static input shapes (chunked at `--codec-chunk-size`) |
| `nn.ModuleDict` (string keys) | Works with `nn.ModuleList` (same state dict keys) |

### Codec attention

`CodecAttention` uses ALiBi (Attention with Linear Biases) instead of RoPE,
with a causal sliding window. The attention bias is constructed per-forward:

```python
rel_pos = positions[None, :] - positions[:, None]    # (T, T)
attn_bias = alibi_slopes[:, None, None] * rel_pos    # (H, T, T)
attn_bias.masked_fill_(rel_pos > 0, -inf)            # causal
attn_bias.masked_fill_(|rel_pos| > window, -inf)     # sliding window
```

ALiBi slopes are pre-computed as a registered buffer using the geometric
sequence `2^(-8/n)` for power-of-2 head counts.

### Layer scale

`CodecTransformerBlock` uses learnable layer scale parameters
(`attention_scale`, `ffn_scale`) initialized based on layer depth:
- Layers 0–17: `0.1`
- Layers 18–24: `1e-5`
- Layers 25+: `1e-6`

## Quantization

Quantization is applied per-component after wrapping:

```bash
# Metal (recommended)
--qlinear fpa4w           # LM decoder + acoustic transformer linears
--qembedding 8w           # text embedding

# XNNPACK/Portable
--qlinear 8da4w           # LM decoder + acoustic transformer linears
--qembedding 8w           # text embedding
```

The `--qlinear` flag quantizes both the LM decoder (via `TextDecoderExport`)
and the acoustic transformer (via `DecodeAudioFrameExport`). The LM head
shares the output linear with the decoder; when quantized, weights are
untied first (`output.weight = Parameter(tok_embeddings.weight.clone())`)
so embedding and output get separate quantization configs.

## Export

Each exported method corresponds to a thin wrapper class in
`export_voxtral_tts.py`:

| Wrapper | Method | Wraps |
|---------|--------|-------|
| `TokenEmbeddingExport` | `token_embedding` | `decoder.tok_embeddings` |
| `TextDecoderExport` | `text_decoder` | `decoder.forward` (hidden states) |
| `LMHeadExport` | `lm_head` | `decoder.output` |
| `DecodeAudioFrameExport` | `decode_audio_frame` | `acoustic_transformer.decode_one_frame` |
| `AudioTokenEmbeddingExport` | `audio_token_embedding` | `audio_token_embedding` |

All exports use `torch.export.export(..., strict=True)` with explicit
bounded `Dim` specs. `Dim.AUTO` is avoided because it produces unbounded
ranges (`int_oo`) that fail in `SymShapeEvalPass` during `to_executorch()`.

`strict=True` is required because the XNNPACK KV cache uses `.item()` to
extract scalar positions (producing unbacked `SymInt`s).

## Checkpoint

Mistral format: `params.json` + `consolidated.safetensors`.

### Memory-efficient loading

`load_model()` follows the same pattern as voxtral_realtime:

1. **Meta device construction** — zero-storage parameter tensors.
2. **safetensors lazy access** — loads tensors on demand.
3. **`assign=True` state dict loading** — replaces meta tensors by reference.
4. **Post-load fixups** — re-tie output weights, materialize KV caches,
   recompute RoPE.

### Weight mapping

| Checkpoint prefix | Model prefix |
|------------------|-------------|
| `layers.*` | `decoder.layers.*` |
| `norm.weight` | `decoder.norm.weight` |
| `output.weight` | `decoder.output.weight` (or tied) |
| `mm_audio_embeddings.tok_embeddings.weight` | `decoder.tok_embeddings.weight` |
| `mm_audio_embeddings.audio_codebook_embeddings.embeddings.*` | `audio_token_embedding.embeddings.*` |
| `acoustic_transformer.*` | `acoustic_transformer.*` |
| `audio_tokenizer.*` | (codec.pte, separate loading) |

`decoder.output.weight` is tied to `decoder.tok_embeddings.weight`.
During export with quantization, the tie is broken so embedding and
output linear get separate quantization configs.

### Codec weight mapping

| Checkpoint prefix | Codec model prefix |
|------------------|-------------------|
| `audio_tokenizer.quantizer.semantic_codebook.*` | `quantizer.semantic.*` |
| `audio_tokenizer.quantizer.acoustic_codebook.*` | `quantizer.acoustic.*` |
| `audio_tokenizer.decoder_blocks.*` | `decoder_blocks.*` |
| `audio_tokenizer.output_proj.*` | `output_proj.*` |

Codec weights with `weight_norm` parametrizations are folded via
`remove_parametrizations` after loading.

## Differences from Voxtral Realtime

| Aspect | Voxtral Realtime (ASR) | Voxtral TTS |
|--------|----------------------|-------------|
| Direction | Audio → Text | Text → Audio |
| Encoder | Whisper (causal conv + transformer) | None |
| Decoder | Mistral + adaptive RMSNorm | Mistral (standard) |
| Extra head | — | Acoustic transformer (flow matching) |
| Audio codec | — | Encoder-decoder with ALiBi |
| KV cache | Decoder + streaming encoder | Decoder only |
| Output | Text tokens | Audio waveform |
| .pte files | 1 (model.pte) | 2 (model.pte + codec.pte) |
| Decoder returns | Logits | Hidden states (split lm_head) |

## References

- Reference implementation: vLLM-omni `voxtral_tts/` model
- Voxtral Realtime: `examples/models/voxtral_realtime/`

Upstream ExecuTorch patterns:

- KV cache: `examples/models/llama/source_transformation/custom_kv_cache.py`
- SDPA: `examples/models/llama/source_transformation/sdpa.py`
- RoPE: split-half convention (NOT `examples/models/llama/rope.py` which uses interleaved)
- Model loading: `examples/models/llama/model.py`
- Export builder: `extension/llm/export/builder.py`
