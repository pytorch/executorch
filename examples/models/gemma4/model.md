# Gemma 4 — Architecture & Design Notes

Developer reference for `examples/models/gemma4/`. For export/usage instructions
see [README.md](README.md). For test results see [TEST_RESULTS.md](TEST_RESULTS.md).

This file documents *why* each design decision was made, with citations to
Google's reference implementations
([gemma_pytorch](https://github.com/google/gemma_pytorch),
[gemma.cpp](https://github.com/google/gemma.cpp)) and the HuggingFace Gemma 4
model code so future maintainers can verify against the source of truth.

## Architecture

```
Image  ----> vision_encoder       (HF Gemma4VisionModel; ViT 16-layer, 8-bit)
                  |                     -> (B, 280, 1536)  soft tokens
                  v
Audio  ----> speech_transform  -> audio_encoder         (HF Gemma4AudioModel; USM Conformer)
              (PCM->mel)               |                     -> (B, 494, 1536)  soft tokens
                                       v
Text   ----> token_embedding     ---->  text_decoder
              (scaled by sqrt(d))         (35-layer Transformer with
                                           PLE / dual RoPE / YOCO)
                                              |
                                              v
                                          logits  (1, vocab=262144)
```

Single `.pte` exposes 5 methods (or 4 if both `--no-audio` and `--no-vision` are
deferred): `token_embedding`, `text_decoder`, `vision_encoder`,
`audio_preprocessor`, `audio_encoder`. The C++ runner uses ExecuTorch's standard
`MultimodalRunner` plus a small `Gemma4DecoderRunner` subclass that adds Gemma 4's
per-layer-input (PLE) token IDs as a 3rd input to `text_decoder`.

## Variants

| | E2B | E4B |
|---|---|---|
| `hidden_size` | 1536 | 2560 |
| `num_hidden_layers` | 35 | 42 |
| `num_attention_heads` | 8 | 16 |
| `num_kv_heads` | 1 (MQA) | 2 |
| `head_dim` (sliding) | 256 | 256 |
| `head_dim` (full, `global_head_dim`) | 512 | 512 |
| `intermediate_size` (FFN) | 6912 | 12288 |
| `vocab_size` | 262144 | 262144 |
| `hidden_size_per_layer_input` (PLE) | 256 | 256 |
| `attention_multiplier` | 1.0 | 1.0 |
| Embedding scale | sqrt(1536) ≈ 39.19 | sqrt(2560) ≈ 50.59 |
| Final logit softcap | 30.0 · tanh(x/30) | 30.0 · tanh(x/30) |

## Key Components

| Component | Description | Reference |
|---|---|---|
| **GemmaRMSNorm** | `x / sqrt(mean(x²) + eps) * (1 + weight)` — unit-offset (Gemma family). | gemma_pytorch/gemma/model.py:166-189 |
| **Embedding scale** | Multiply token embedding by `sqrt(hidden_size)` after lookup. | gemma_pytorch/gemma/model.py:273 |
| **PLE (Per-Layer Embedding)** | Separate `(vocab, n_layers × 256)` embedding table, sliced per layer; combined with `pli_projection(h)`; gated through bottleneck `act_fn(per_layer_input_gate(h)) * per_layer_projection(per_layer_input)`. | HF `Gemma4Model.forward` (image/audio positions use `pad_token_id=0`); HF `Gemma4DecoderLayer.forward` |
| **Dual RoPE (sliding)** | θ=10k, partial_rotary_factor=1.0 (rotate full head_dim), no scaling. | HF `Gemma4Config.rope_parameters["sliding_attention"]` |
| **Dual RoPE (full, "proportional")** | θ=1e6, partial_rotary_factor=0.25; denominator stays at full `head_dim`; trailing dims zero-padded so cos=1 sin=0 (pass-through). | HF `_compute_proportional_rope_parameters` |
| **YOCO KV sharing** | Last 20 layers share K/V from a "donor" layer. Per-type donor map: sliding-shared layer takes from last sliding donor, full-shared from last full donor (Gemma 4 mixes head dims so a single global donor is wrong). | "You Only Cache Once" (Sun et al.); HF `Gemma4Attention` shared-cache wiring |
| **Attention multiplier = 1.0** | Gemma 4 sets `self.scaling = 1.0`. No implicit `1/sqrt(head_dim)` divide; the QK softmax temperature is baked into the model. | HF `Gemma4Attention.__init__` |
| **v_norm** | Inline RMS normalization on V tensor before attention. dtype-aware so it composes with bf16 weights. | HF `Gemma4Attention.forward` |
| **Layer scalar** | Learnable scalar multiplier on the attention residual per layer. | HF `Gemma4DecoderLayer.layer_scalar` |
| **Post-norms** | `post_attention_norm` + `post_ffn_norm` in each layer, applied *before* the residual add (Gemma 2/3/4 pattern). | gemma_pytorch/gemma/model.py:Gemma2DecoderLayer |
| **Activation function** | `gelu_approx` ("Gelu approximate tanh") in FFN. | gemma.cpp/gemma/activations.h |
| **Final logit softcap** | `c · tanh(logits / c)`, c=30.0 (Gemma 2 softcap). | Gemma 2 paper §3.4 |

## Method ABI

| Method | Inputs | Output | Notes |
|---|---|---|---|
| `token_embedding` | `token_ids[1, S]` | `(1, S, hidden)` | Output is pre-scaled by sqrt(hidden), ready to feed `text_decoder`. |
| `text_decoder` | `embeds[1, S, hidden]`, `cache_position[1]`, `pli_token_ids[1, S]` | `logits[1, vocab]` | Dynamic S; static cache_position size 1; pli_token_ids: real token IDs for text positions, `pad_token_id=0` for image/audio soft-token positions (matches HF `Gemma4Model.forward` line 2215). |
| `vision_encoder` | `image[1, 3, H, W]` (E2B: 672×960) in `[0, 1]` | `(1, 280, hidden)` | Patchification + 60×42 position grid baked into graph. The `Gemma4VisionPatchEmbedder` applies `2*(v - 0.5)` internally — pass raw `[0, 1]` floats. |
| `audio_preprocessor` | `waveform[1, N_pcm]` | `(1, T, 128)` | Dynamic T, time-major. PCM→log-mel. |
| `audio_encoder` | `mel[1, 128, T_mel]` (channels-first; T_mel=1976 for ~20 s) | `(1, T_mel/4, hidden)` | T_mel must satisfy `T_mel = 48*k - 40` (stride-48 conv constraint). |

## Quantization

`export.py --qmode 8da4w --group-size 32` quantizes the text backbone (Linear
weights → int4 grouped, activations → int8 dynamic) via TorchAO's
`Int8DynamicActivationIntxWeightConfig`. Vision and audio encoders are kept FP32
by default (`--audio-quantize none --vision-quantize none`); flip those flags
to quantize them too.

## Tokenizer & Chat Template

We use the HuggingFace `tokenizer.json` (preferred over sentencepiece's
`tokenizer.model` — both files exist in the HF checkpoint, but `tokenizer.json`
is the source of truth for the special tokens added to Gemma 4).

Special tokens used:
- `<bos>` = 2
- `<eos>` = 1
- `<|turn>` = 105 (begin assistant/user/system turn)
- `<turn|>` = 106 (end of turn — primary EOS for chat)
- `<|image>` = 255999 (image placeholder; encoded by template, soft tokens injected by encoder)
- `<|audio>` = 256000 (audio placeholder; same pattern)

Chat template: `<bos><|turn>user\n[<|image>\n][<|audio>\n][prompt]<turn|>\n<|turn>model\n`.
The official jinja template is at `chat_template.jinja`; render via `render_chat.py`.

## References

- HF Gemma 4 model card: https://huggingface.co/google/gemma-4-E2B-it
- HF transformers Gemma 4 source: `transformers.models.gemma4.modeling_gemma4`
- gemma_pytorch (Gemma 1/2/3 reference): https://github.com/google/gemma_pytorch — *note: does not include Gemma 4 yet; used for shared primitives*
- gemma.cpp (C++ reference): https://github.com/google/gemma.cpp — *also Gemma 1/2/3*
- "You Only Cache Once" (YOCO): https://arxiv.org/abs/2405.05254
- ExecuTorch MultimodalRunner: `extension/llm/runner/multimodal_runner.h`
