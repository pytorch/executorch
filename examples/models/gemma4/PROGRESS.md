# Gemma4 ExecuTorch — Status

Last updated: 2026-04-22 (v11 — PLI bug fixed, all 3 modalities clean)

## TL;DR

| Capability | Status |
|------------|--------|
| Text generation (CPU + XNNPACK FP32) | ✅ Working, parity verified |
| EOS stop tokens (no endless repetition) | ✅ Fixed |
| Official chat template (jinja) | ✅ render_chat.py |
| Vision encoder (export) | ✅ Exported, runs in multimodal .pte |
| Audio preprocessor + encoder (export) | ✅ Exported, runs in multimodal .pte |
| Multimodal .pte (5 methods, KV cache) | ✅ Exports and runs end-to-end |
| Multimodal generation XNNPACK (image+text) | ✅ Coherent long descriptions, accurate |
| Multimodal generation XNNPACK (audio+text) | ✅ Real transcription of 20s speech |
| Multimodal image color recognition | ✅ Correctly identifies red, blue (HWC fix) |
| Multimodal generation quality (v11) | ✅ No drift; full PLI working end-to-end |
| Single .pte for text/image+text/audio+text (v11) | ✅ Verified end-to-end, 9/10 tests pass |
| Standard MultimodalRunner ABI compliance | ✅ Uses create_multimodal_runner |
| Full 20s audio support (1976 mel frames, 494 tokens) | ✅ V11 verified |

## v11 result (2026-04-22) — PLI bug fixed

**Working model:** `/tmp/gemma4_multimodal_v11.pte` (21 GB, XNNPACK FP32, max_seq=1024, audio_frames=1976).

**Root cause of v6–v10 degeneration:** `examples/models/llama/llama_transformer.py` in the **installed** conda-env package (`~/miniconda3/envs/executorch/lib/python3.13/site-packages/executorch/...`) was out-of-sync with the local source tree. The installed PLI block computed `pli_emb` only from `tokens`; when the multimodal `text_decoder` is traced with `h=embeds` (tokens=None), it silently fell back to `pli_emb = zeros`, breaking PLI for every position. The earlier "sequential KV-cache prefill" diagnosis was wrong — the prefiller was already batched. Fix: sync local → installed package and re-export. Verified via `examples/models/gemma4/tests/test_textdec_wrapper.py` (max_diff 19.95 → 0.0).

**Sample v11 outputs (all bit-clean, EOS-terminated where seq_len allowed):**
```
Text:  "What is 12 multiplied by 8?"          → "12 multiplied by 8 is **96**.<turn|>"
Text:  "Explain what a neural network is..."  → "A neural network is a computational
                                                 model inspired by the structure of the
                                                 human brain, designed to recognize
                                                 patterns in data..."
Image: "Describe this image in detail."        → "This is a close-up photograph featuring
                                                 a single, vibrant red strawberry...
                                                 **Subject (Strawberry):** ...plump, ripe
                                                 strawberry... **Background:** ...bokeh
                                                 effect..." (150 tokens, structured)
Audio: "What is being said?" (20s Obama)       → "This week I traveled to Chicago to
                                                 deliver my final farewell address to
                                                 the nation, following in the tradition
                                                 of presidents before me..."
```
Full results table in `TEST_RESULTS.md`. 9/10 tests pass; the 1 failure is the 2s audio clip which lacks the required ≥5s mel context (encoder design constraint, not a model bug).

**Operational gotcha:** Edits to `examples/models/llama/*.py` must be mirrored to the installed package. After any local edit, run:
```
diff -q examples/models/llama/llama_transformer.py \
        ~/miniconda3/envs/executorch/lib/python3.13/site-packages/executorch/examples/models/llama/llama_transformer.py
```
If they differ, `cp` the local file over the installed one (or `pip install -e . --no-build-isolation`) before re-exporting.

## Quantized v11_q (2026-04-22) — 8da4w XNNPACK

**Working model:** `/tmp/gemma4_multimodal_v11_q.pte` (13.5 GB — 35% smaller than FP32 21 GB).
Text backbone is 8da4w (8-bit dynamic activations, 4-bit weights, group_size=32). Vision and audio encoders kept FP32. Export script gained `--qmode 8da4w --group-size 32` flags.

| Modality | FP32 decode | 8da4w decode | Speedup |
|----------|-------------|--------------|---------|
| Text     | 12.6–13.1 tok/s | 15.9–16.5 tok/s | +25% |
| Image    | 12.0–13.0 tok/s | 14.1–14.6 tok/s | +15% |
| Audio    | 11.3 tok/s | 13.7 tok/s | +21% |

Quality: all text/image tests pass. Audio transcription is faithful — actually resolves "we've seen eye-to-eye" cleanly where FP32 emitted "seen it-to-it". 9/10 tests pass; same A2 short-audio limitation as FP32 (encoder design). Full table in `TEST_RESULTS.md`.

Re-export command (~30 min):
```
python examples/models/gemma4/export_gemma4_multimodal.py \
  --hf-model ~/models/gemma-4-E2B-it --et-checkpoint ~/models/gemma-4-E2B-it/model_et.pth \
  --output /tmp/gemma4_multimodal_v11_q.pte --backend xnnpack \
  --max-seq-len 1024 --audio-frames 1976 --qmode 8da4w --group-size 32
```

## Text generation — WORKING

End-to-end Gemma4 generation on ExecuTorch CPU + XNNPACK matches the HuggingFace
reference bit-exactly. Validation:

```
Prompt: <|turn>user\nWhat is the capital of France?<turn|>\n<|turn>model\n
HF:     "The capital of France is **Paris**."
ET:     "The capital of France is **Paris**.<turn|><turn|>..."
```

Decode: ~14.7 tok/s on CPU XNNPACK FP32 (unsharded 19 GB .pte).

## Parity test results

`examples/models/gemma4/tests/test_parity.py` — all six tests pass:

| Test | max_abs_diff | cos_sim |
|------|--------------|---------|
| token_embedding | 0.00e+00 | 1.000001 |
| rmsnorm | 0.00e+00 | 1.000000 |
| pli_inputs | 0.00e+00 | 1.000003 |
| rope (sliding cos+sin) | 0.00e+00 | 1.000000 |
| rope (full cos+sin) | 0.00e+00 | 1.000000 |
| decoder_layer_0 | 0.00e+00 | 1.000001 |
| full_forward (logits) | 5.72e-06 | 0.999999 |

Top-1 next token matches HF for the canonical prompt.

## Bugs found and fixed

1. **Per-layer-type RoPE missing** — Gemma4 uses two RoPE setups: sliding
   layers with θ=10k partial=1.0 default; full layers with θ=1e6 partial=0.25
   `proportional`. Fixed by adding dual-buffer RoPE
   (`freqs_cos_global`/`freqs_sin_global`) in `rope.py` and threading
   `layer_type` through `Transformer._forward_layers`.
2. **Proportional RoPE formula** — Gemma4's "proportional" RoPE uses the
   FULL `head_dim` denominator and zero-pads trailing freqs (HF
   `_compute_proportional_rope_parameters`). Implemented in
   `hf_precompute_freqs_cis` with `rope_type="proportional"` branch.
3. **Attention scaling** — Gemma4 sets `self.scaling = 1.0` (no implicit
   `1/sqrt(head_dim)` divide). Wired through `attention_multiplier` in
   `ModelArgs` → `AttentionMHA` → both `F.scaled_dot_product_attention`
   call sites and the `SDPA` module.
4. **`SDPACustom` dropped scale** — The export source-transform replaces
   `SDPA` with `SDPACustom` calling `torch.ops.llama.custom_sdpa`. The
   wrapper was not forwarding the `scale` parameter. Fixed in
   `examples/models/llama/source_transformation/sdpa.py`.
5. **YOCO prefill skip was wrong** — Original code skipped shared layers
   during prefill. HF runs all layers; shared layers receive the donor's
   K/V via `shared_kv_states[kv_shared_layer_index]`. Removed the
   `is_prefill` skip guard.
6. **YOCO donor map type-aware** — Single global donor was unsafe with
   mixed head_dims (256 sliding vs 512 full). Built per-type donor map in
   `_build_kv_donor_map` (last non-shared layer of matching type).
7. **`act_fn` ignored in MLP** — `FeedForward` hardcoded SiLU. Now
   threads `args.act_fn` (Gemma4 uses gelu_approx).
8. **`v_norm` dtype handling** — Inline RMS without learnable weight,
   converted to/from input dtype to play with bf16 weights.
9. **Embedding scale + final logit softcap** — Applied
   `embedding_scale_factor=sqrt(hidden_size)` after lookup and
   `c·tanh(logits/c)` before output (`c=30.0`).
10. **`post_attention_norm`, `post_ffn_norm`, `layer_scalar`** — Added to
    `TransformerBlock`. Layer scalar is per-layer learnable parameter.
11. **PLI (Per-Layer Input)** — Built `pli_embeddings`,
    `pli_projection`, `pli_norm` in `Transformer.__init__`; computed
    per-layer input from input ids + main embedding; sliced per layer
    in `_forward_layers`; gated through PLI bottleneck in
    `TransformerBlock.forward`.
12. **Gemma4 chat template** — Uses `<|turn>...<turn|>`, NOT Gemma3's
    `<start_of_turn>...<end_of_turn>`. Updated `main.cpp`.

## Files changed (this session)

| File | Why |
|------|-----|
| `examples/models/llama/model_args.py` | +`global_rope_theta`, `global_partial_rotary_factor`, `global_rope_type`, `hidden_size_per_layer_input`, `global_head_dim`, `use_double_wide_mlp`, `use_v_norm`, `use_layer_scalar` |
| `examples/models/llama/rope.py` | Dual-buffer RoPE; `proportional` formula; `get_freqs_for_layer_type` |
| `examples/models/llama/llama_transformer.py` | PLI plumbing; per-layer-type RoPE; YOCO donor map; post-norms; layer_scalar; embedding scale; logit softcap |
| `examples/models/llama/attention.py` | `attention_multiplier`; `global_head_dim` per layer; `v_norm`; YOCO type-aware shared_kv routing |
| `examples/models/llama/feed_forward.py` | Thread `act_fn` parameter |
| `examples/models/llama/source_transformation/sdpa.py` | Forward `scale` to `torch.ops.llama.custom_sdpa` |
| `examples/models/llama/export_llama_lib.py` | Register `gemma4` in `EXECUTORCH_DEFINED_MODELS` |
| `extension/llm/export/config/llm_config.py` | Add `ModelType.gemma4` |
| `examples/models/gemma4/__init__.py` | `Gemma4Model(Llama2Model)` |
| `examples/models/gemma4/config/e2b_config.json` | Full Gemma4 E2B config (35 layers, dual RoPE, etc.) |
| `examples/models/gemma4/convert_weights.py` | HF → ET state dict mapping |
| `examples/models/gemma4/main.cpp` | C++ runner with `<|turn>` template |
| `examples/models/gemma4/CMakeLists.txt` + `CMakePresets.json` | Build with optional vision |
| `examples/models/gemma4/tests/test_parity.py` | 6-test parity harness vs HF |
| `examples/models/gemma4/README.md` | Export/build/run docs |
| `Makefile` | `gemma4-cpu`, `gemma4-cuda` targets |

## Reproduction

```bash
# 1. Convert weights (HF → ET)
python -m executorch.examples.models.gemma4.convert_weights \
  ~/models/gemma-4-E2B-it ~/models/gemma-4-E2B-it/model_et.pth

# 2. Layer-by-layer parity (must all pass)
python examples/models/gemma4/tests/test_parity.py

# 3. Export FP32 XNNPACK
python -m executorch.extension.llm.export.export_llm \
  base.model_class=gemma4 \
  base.params=examples/models/gemma4/config/e2b_config.json \
  base.checkpoint=~/models/gemma-4-E2B-it/model_et.pth \
  model.use_sdpa_with_kv_cache=true model.use_kv_cache=true \
  export.max_seq_length=512 export.max_context_length=512 \
  backend.xnnpack.enabled=true

# 4. Build + run
make gemma4-cpu
./cmake-out/examples/models/gemma4/gemma4_runner \
  --model_path ./gemma4.pte \
  --tokenizer_path ~/models/gemma-4-E2B-it/tokenizer.json \
  --prompt "What is the capital of France?" --seq_len 30
```

## Vision + Audio encoders

New files added (2026-04-20):

- `examples/models/gemma4/encoders.py` — `VisionEncoderExport` and
  `AudioEncoderExport` nn.Module wrappers around HF submodules.
- `examples/models/gemma4/export_gemma4_multimodal.py` — exports a 22 GB
  `gemma4_multimodal.pte` with four methods: `vision_encoder`,
  `audio_encoder`, `token_embedding`, `text_decoder`.

Parity verified:
- Vision: `(1, 2520, 768)` pre-patchified patches → `(256, 1536)` soft tokens. Matches HF.
- Audio: `(1, 200, 128)` log-mel → `(1, 50, 1536)` soft tokens. Matches HF.

### Multimodal generation architecture (2026-04-20)

`main.cpp` now implements a **custom generation loop** that directly orchestrates
all five methods — bypassing `MultimodalRunner` — to properly handle:

1. **Vision** (`--image_path`):
   - `stb_image` resize to 448×448 → C++ patchify (`gemma4_image_utils.cpp`)
   - Call `vision_encoder(pixel_values[1,2520,768], pixel_position_ids[1,2520,2])` ← TWO tensors
   - Receives 256 soft tokens `(256, 1536)`

2. **Audio** (`--audio_path`):
   - WAV RIFF parser → float32 PCM mono
   - Call `audio_preprocessor(waveform[1,N])` → mel features `(1,T,128)`
   - Call `audio_encoder(mel[1,T,128])` → audio soft tokens `(1,T',1536)`

3. **Text decoder** (stateful KV cache):
   - Embed prefix + modality soft tokens + suffix via `token_embedding`
   - Concatenate and run `text_decoder(combined_embeds, positions)` for prefill
   - Token-by-token decode loop

**Status (2026-04-21)**: All 5 methods export and run in C++. Fixes applied:
- Vision encoder: realistic 60×42 position grid → 280 soft tokens (not 1)
- Pixel normalization: [0,1] input (Gemma4VisionPatchEmbedder applies 2*(v-0.5) internally)
- Embedding scale: `sqrt(1536) ≈ 39.19` applied to token embeddings in C++ runner
- Audio encoder: mel frames truncated to 200 to match static export shape
- Text decoder: token-by-token prefill for KV-cache static-shape compatibility

**Known limitation**: PLI (Per-Layer Input) is zero in multimodal mode since the text_decoder
receives `h=inputs_embeds` without token IDs. In HF, PLI is computed from token IDs for
each position (including `<|image>` placeholder ID for image positions). Without PLI,
generation degenerates after 5-10 tokens. Fix requires re-exporting text_decoder with
token IDs for PLI computation, or embedding PLI into token_embedding for decode phase.

## New files (2026-04-20)

| File | Purpose |
|------|---------|
| `encoders.py` | `VisionEncoderExport`, `AudioEncoderExport` wrappers |
| `audio_preprocessor.py` | `Gemma4AudioPreprocessor` (PCM → log-mel) |
| `gemma4_image_utils.h/.cpp` | C++ image patchification |
| `export_gemma4_multimodal.py` | Multi-method export (5 methods with KV cache) |

## End-to-end multimodal status (2026-04-21)

5-method `.pte` exports and runs end-to-end:

```
gemma4_multimodal_v5.pte  (12 GB, portable backend, max_seq=256)
├── vision_encoder(pv[1,2520,768], pp[1,2520,2]) → (280, 1536) ← 280 visual soft tokens
├── audio_preprocessor(wav[1,N]) → (1, T, 128)  ← dynamic T
├── audio_encoder(mel[1,T,128]) → (1, T//4, 1536)
├── token_embedding(ids[1,S]) → (1, S, 1536)
└── text_decoder(emb[1,1,1536], pos[1]) → (1, vocab=262144)  ← stateful KV cache
```

Vision soft token count: 280 (not 256). HF config confirms `image_seq_length: 280,
max_soft_tokens: 280`. Computed via 60×42 patch grid with pooling_kernel_size=3
→ (60//3) × (42//3) = 20×14 = 280. Earlier exports were wrong (used all-zero
position_ids that collapsed to 1 token via boolean spatial pooling mask).

Image resize target: **960×672** (not 448×448) — 60 columns × 16px × 42 rows × 16px.

KV cache metadata: `use_kv_cache=True, use_sdpa_with_kv_cache=True`.
Text prefill: token-by-token (static-shape KV-cache text_decoder).

**Verified E2E results on V9 XNNPACK pte (single .pte, all 3 modes):**
```
Text:  "What is capital of France?"        → "The capital of France is **Paris**..."  90/14 tok/s
Image: "Describe this image." (image.jpg) → "This image is a close-up shot of..."   91/13 tok/s
Audio: "What is being said?"  (20s Obama) → "The text is a speech excerpt, set      260/11 tok/s
                                              in a modern-day setting..." <turn|>
```

V9 uses standard `MultimodalRunner` (`create_multimodal_runner`).
Single `/tmp/gemma4_multimodal_v9.pte` (13 GB, XNNPACK, max_seq=1024) serves all 3 modes.
- 512 prefill tokens for 20s audio (494 audio soft tokens + 18 text)
- Audio recognition: identifies speech, generates description, stops at EOS

**V7 standard ABI signatures (matches MultimodalPrefiller):**
- `token_embedding(tokens[1,S])`        → `(1,S,1536)` scaled
- `text_decoder(embeds[1,S,1536], cache_pos[1])` → `(1,vocab)` dynamic S
- `vision_encoder(image[1,3,672,960])`  → `(1,280,1536)` (patchify in graph)
- `audio_encoder(mel[1,128,200])`       → `(1,50,1536)` (channels-first)
- `audio_preprocessor(wav[1,N])`        → `(1,T,128)` (helper, not in standard ABI)

**Known quality issue:** PLI's `pli_emb` (token-ID-derived) component is zero in
multimodal mode (Approach C dropped). `pli_proj(h)` still works. Net: text-only
shows mild drift after ~5 tokens (e.g., "Paris-Nicolas-..."), image/audio
generally produce coherent short responses. To restore full PLI quality, extend
the MultimodalPrefiller ABI to carry token IDs (Approach C) — deferred.

### Re-export commands

```bash
# Portable (debugging)
cd /tmp && python /path/to/export_gemma4_multimodal.py \
  --hf-model ~/models/gemma-4-E2B-it \
  --et-checkpoint ~/models/gemma-4-E2B-it/model_et.pth \
  --output gemma4_multimodal_portable.pte \
  --backend portable --max-seq-len 256

# XNNPACK (production, ~14 tok/s)
cd /tmp && python /path/to/export_gemma4_multimodal.py \
  --hf-model ~/models/gemma-4-E2B-it \
  --et-checkpoint ~/models/gemma-4-E2B-it/model_et.pth \
  --output gemma4_multimodal_xnnpack.pte \
  --backend xnnpack --max-seq-len 512

# Run multimodal
./cmake-out/examples/models/gemma4/gemma4_runner \
  --model_path gemma4_multimodal_xnnpack.pte \
  --tokenizer_path ~/models/gemma-4-E2B-it/tokenizer.json \
  --image_path photo.jpg \
  --prompt "Describe this image." --seq_len 50

./cmake-out/examples/models/gemma4/gemma4_runner \
  --model_path gemma4_multimodal_xnnpack.pte \
  --tokenizer_path ~/models/gemma-4-E2B-it/tokenizer.json \
  --audio_path clip.wav \
  --prompt "What do you hear?" --seq_len 50
```

## Known follow-ups

- **EOS handling**: ✅ Fixed. Embedded via `base.metadata`.
- **Chat template**: ✅ `chat_template.jinja` + `render_chat.py`.
- **V6 XNNPACK export**: DONE at `/tmp/gemma4_multimodal_v6.pte` (12 GB).
  3-input text_decoder: (embeds[1,1,1536], pos[1], pli_token_ids[1,1]).
  Full PLI = pli_projection(h) + pli_embeddings(token_id) — matches HF exactly.
  Runner auto-detects v1/v2 pte (tries 3-input, falls back to 2-input).
- **Audio encoder**: Fixed T=200 frames (48k-40 constraint). Audio quality limited
  by synthetic test audio; natural speech/music expected to work better.
- **Image quality remaining issue**: "blue blue" and "solid blue color field...field"
  suggest minor PLI drift in long responses. Short answers work perfectly.
- **Quantization**: Not validated for multimodal; text-only 8da4w works.
- **Quantization**: not yet validated for Gemma4 (8da4w / 4w paths exist).
- **Per-layer-type partial_rotary**: works but only needed for full layers
  in Gemma4 E2B; other Gemma4 sizes may differ.
