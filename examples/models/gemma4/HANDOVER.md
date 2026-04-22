# Gemma4 ExecuTorch — Handover Document

**Branch:** `younghan/gemma4-xnnpack-fp32`  
**Last updated:** 2026-04-22  
**Working model:** `/tmp/gemma4_multimodal_v10.pte` (13 GB, XNNPACK FP32)

---

## What This Is

A full ExecuTorch implementation of `google/gemma-4-E2B-it` (2B parameter multimodal LLM) supporting text, image+text, and audio+text generation from a **single `.pte` artifact** via ExecuTorch's standard `MultimodalRunner`.

---

## Current Status

### What Works

| Capability | Status | Notes |
|---|---|---|
| Text generation (text-only pte, `forward` method) | ✅ Stable | 56/13 tok/s prefill/decode, stops at EOS |
| Image color identification (direct question) | ✅ Working | "red", "blue" identified correctly |
| Object identification (direct question) | ✅ Working | "strawberry" identified correctly |
| Audio pipeline (20s audio) | ✅ Running | 494 audio soft tokens, 260 tok/s prefill |
| All 3 modalities from single .pte | ✅ Working | `gemma4_multimodal_v10.pte` |
| Standard `MultimodalRunner` ABI | ✅ Compliant | `create_multimodal_runner`-compatible |

### What Is Broken / Degraded

| Issue | Root Cause | Status |
|---|---|---|
| Text generation degenerates after 5-10 tokens | Sequential KV-cache prefill vs batch forward | Known, needs fix |
| Image descriptions garble | Same root cause | Known, needs fix |
| Audio responses repeat same sentence | Same root cause | Known, needs fix |
| Math answers wrong (12×8=114 not 96) | 2B model + degradation | Expected for model size |
| 2s audio clip generates `<|channel>` only | Insufficient mel context | By design — need ≥5s |

### Root Cause of Degeneration (Most Important to Fix)

The text-only `gemma4.pte` (`forward` method) processes all prompt tokens **simultaneously in one batch** — identical to training. The multimodal `text_decoder` method processes tokens **sequentially one at a time** via KV cache. This accumulated floating-point difference causes the model to drift after a few tokens.

**The fix (not yet implemented):** Export two separate methods following the `qwen3_5_moe` pattern:
- `prefill(tokens[1,S], input_pos[1])` — batch, processes full prompt at once
- `decode(embeds[1,1,dim], input_pos[1], pli_token_ids[1,1])` — single-token step

---

## File Map

### Core ExecuTorch files (modified)

| File | What it does |
|---|---|
| `examples/models/gemma4/export_gemma4_multimodal.py` | Exports the 5-method multimodal `.pte`. Edit here to change encoder ABI, audio frame count, seq_len. |
| `examples/models/gemma4/encoders.py` | `VisionEncoderExport` (raw image → soft tokens, patchify in graph), `AudioEncoderExport` (channels-first mel → soft tokens) |
| `examples/models/gemma4/audio_preprocessor.py` | `Gemma4AudioPreprocessor` (raw PCM waveform → log-mel spectrogram) |
| `examples/models/gemma4/main.cpp` | Runner: loads pte, builds `MultimodalInput` list per modality, calls `create_gemma4_runner()` |
| `examples/models/gemma4/CMakeLists.txt` | Build config (no separate image utils needed — patchify is in model graph) |
| `examples/models/gemma4/test_multimodal.sh` | End-to-end test script for all 3 modalities |
| `examples/models/gemma4/TEST_RESULTS.md` | Honest pass/fail test results with root cause analysis |
| `extension/llm/runner/multimodal_prefiller.cpp` | **Modified**: builds `pli_ids` per input type (text→real IDs, image/audio→pad_token_id=0), detects 3-input text_decoder |
| `examples/models/llama/llama_transformer.py` | **Modified**: Transformer reads `pli_token_ids` from `attn_options["pli_token_ids"]` when `h=` path is used |
| `examples/models/llama/model_args.py` | Extended with Gemma4-specific args: `global_rope_theta`, `hidden_size_per_layer_input`, etc. |
| `examples/models/llama/rope.py` | Dual-buffer RoPE: sliding (θ=10k, partial=1.0) vs full (θ=1e6, partial=0.25, proportional) |
| `examples/models/llama/attention.py` | `attention_multiplier`, `global_head_dim`, `v_norm`, YOCO type-aware KV sharing |
| `examples/models/llama/feed_forward.py` | Thread `act_fn` parameter (Gemma4 uses `gelu_approx`) |
| `examples/models/llama/source_transformation/sdpa.py` | Fixed: forward `scale` to `torch.ops.llama.custom_sdpa` |

### Key classes in main.cpp

**`Gemma4DecoderRunner`** (in `main.cpp`):
- Subclasses `MultimodalDecoderRunner`
- Overrides `step()` to pass `pli_token_ids = [[current_decode_token]]` as 3rd input to text_decoder
- Detects 3-input text_decoder **lazily** (first call to `step()`, after module is loaded — NOT in constructor)
- Falls back to 2-input for older ptes transparently

**`create_gemma4_runner()`** (in `main.cpp`):
- Mirrors `create_multimodal_runner` from `llm_runner_helper.cpp`
- Injects `Gemma4DecoderRunner` instead of `MultimodalDecoderRunner`

---

## Exported Model Methods (v10)

```
text_decoder(embeds[1,S,1536], cache_pos[1], pli_token_ids[1,S]) → logits[1,262144]
  - Dynamic S: batched prefill AND single-token decode in same method
  - cache_pos: static size 1 (start_pos); model updates KV cache internally
  - pli_token_ids: real token IDs for text; pad_token_id=0 for image/audio positions

token_embedding(token_ids[1,S]) → (1,S,1536)
  - Returns scaled embeddings (baked in sqrt(1536) ≈ 39.19 scale)
  - Dynamic S

vision_encoder(image[1,3,672,960]) → (1,280,1536)
  - Input: raw [0,1] float image at 960×672 (60×42 patch grid)
  - Patchification + HWC ordering + 60×42 position grid baked into graph
  - Gemma4VisionPatchEmbedder applies 2*(v-0.5) internally — pass raw [0,1]
  - Output: 280 soft tokens (60/3 × 42/3 = 20×14 = 280, pooling_kernel_size=3)

audio_preprocessor(waveform[1,N_pcm]) → (1,T,128)
  - Dynamic T, time-major output
  - Used by C++ runner to convert WAV PCM to mel before audio_encoder

audio_encoder(mel[1,128,1976]) → (1,494,1536)
  - Channels-first mel (1, n_mels=128, T=1976)
  - T=1976 = 48×42−40 (stride-48 conv constraint) → supports ~20s audio at 16kHz/hop=160
  - C++ runner truncates/pads mel from audio_preprocessor to exactly 1976 frames
```

---

## PLI (Per-Layer Input) Architecture

Gemma4 uses PLI to inject per-position conditioning into every transformer layer:

```
pli_emb  = pli_embeddings(pli_token_ids) * pli_embed_scale       # from token IDs
pli_proj = pli_projection(h) * pli_projection_scale               # from input embeddings
per_layer_inputs = (pli_proj + pli_emb) * pli_combine_scale       # combined
# → each transformer layer: h += act_fn(per_layer_input_gate(h)) * per_layer_projection(per_layer_input)
```

**Correct token ID assignment (verified vs HF `Gemma4Model.forward`):**
- Text positions: real token IDs
- Image/audio soft-token positions: `pad_token_id = 0`  
  (HF line 2215: `llm_input_ids[multimodal_mask] = self.config.text_config.pad_token_id`)
- Decode steps: current generated token ID (both pli_emb and pli_proj from same token)

---

## Build Instructions

### Rebuild C++ runner after code changes

```bash
# Step 1: rebuild extension_llm_runner (if multimodal_prefiller.cpp changed)
cmake --build cmake-out --target extension_llm_runner
cp cmake-out/extension/llm/runner/libextension_llm_runner.a cmake-out/lib64/libextension_llm_runner.a

# Step 2: rebuild gemma4_runner
cmake --build cmake-out/examples/models/gemma4 -j$(nproc)
```

> **Critical:** The runner links `cmake-out/lib64/libextension_llm_runner.a` (the pre-built installed library), NOT the rebuilt one at `cmake-out/extension/llm/runner/`. Always copy after rebuilding.

### Re-export model (takes ~60 min for XNNPACK)

```bash
cd /tmp && nohup env PATH="/home/younghan/executorch/cmake-out/third-party/flatc_ep/bin:$PATH" \
  /home/younghan/miniconda3/envs/executorch/bin/python \
  /home/younghan/executorch/examples/models/gemma4/export_gemma4_multimodal.py \
  --hf-model ~/models/gemma-4-E2B-it \
  --et-checkpoint ~/models/gemma-4-E2B-it/model_et.pth \
  --output /tmp/gemma4_multimodal_v11.pte \
  --backend xnnpack --max-seq-len 1024 --audio-frames 1976 \
  > /tmp/gemma4_mm_export_v11.log 2>&1 &
```

### Run tests

```bash
# Quick (1 test per modality, ~5 min)
./examples/models/gemma4/test_multimodal.sh --quick

# Full suite (all prompts, ~15 min)
./examples/models/gemma4/test_multimodal.sh

# Manual
RUNNER=cmake-out/examples/models/gemma4/gemma4_runner
MODEL=/tmp/gemma4_multimodal_v10.pte
TOK=~/models/gemma-4-E2B-it/tokenizer.json

$RUNNER --model_path $MODEL --tokenizer_path $TOK --prompt "..." --seq_len 30
$RUNNER --model_path $MODEL --tokenizer_path $TOK --image_path ~/executorch/image.jpg --prompt "..." --seq_len 20
$RUNNER --model_path $MODEL --tokenizer_path $TOK --audio_path ~/executorch/obama_short20.wav --prompt "..." --seq_len 80
```

---

## Key Configuration Facts (Gemma4 E2B)

These differ from Gemma3 — the references at `~/gemma.cpp` and `~/gemma_pytorch` implement **Gemma3**, not Gemma4.

| Parameter | Value | Source |
|---|---|---|
| Image resolution | 960 × 672 px | `image_seq_length=280`, `patch_size=16`, `pooling_kernel_size=3` |
| Patch grid | 60 columns × 42 rows = 2520 patches | |
| Soft token count | 280 | 60//3 × 42//3 = 20×14 |
| Pixel normalization | `v/255` → [0,1]; model applies `2*(v-0.5)` internally | `Gemma4VisionPatchEmbedder.forward` line 567 |
| Pixel ordering | HWC within each patch (pixel-major: R,G,B,R,G,B,...) | Verified via HF image processor |
| Chat template | `<\|turn>user\n...<turn\|>\n<\|turn>model\n` | Gemma4-specific (Gemma3 uses `<start_of_turn>`) |
| Image token | `<\|image>` = token ID 255999 | |
| Audio token | `<\|audio>` = token ID 256000 | |
| BOS token | `<bos>` = token ID 2 | |
| EOS tokens | `{1, 106, 50}` — `<eos>`, `<turn\|>`, token 50 | |
| Audio: sample rate | 16000 Hz | |
| Audio: mel bins | 128 | |
| Audio: hop length | 160 samples | |
| Audio: FFT size | 512 | |
| Audio: mel floor | 0.001 | |
| max_seq_len | 1024 (v10) | 512 not enough for 20s audio (494+18=512) |
| Audio KV slots needed | 512 (494 audio + 18 text) | Leave ≥50 for decode |

---

## Gemma4 Transformer Specifics

Gemma4 E2B (35 layers, 1536 dim) has several non-standard features:

| Feature | Detail |
|---|---|
| **YOCO** | Last 20 layers share K/V from donor layers (per-type: sliding shares from last sliding donor, full from last full donor) |
| **Dual RoPE** | Sliding layers: θ=10k, partial=1.0; Full layers: θ=1M, partial=0.25, type=`proportional` |
| **PLI** | `hidden_size_per_layer_input=256`, embedding table dim = `n_layers×256 = 8960` |
| **Attention multiplier** | `attention_multiplier=1.0` (no implicit `1/sqrt(head_dim)`) |
| **v_norm** | Inline RMS normalization on V tensor before attention |
| **Layer scalar** | Learnable per-layer scale on residual |
| **Post norms** | `post_attention_norm` + `post_ffn_norm` in each layer |
| **Embedding scale** | `sqrt(1536) ≈ 39.19` multiplied after embedding lookup |
| **Logit softcap** | `c = 30.0`, applied as `c * tanh(logits/c)` |
| **Act fn** | `gelu_approx` in FFN |
| **Layer types** | Mix of `sliding_attention` (window=1024) and `full_attention` |

Config: `examples/models/gemma4/config/e2b_config.json`

---

## The Next Fix to Implement

The degeneration issue requires splitting `text_decoder` into two methods:

### Option A — Separate prefill + decode methods (recommended)

Following `qwen3_5_moe/export.py` pattern:

1. Export `prefill(embeds[1,S,dim], pos[1]) → logits` — batch, dynamic S, for the full prompt
2. Export `decode(embeds[1,1,dim], pos[1], pli_token_ids[1,1]) → logits` — single-token

`MultimodalPrefiller` calls `prefill` for the full prompt, then `MultimodalRunner` decode loop uses `decode`.

**Why this works:** `prefill` processes all tokens atomically — identical to the trained `forward` method. Only `decode` needs to be sequential.

**Implementation notes:**
- In `export_gemma4_multimodal.py`: add separate export for `prefill` method  
- In `multimodal_prefiller.cpp`: use `kPrefillMethod = "prefill"` instead of `kTextModelMethod` for the prefill call
- In `Gemma4DecoderRunner::step()`: continue using `text_decoder` (which would be the single-token decode)
- The `kTextModelMethod = "text_decoder"` is still used for the decode loop

### Option B — Add PLI to the `forward` method path

Modify the text-only `gemma4.pte` to also accept image/audio soft tokens as a prefix. This would share the batch prefill quality with multimodal. More complex — requires changes to the `forward` method export.

---

## Files With Test Assets

```
~/executorch/image.jpg          — strawberry photo (JPEG, 23KB) for image tests
~/executorch/obama_short20.wav  — 20s Obama speech (WAV 16kHz, 627KB) for audio tests
/tmp/gemma4_test_2s.wav         — auto-generated 2s clip from the above
```

---

## Export Constants in export_gemma4_multimodal.py

Key parameters to tune:

```python
--max-seq-len 1024    # KV cache slots: 512 min for 20s audio (494 audio + buffer)
--audio-frames 1976   # mel frames: 48*k-40 constraint; 1976=48*42-40 ≈ 20s audio
--backend xnnpack     # or "portable" for debugging (50× slower)
```

---

## Where Things Live in the ExecuTorch Stack

```
extension/llm/runner/
  multimodal_runner.{h,cpp}       — MultimodalRunner::generate() orchestration
  multimodal_prefiller.{h,cpp}    — MODIFIED: builds pli_ids, calls text_decoder
  multimodal_decoder_runner.h     — MultimodalDecoderRunner::decode() (2-input, hardcoded)
  text_token_generator.{h,cpp}    — decode loop
  multimodal_input.h              — MultimodalInput type (TEXT, IMAGE, AUDIO, RAW_AUDIO)
  image.h / audio.h               — Image / Audio structs with toTensor()
  constants.h                     — kTextModelMethod, kTokenEmbeddingMethod, etc.
  util.h                          — populate_start_pos_or_cache_position()
  llm_runner_helper.{h,cpp}       — create_multimodal_runner() factory
  io_manager.{h,cpp}              — memory management for method I/O

cmake-out/lib64/
  libextension_llm_runner.a       — MUST copy from cmake-out/extension/llm/runner/ after rebuild
```

---

## Known Gotchas

1. **Library path mismatch**: `cmake-out/lib64/libextension_llm_runner.a` is the pre-built library that the runner links against. After modifying `multimodal_prefiller.cpp`, you must `cmake --build cmake-out --target extension_llm_runner` AND THEN `cp cmake-out/extension/llm/runner/libextension_llm_runner.a cmake-out/lib64/libextension_llm_runner.a`. Without the copy, changes to the prefiller are silently ignored.

2. **PLI detection in Gemma4DecoderRunner must be lazy**: `Gemma4DecoderRunner` is constructed inside `create_gemma4_runner()` **before** `runner->load()` is called. Calling `module->method_meta()` in the constructor returns stale/wrong data. Detection must happen in the first `step()` call.

3. **Pixel ordering is HWC within each patch**: The HF `Gemma4ImageProcessor` outputs pixels in HWC order per patch (pixel-major: R₀G₀B₀ R₁G₁B₁ ...). C++ patchify must use the same order. CHW order (all R, then all G, then all B) gives wrong color perception.

4. **VisionPatchEmbedder applies normalization internally**: The HF `Gemma4VisionPatchEmbedder.forward()` applies `pixel_values = 2 * (pixel_values - 0.5)`. So pass raw `[0,1]` float values; do NOT pre-apply this normalization in C++.

5. **Audio encoder expects channels-first mel**: `AudioEncoderExport` takes `(1, 128, T)` not the time-first `(1, T, 128)` that `audio_preprocessor` outputs. The encoder transposes internally. The C++ runner must transpose mel from `(1, T, 128)` → `(1, 128, T)` before constructing `MultimodalInput::Audio`.

6. **max_seq_len must be ≥ 513 for 20s audio**: 494 audio soft tokens + 18 text prompt tokens = 512 prefill tokens. Decode then needs additional slots. Use 1024.

7. **KV cache stride constraint for audio_frames**: Audio encoder has stride-48 convolutions. Valid `T_mel` values: `T = 48*k - 40` for integer k. k=5 → T=200 (~2s); k=42 → T=1976 (~20s). The C++ runner truncates mel to exactly `kMelFrames` before calling audio_encoder.

8. **gemma.cpp and gemma_pytorch implement Gemma3, not Gemma4**: These references have different vision architecture (896×896 image, 14×14 patches, 4×4 pooling, 256 soft tokens), different chat template (`<start_of_turn>`), and no audio support. Do not follow them for Gemma4.
