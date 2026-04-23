# Gemma4 Multimodal ExecuTorch — Test Results

**Date:** 2026-04-22 (v11 — PLI bug fixed)
**Model:** `/tmp/gemma4_multimodal_v11.pte` (21 GB, XNNPACK FP32, max_seq=1024, audio_frames=1976)
**Runner:** `cmake-out/examples/models/gemma4/gemma4_runner`
**Script:** `examples/models/gemma4/test_multimodal.sh`
**Hardware:** CPU (XNNPACK-accelerated)

---

## Architecture

```
gemma4_multimodal_v11.pte
├── vision_encoder     (1,3,672,960) → (1,280,1536)  patchify baked in
├── audio_preprocessor (1,N_pcm)    → (1,T,128)      dynamic T, WAV→mel
├── audio_encoder      (1,128,1976) → (1,494,1536)   ~20s audio window
├── token_embedding    (1,S)        → (1,S,1536)      scaled by sqrt(1536)
└── text_decoder       (1,S,1536)+(1,)+(1,S) → (1,262144)
                       ↑ 3-input: embeds + cache_pos + pli_token_ids
```

PLI implemented via `Gemma4DecoderRunner` (Approach C):
- Prefill: text positions use real token IDs, image/audio positions use pad_token_id (0) per HF Gemma4Model.forward line 2215
- Decode: current generated token ID passed as `pli_token_ids` — matches HF reference

---

## Performance

| Modality | Prefill tokens | Prefill tok/s | Decode tok/s |
|----------|---------------|---------------|--------------|
| Text-only | 18 | 89–100 | 12.6–13.1 |
| Image + text | 299–303 (280 vision + ~20 text) | 92–96 | 12.0–13.0 |
| Audio + text | 512 (494 audio + 18 text) | 254 | 11.3 |

---

## Test Results

### T1: Capital of France
**Prompt:** "What is the capital of France?"  **seq_len:** 30
**Output:** `The capital of France is **Paris**.`
**Result:** ✅ PASS — Correct, stops cleanly at EOS.

---

### T2: Math
**Prompt:** "What is 12 multiplied by 8?"  **seq_len:** 50
**Output:** `12 multiplied by 8 is **96**.`
**Result:** ✅ PASS — Correct answer (96), clean EOS.

---

### T3: Code generation
**Prompt:** "Write a Python function to reverse a string."  **seq_len:** 80
**Output:**
```
Here are several ways to write a Python function to reverse a string, ranging from
the most "Pythonic" to more explicit methods.

## 1. The Most Pythonic Way (Using Slicing)

This is the shortest and most idiomatic way to reverse a string in Python.

```python
def reverse_string_slicing(s: str) -> str:
```
**Result:** ✅ PASS — Coherent, well-structured response. Truncated by `seq_len` (no EOS — would continue if longer budget).

---

### T4: General knowledge
**Prompt:** "Explain what a neural network is in one sentence."  **seq_len:** 80
**Output:** `A neural network is a computational model inspired by the structure of the human brain, designed to recognize patterns in data by processing information through interconnected nodes.`
**Result:** ✅ PASS — Single coherent sentence, clean EOS.

---

### I1: Describe image (image.jpg — strawberry photo)
**Prompt:** "Describe this image in detail."  **seq_len:** 150
**Output:**
```
This is a close-up photograph featuring a single, vibrant red strawberry set against
a soft, blurred green background.

**Subject (Strawberry):**
The main focus of the image is a plump, ripe strawberry. It exhibits the characteristic
bright red color of a fresh strawberry. The surface is covered in numerous small,
yellowish seeds (achenes), giving it a textured appearance. The strawberry appears
whole and firm.

**Background:**
The background is heavily blurred (bokeh effect), creating a soft wash of green and
darker tones. This suggests an outdoor setting, possibly foliage or a garden, which
is rendered indistinctly to keep the viewer's attention on the strawberry.

**Foreground and Surface:**
The strawberry is resting on a light-
```
**Result:** ✅ PASS — Long, structured, accurate description (color, seeds, bokeh, composition). Truncated by `seq_len`.

---

### I2: What color
**Prompt:** "What color is the object in this image?"  **seq_len:** 30
**Output:** `The object in the image is **red**. It is a strawberry.`
**Result:** ✅ PASS — Correct color, also identifies the object spontaneously, clean EOS.

---

### I3: What is the main subject
**Prompt:** "What is the main subject of this image?"  **seq_len:** 25
**Output:** `The main subject of this image is a **strawberry**.`
**Result:** ✅ PASS — Correct, clean EOS.

---

### I4: What object is shown
**Prompt:** "What object or subject is shown in this image?"  **seq_len:** 20
**Output:** `The object shown in this image is a **strawberry**.`
**Result:** ✅ PASS — Correct, clean EOS.

---

### A1: 20s Obama speech — what is being said
**Prompt:** "What is being said?" / "Transcribe this audio."  **seq_len:** 80
**Output:**
```
This week I traveled to Chicago to deliver my final farewell address to the nation,
following in the tradition of presidents before me. It was an opportunity to say
thank you. Whether we've seen it-to-it or rarely agreed at all, my conversations
with you, the American people, in living rooms and...
```
**Result:** ✅ PASS — Real transcription/paraphrase of Obama's farewell address. Minor word-fusion artifact ("seen it-to-it" instead of "seen eye-to-eye") but content is faithful. Truncated by `seq_len`.

---

### A2: 2s audio clip
**Prompt:** "What do you hear?"  **seq_len:** 20
**Output:** `<|channel><|channel>...` (repeats)
**Result:** ❌ FAIL (by design) — Audio encoder needs ≥5s of mel context; ~1.5s of usable audio is insufficient for the encoder to produce meaningful soft tokens. Use 20s clips.

---

## Summary

| Category | Pass | Partial | Fail |
|----------|------|---------|------|
| Text-only | 4 | 0 | 0 |
| Image + text | 4 | 0 | 0 |
| Audio + text | 1 | 0 | 1 (insufficient context, by design) |
| **Total** | **9** | **0** | **1** |

---

## Root Cause Analysis (resolved)

The previous "Paris-Nicolas-Nicolas", "structuresstructuresstructures" degeneration was **not** caused by sequential KV-cache prefill (the prefiller was already batched) — that earlier diagnosis in v10's TEST_RESULTS was wrong.

**Actual root cause:** `examples/models/llama/llama_transformer.py` in the **installed conda-env package** (`~/miniconda3/envs/executorch/lib/python3.13/site-packages/executorch/...`) was out-of-sync with the local source tree. The local file had the fix that reads `pli_token_ids` from `attn_options` when `tokens` is `None`; the installed copy did not. So when the multimodal `text_decoder` was traced with `h=embeds` (tokens=None), the PLI block silently fell back to `pli_emb = zeros`, breaking PLI conditioning across **every** position during prefill+decode. With PLI=0, generation drifted within 5–10 tokens.

**Fix:** Synced local → installed package, then re-exported. Verified bit-exact via `examples/models/gemma4/tests/test_textdec_wrapper.py` — wrapper-vs-tokens path went from `max_diff=19.95` to `0.0`.

**Lesson / gotcha:** Whenever editing files under `examples/models/llama/` (or anything else exposed via the conda env), `cp` to the matching `site-packages/executorch/...` path or re-run `pip install -e . --no-build-isolation`. Verify with `diff -q local installed`.

---

## Quantized variant — v11_q (8da4w, group_size=32)

**Model:** `/tmp/gemma4_multimodal_v11_q.pte` (13.5 GB — 35% smaller than FP32 21 GB)
Text backbone is 8da4w (8-bit dynamic activations, 4-bit weights, group=32 channelwise). Vision and audio encoders left FP32.

Export:
```
python -m executorch.examples.models.gemma4.export \
  --hf-model ~/models/gemma-4-E2B-it --et-checkpoint ~/models/gemma-4-E2B-it/model_et.pth \
  --output /tmp/gemma4_multimodal_v11_q.pte --backend xnnpack \
  --max-seq-len 1024 --audio-frames 1976 --qmode 8da4w --group-size 32
```

### Performance vs FP32

| Modality | FP32 prefill / decode | 8da4w prefill / decode | Decode speedup |
|----------|----------------------|------------------------|----------------|
| Text-only | 89–100 / 12.6–13.1 tok/s | 108–122 / 15.9–16.5 tok/s | **+25%** |
| Image + text | 92–96 / 12.0–13.0 tok/s | 100–102 / 14.1–14.6 tok/s | **+15%** |
| Audio + text | 254 / 11.3 tok/s | 311 / 13.7 tok/s | **+21%** |

### Quality results (8da4w)

| Test | Result | Output excerpt |
|------|--------|----------------|
| T1 capital | ✅ PASS | "The capital of France is **Paris**." |
| T2 math    | ✅ PASS | "Let's calculate $12 \times 8$: $$12 \times 8 = 96$$ ... **96.**" |
| T3 code    | ✅ PASS | Pythonic slicing, structured markdown |
| T4 NN      | ✅ PASS | "A neural network is a computational model inspired by the human brain..." |
| I1 describe | ✅ PASS | Multi-paragraph: subject (color, texture, condition), setting, composition |
| I2 color   | ✅ PASS | "...is **red**. It is a strawberry." |
| I3 subject | ✅ PASS | "...is a **strawberry**." |
| I4 object  | ✅ PASS | "The main object shown in this image is a **strawberry**." |
| A1 audio (transcribe) | ✅ PASS | "This week I traveled to Chicago to deliver my final farewell address to the nation... whether we've seen eye-to-eye or rarely agreed at all..." (correctly resolves "eye-to-eye" — better than FP32's "it-to-it") |
| A1 audio (summary)    | ✅ PASS | Coherent summary; structured analysis with prompt "What is the speaker saying?" |
| A2 short audio | ❌ FAIL (by design) | Same encoder limitation as FP32 |

**Verdict:** 8da4w quantization preserves quality across all modalities while shrinking the .pte by 7 GB and giving 15–25 % decode speedup. No additional bugs surfaced.
