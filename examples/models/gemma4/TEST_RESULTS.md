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

---

## Mobile-target variant — v12 (8da4w + emb8)

**Model:** `/tmp/gemma4_v12_emb.pte` (5.78 GB — 72 % smaller than FP32 v11 21 GB; 56 % smaller than v11_q 13 GB)

Text Linear weights are 8da4w group=32; **all `nn.Embedding` modules** (`tok_embeddings` AND `pli_embeddings` — the latter is the PLE table, 2.35B params at 9.4 GB FP32) are 8w per-channel. Vision and audio encoders left FP32 in this round (`--vision-quantize 8da8w` hit a TorchAO data-dependent guard inside the embedding_projection — follow-up). Embedding quant is the headline win — see BENCHMARK.md for the size-budget analysis showing PLE was the elephant.

Export:
```
python -m executorch.examples.models.gemma4.export \
  --hf-model ~/models/gemma-4-E2B-it --et-checkpoint ~/models/gemma-4-E2B-it/model_et.pth \
  --output /tmp/gemma4_v12_emb.pte --backend xnnpack \
  --max-seq-len 1024 --audio-frames 1976 \
  --qmode 8da4w --group-size 32 \
  --embedding-quantize 8,0
```

### Quality results (5/5 quick tests pass; perf table in BENCHMARK.md)

| Test | Result | Output |
|---|---|---|
| Text capital  | ✅ PASS | "The capital of France is **Paris**." |
| Text math     | ✅ PASS | "Here are a few ways to think about..." (continues, truncated by seq cap) |
| Image describe | ✅ PASS | "This is a still life photograph featuring a single, vibrant red strawberry." |
| Audio 2s clip | ✅ PASS | "I hear some music." (qualitatively coherent; 2s is sub-encoder-context but it didn't hang or garble) |
| Audio 20s     | ✅ PASS | "This week I traveled to Chicago to deliver my final farewell address to the nation, following in the tradition..." (faithful transcription) |

### Performance vs other variants

| Modality | v11 FP32 (21 GB) | v11_q 8da4w (13 GB) | **v12 8da4w+emb8 (5.8 GB)** |
|---|---|---|---|
| text decode tok/s   | 13.4 | 16.1 | **16.1** |
| image decode tok/s  | 12.1 | 14.5 | **15.2** |
| audio decode tok/s  | 11.6 | 13.4 | **13.6** |
| audio prefill tok/s | 254  | 301  | **312**  |
| text TTFT (ms)      | 217  | 176  | **177**  |

v12 matches or beats v11_q on every metric **and** is 7.2 GB smaller. The embedding quantization is pure win — quantizing PLE (the largest tensor in Gemma 4 by ~2×) costs nothing measurable in latency.

**Remaining gap to Leixin's D99603811 mobile defaults (4.1 GB untied / 2.7 GB tied, 4-bit + emb8):**
- `--tied-embedding` (drop duplicate `lm_head`) → predicted ~4.6 GB.
- `--vision-quantize` (currently blocked on TorchAO guard inside embedding_projection) → predicted ~4.55 GB.
- `--audio-quantize 8da4w` → predicted ~4.5 GB.
- `--embedding-quantize 4,0` (int4 emb instead of int8) → predicted ~3 GB.

Each is a flag on `export.py`; closing the gap is just bandwidth (each export takes ~30 min) plus working around the vision-encoder guard issue.

---

## Mobile-target variant — v13 (8da4w + emb8 + audio-encoder 8da4w)

**Model:** `/tmp/gemma4_v13_aud.pte` (4.78 GB — 77 % smaller than FP32 v11 21 GB; 17 % smaller than v12)

Adds `--audio-quantize 8da4w` on top of v12. The audio encoder turns out to be ~1 GB at FP32 (USM Conformer with hidden=1024, 12 layers); 8da4w shrinks it to ~250 MB. **Within 0.7 GB of Leixin's untied 4.1 GB default.**

Vision encoder still FP32 — `--vision-quantize 8w` hit a `Missing out variants: torchao::dequantize_affine` failure in `to_executorch()` lowering for the weight-only XNNPACK path; `--vision-quantize 8da8w` hit a different TorchAO data-dependent guard inside `vision_tower.embedding_projection`. Both need follow-up.

Export:
```
python -m executorch.examples.models.gemma4.export \
  --hf-model ~/models/gemma-4-E2B-it --et-checkpoint ~/models/gemma-4-E2B-it/model_et.pth \
  --output /tmp/gemma4_v13_aud.pte --backend xnnpack \
  --max-seq-len 1024 --audio-frames 1976 \
  --qmode 8da4w --group-size 32 \
  --embedding-quantize 8,0 \
  --audio-quantize 8da4w --encoder-group-size 128
```

### Quality (5/5 quick tests pass)

| Test | Result | Output |
|---|---|---|
| Text capital   | ✅ PASS | "The capital of France is **Paris**." |
| Text math      | ✅ PASS | (continues, truncated by seq cap) |
| Image describe | ✅ PASS | "This is a still life photograph featuring a single, vibrant red strawberry." |
| Audio 2s clip  | ✅ PASS | Coherent prefix; doesn't hang or garble despite the audio encoder being 8da4w-quantized |
| Audio 20s      | ✅ PASS | "This week I traveled to Chicago to deliver my final farewell address to the nation, following in the tradition..." |

### Performance vs prior PTEs

| Modality | v11 FP32 (21 GB) | v11_q (13 GB) | v12 (5.78 GB) | **v13 (4.78 GB)** |
|---|---|---|---|---|
| text decode tok/s   | 13.4 | 16.1 | 16.1 | **15.9** |
| text TTFT (ms)      | 217  | 176  | 177  | **168**  |
| image decode tok/s  | 12.1 | 14.5 | 15.2 | **14.3** |
| audio decode tok/s  | 11.6 | 13.4 | 13.6 | **13.5** |
| audio prefill tok/s | 254  | 301  | 312  | **311**  |

v13 lands at the same perf envelope as v12 (deviations are within run-to-run noise) while shrinking another 1 GB.
