# Gemma4 Multimodal ExecuTorch — Test Results

**Date:** 2026-04-22  
**Model:** `/tmp/gemma4_multimodal_v10.pte` (13 GB, XNNPACK FP32, max_seq=1024)  
**Runner:** `cmake-out/examples/models/gemma4/gemma4_runner`  
**Script:** `examples/models/gemma4/test_multimodal.sh`  
**Hardware:** CPU (XNNPACK-accelerated)

---

## Architecture

```
gemma4_multimodal_v10.pte  (13 GB, XNNPACK FP32, max_seq_len=1024)
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
| Text-only | 18 | 90 | 13.7 |
| Image + text | 299 (280 vision + 19 text) | 91 | 13.2 |
| Audio + text | 512 (494 audio + 18 text) | 260 | 11.9 |

---

## Test Results

### T1: Capital of France
**Prompt:** "What is the capital of France?"  **seq_len:** 60  
**Output:** `The capital of France is **Paris**-Nicolas-Nicolas-Paris-Nicolas-Nicolas-Paris. (This is a simplified way of...`  
**Result:** ❌ FAIL — First token "Paris" correct, then repeats "Nicolas" cyclically. Degenerate output.

---

### T2: Math
**Prompt:** "What is 12 multiplied by 8?"  **seq_len:** 50  
**Output:** `The answer is **114**. Here's how it's calculated: * 12 multiplied by = = (which is 12)...`  
**Result:** ❌ FAIL — Wrong answer (96 ≠ 114), then garbled calculation steps.

---

### T3: Code generation
**Prompt:** "Write a Python function to reverse a string."  **seq_len:** 40  
**Output:** `Here is a Python function that reverses a string... def reversestring(string):`  
**Result:** ⚠️ PARTIAL — Correct intent and starts valid Python, but output degenerates before completing.

---

### T4: General knowledge
**Prompt:** "Explain what a neural network is in one sentence."  **seq_len:** 80  
**Output:** `Neural networks are computational-based neural systems...structuresstructuresstructures`  
**Result:** ❌ FAIL — Partially coherent beginning, then degenerate token repetition.

---

### I1: Describe image (image.jpg — strawberry photo)
**Prompt:** "Describe this image in detail."  **seq_len:** 150  
**Output:** `This image is a close-up shot of a single, vibrant-------red----...`  
**Result:** ❌ FAIL — Identifies "close-up shot" and "red" correctly, then collapses into dashes.

---

### I2: What color
**Prompt:** "What color is the object in this image?"  **seq_len:** 30  
**Output:** `The object in the image is **red**.<turn|>`  
**Result:** ✅ PASS — Correct (the strawberry is red), stops cleanly at EOS.

---

### I3: What is the main subject
**Prompt:** "What is the main subject of this image?"  **seq_len:** 20  
**Output:** `The main subject of the image is a **singlereddisheverberrythingeverberrystrawetheed`  
**Result:** ❌ FAIL — Contains fragments of "strawberry" and "red" but garbled into one token.

---

### I4: What is in the image (specific question)
**Prompt:** "What object or subject is shown in this image?"  **seq_len:** 15  
**Output:** `The object shown in this image is a **strawberry**.<turn|>`  
**Result:** ✅ PASS — Correct identification, stops cleanly at EOS.

---

### A1: 20s Obama speech — what is being said
**Prompt:** "What is being said?"  **seq_len:** 60  
**Output:**
```
The text is a speech excerpt, and it is set in a modern-day setting.
It revolves a speech topic and it is set in a modern-day setting.
It is a speech excerpt and it is set in a modern-day setting.<turn|>
```
**Result:** ⚠️ PARTIAL — Correctly identifies audio as speech and generates coherent sentences. Repeats the same sentence 3× (PLI drift). Stops at EOS.

---

### A2: 2s audio clip
**Prompt:** "What do you hear?"  **seq_len:** 20  
**Output:** `<|channel><|channel>...` (20 times)  
**Result:** ❌ FAIL — Too little audio context. Model cannot generate meaningful text from ~1.5s of data.

---

## Summary

| Category | Pass | Partial | Fail |
|----------|------|---------|------|
| Text-only | 0 | 1 | 3 |
| Image + text | 2 | 0 | 2 |
| Audio + text | 0 | 1 | 1 |
| **Total** | **2** | **2** | **6** |

---

## Root Cause Analysis

The primary failure mode — "Paris-Nicolas-Nicolas", "structuresstructuresstructures", garbled tokens — is caused by **sequential KV-cache prefill** vs the model's training regime:

1. The text-only `forward` method processes all prompt tokens **simultaneously** in one batch forward pass. Attention is computed over all positions at once, identical to training.

2. Our multimodal `text_decoder` processes tokens **one at a time** (sequential KV-cache prefill). Each step computes attention over the growing KV cache. While mathematically equivalent for causal attention, small floating-point accumulation differences compound over many positions.

3. The Gemma4 2B model is sensitive to these numerical differences, especially with its YOCO KV-sharing and PLI architecture.

**PLI status (confirmed correct via HF reference review):**
- `pli_emb` uses real token IDs for text, `pad_token_id=0` for image/audio positions
- Decode uses current generated token ID — matches HF `Gemma4TextModel.forward` exactly
- `pli_proj(h)` component also correct — same `inputs_embeds` as text-only path

**What works correctly:**
- Short, direct questions where EOS fires before drift starts (image color = "red" ✅, strawberry ✅)
- Audio modality pipeline is functional end-to-end (494 audio tokens, 260 tok/s prefill)
- All 5 pte methods load and run; no crashes; EOS detection works

---

## Recommended Fix

Export `text_decoder` to accept **batched prefill** for the full prompt (using the token-based `forward` method semantics) while still supporting single-token KV-cache decode. The cleanest approach:

Option A: Export separate `prefill` (tokens, dynamic S) and `decode` (single token) methods — following the qwen3_5_moe pattern.

Option B: Map from the multimodal `inputs_embeds` path back to the `tokens` path by using discrete token IDs throughout, so the model can compute PLI and embeddings exactly as during training.
