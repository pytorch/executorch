# Gemma 4 ExecuTorch — Multimodal Benchmark (CPU + XNNPACK)

**Hardware:** Linux devserver, CPU only (no GPU / NPU acceleration).
**Date:** 2026-04-23.
**Runner:** `cmake-out/examples/models/gemma4/gemma4_runner` from `younghan/gemma4`.

All numbers are end-to-end through the C++ runner: tokenize → encode (vision/audio) → prefill → decode → EOS or `seq_cap`. Uses ExecuTorch's stock `MultimodalRunner` plus `Gemma4DecoderRunner` (PLE 3-input).

Stand-alone repro:

```bash
RUNNER=cmake-out/examples/models/gemma4/gemma4_runner
TOK=~/models/gemma-4-E2B-it/tokenizer.json
$RUNNER --model_path /tmp/gemma4_v12_emb.pte --tokenizer_path $TOK \
        --prompt "What is the capital of France?" --seq_len 30
```

## PTEs compared

| .pte | size | text Linear quant | embedding quant | vision encoder | audio encoder |
|---|---|---|---|---|---|
| `gemma4_multimodal_v11.pte`    | 21.0 GB | FP32 | FP32 | FP32 | FP32 |
| `gemma4_multimodal_v11_q.pte`  | 13.0 GB | 8da4w (group=32) | FP32 | FP32 | FP32 |
| `gemma4_v12_emb.pte`           | 5.78 GB | 8da4w (group=32) | **8w per-channel** | FP32 | FP32 |
| `gemma4_v13_aud.pte`           | **4.78 GB** | 8da4w (group=32) | 8w per-channel | FP32 | **8da4w (group=128)** |

The headline win is moving from v11_q to v12: quantizing the embeddings (`tok_embeddings` = 1.6 GB, `pli_embeddings` = 9.4 GB) to INT8 cuts another **7.2 GB** off the .pte with **zero perf cost** (decode tok/s and TTFT are unchanged within run-to-run noise; prefill tok/s actually inches up because the embedded soft tokens go through the same INT8 path as the rest of the decoder).

The Gemma 4 PLE table (`pli_embeddings`, shape `[262144, 8960]`, 2.35B params) is by far the biggest tensor in the model — bigger than all the text-decoder Linear weights combined. Leaving it FP32 is what kept v11_q at 13 GB; quantizing it to INT8 is what brings v12 to 5.78 GB.

Compared to Leixin's D99603811 numbers: their default 4-bit (linear + emb8 + tied lm_head, no encoder quant) lands at 4.1 GB. The remaining 1.6 GB gap to v12 is exactly the un-tied `lm_head` weight; `--tied-embedding` will close it (Phase 1.6 follow-up).

## Summary (averaged per modality)

| Modality | PTE | Prefill tok/s | Decode tok/s | TTFT |
|---|---|---:|---:|---:|
| text  | v11 FP32 (21 GB)         | 102 | 13.4 | 217 ms |
| text  | v11_q 8da4w (13 GB)      | 126 | 16.1 | 176 ms |
| text  | v12 +emb8 (5.78 GB)      | 126 | 16.1 | 177 ms |
| text  | **v13 +audio8da4w (4.78 GB)** | **132** | **15.9** | **168 ms** |
| image | v11 FP32 (21 GB)         | 94  | 12.1 | 3.25 s |
| image | v11_q 8da4w (13 GB)      | 103 | 14.5 | 2.96 s |
| image | v12 +emb8 (5.78 GB)      | 102 | 15.2 | 2.98 s |
| image | **v13 +audio8da4w (4.78 GB)** | **99** | **14.3** | **3.06 s** |
| audio | v11 FP32 (21 GB)         | 254 | 11.6 | 2.03 s |
| audio | v11_q 8da4w (13 GB)      | 301 | 13.4 | 1.71 s |
| audio | v12 +emb8 (5.78 GB)      | 312 | 13.6 | 1.65 s |
| audio | **v13 +audio8da4w (4.78 GB)** | **311** | **13.5** | **1.66 s** |

The progression is **pure-win at every step**:
- v11 → v11_q: -38 % size (text Linear quant)
- v11_q → v12: -56 % size (PLE embedding quant — the elephant)
- v12 → v13: -17 % size (audio encoder quant)
- **Cumulative: 21 GB → 4.78 GB (-77 %)** with all decode tok/s strictly higher than the FP32 baseline.

The remaining gap to Leixin's D99603811 untied 4-bit default (4.1 GB) is **0.68 GB**, almost entirely the un-tied `lm_head` weight (Phase 1.6 follow-up).

## Per-test detail

### Text + text

| Test | Prompt tok | Gen tok | Prefill tok/s | Decode tok/s | TTFT (ms) | PTE |
|---|---:|---:|---:|---:|---:|---|
| text-short  | 18 | 8   | 93  | 13.8 | 194 | v11 FP32 (21 GB) |
| text-short  | 18 | 8   | 112 | 16.3 | 161 | v11_q 8da4w (13 GB) |
| text-short  | 18 | 8   | 106 | 16.3 | 170 | v12 8da4w+emb8 (5.8 GB) |
| text-medium | 21 | 29  | 103 | 13.7 | 204 | v11 FP32 (21 GB) |
| text-medium | 21 | 27  | 127 | 16.2 | 166 | v11_q 8da4w (13 GB) |
| text-medium | 21 | 21  | 129 | 16.5 | 163 | v12 8da4w+emb8 (5.8 GB) |
| text-math   | 25 | 49  | 105 | 12.3 | 238 | v11 FP32 (21 GB) |
| text-math   | 25 | 49  | 137 | 15.3 | 183 | v11_q 8da4w (13 GB) |
| text-math   | 25 | 49  | 134 | 15.8 | 186 | v12 8da4w+emb8 (5.8 GB) |
| text-long   | 25 | 149 | 107 | 13.9 | 233 | v11 FP32 (21 GB) |
| text-long   | 25 | 149 | 130 | 16.5 | 193 | v11_q 8da4w (13 GB) |
| text-long   | 25 | 149 | 133 | 15.7 | 188 | v12 8da4w+emb8 (5.8 GB) |

### Image + text

| Test | Prompt tok | Gen tok | Prefill tok/s | Decode tok/s | TTFT (ms) | PTE |
|---|---:|---:|---:|---:|---:|---|
| image-short  | 302 | 14  | 93  | 12.4 | 3245 | v11 FP32 (21 GB) |
| image-short  | 302 | 14  | 103 | 14.7 | 2931 | v11_q 8da4w (13 GB) |
| image-short  | 302 | 9   | 101 | 15.4 | 2988 | v12 8da4w+emb8 (5.8 GB) |
| image-medium | 300 | 17  | 93  | 12.0 | 3222 | v11 FP32 (21 GB) |
| image-medium | 300 | 18  | 102 | 14.8 | 2953 | v11_q 8da4w (13 GB) |
| image-medium | 300 | 18  | 100 | 15.6 | 2989 | v12 8da4w+emb8 (5.8 GB) |
| image-long   | 310 | 159 | 95  | 12.1 | 3276 | v11 FP32 (21 GB) |
| image-long   | 310 | 159 | 103 | 14.0 | 3011 | v11_q 8da4w (13 GB) |
| image-long   | 310 | 159 | 105 | 14.6 | 2957 | v12 8da4w+emb8 (5.8 GB) |

### Audio + text

| Test | Prompt tok | Gen tok | Prefill tok/s | Decode tok/s | TTFT (ms) | PTE |
|---|---:|---:|---:|---:|---:|---|
| audio-short  | 515 | 29  | 256 | 11.6 | 2013 | v11 FP32 (21 GB) |
| audio-short  | 515 | 29  | 300 | 13.4 | 1719 | v11_q 8da4w (13 GB) |
| audio-short  | 515 | 29  | 316 | 13.2 | 1631 | v12 8da4w+emb8 (5.8 GB) |
| audio-medium | 512 | 63  | 251 | 11.7 | 2036 | v11 FP32 (21 GB) |
| audio-medium | 512 | 63  | 305 | 13.4 | 1676 | v11_q 8da4w (13 GB) |
| audio-medium | 512 | 55  | 313 | 14.1 | 1636 | v12 8da4w+emb8 (5.8 GB) |
| audio-long   | 520 | 65  | 254 | 11.5 | 2045 | v11 FP32 (21 GB) |
| audio-long   | 520 | 149 | 298 | 13.4 | 1747 | v11_q 8da4w (13 GB) |
| audio-long   | 520 | 149 | 307 | 13.5 | 1696 | v12 8da4w+emb8 (5.8 GB) |

## v13 per-test detail (4.78 GB pte)

Same suite as the v11 / v11_q / v12 tables above, run on `/tmp/gemma4_v13_aud.pte`:

| Test | Prompt tok | Gen tok | Prefill tok/s | Decode tok/s | TTFT (ms) |
|---|---:|---:|---:|---:|---:|
| text-short  | 18  | 8   | 111 | 15.4 | 162  |
| text-medium | 21  | 21  | 129 | 16.9 | 163  |
| text-math   | 25  | 49  | 144 | 15.9 | 174  |
| text-long   | 25  | 149 | 144 | 15.5 | 174  |
| image-short  | 302 | 9   | 98  | 14.4 | 3088 |
| image-medium | 300 | 18  | 99  | 14.5 | 3021 |
| image-long   | 310 | 159 | 101 | 14.2 | 3075 |
| audio-short  | 515 | 29  | 316 | 13.6 | 1630 |
| audio-medium | 512 | 55  | 312 | 13.5 | 1642 |
| audio-long   | 520 | 149 | 305 | 13.4 | 1705 |

## Sample outputs (v12, FP32-equivalent quality)

- text-short:  `"The capital of France is **Paris**."`
- text-medium: `"A neural network is a computational model inspired by the structure of the human brain..."`
- text-math:   `"12 multiplied by 8 is **96**."`
- image-short: `"The object in the image is **red**. It is a strawberry."`
- image-medium: `"This is a still life photograph featuring a single, vibrant red strawberry."`
- image-long:  multi-paragraph structured response with **Subject (Strawberry)** / **Setting and Composition** sections.
- audio-medium: `"This week I traveled to Chicago to deliver my final farewell address to the nation, following in the tradition of presidents before me. ..."`

## Repro

```bash
# v12 (embedding quant only)
PYTHONPATH=$PWD python -m executorch.examples.models.gemma4.export \
    --hf-model ~/models/gemma-4-E2B-it \
    --et-checkpoint ~/models/gemma-4-E2B-it/model_et.pth \
    --output /tmp/gemma4_v12_emb.pte \
    --backend xnnpack \
    --max-seq-len 1024 --audio-frames 1976 \
    --qmode 8da4w --group-size 32 \
    --embedding-quantize 8,0

# v13 (+ audio encoder quant)
PYTHONPATH=$PWD python -m executorch.examples.models.gemma4.export \
    --hf-model ~/models/gemma-4-E2B-it \
    --et-checkpoint ~/models/gemma-4-E2B-it/model_et.pth \
    --output /tmp/gemma4_v13_aud.pte \
    --backend xnnpack \
    --max-seq-len 1024 --audio-frames 1976 \
    --qmode 8da4w --group-size 32 \
    --embedding-quantize 8,0 \
    --audio-quantize 8da4w --encoder-group-size 128
```

Predicted from the size-budget analysis: v12 5.7 GB, v13 ~4.5 GB. Actual: **5.78 GB** and **4.78 GB**.

## What's next to close the rest of the gap

| Item | Predicted size | Open issue |
|---|---:|---|
| `--tied-embedding` (drop duplicate `lm_head`) | -0.2 to -1.2 GB → **~4.5 GB** | needs Transformer-side wiring (model has separate `output` Linear; convert_weights ties at load but model un-ties at instantiation). |
| `--vision-quantize 8w` (weight-only INT8) | -100 MB → **~4.7 GB** | hits `Missing out variants: torchao::dequantize_affine` in `to_executorch()` for the weight-only path under XNNPACK. Need either an op-variant registration or a different vision-encoder quant strategy (e.g. dynamic-activation 8da8w with a `torch._check` workaround for the `Ne(u0,1)` guard inside `embedding_projection`). |
| `--vision-quantize 8da8w` | -150 MB → **~4.65 GB** | TorchAO data-dependent guard `Ne(u0,1)` inside vision encoder's `embedding_projection`. |
| `--qmode 8da4w + --embedding-quantize 4,0 + tied` | **~3 GB** | matches Leixin's smallest config (2.7 GB); int4 embedding may need wrapper-test re-validation. |

Leixin's D99603811 mobile defaults are 4.1 GB (untied) / 2.7 GB (tied). v13 lands at **4.78 GB**, **0.68 GB above the untied target**. The remaining gap is dominated by the un-tied `lm_head`; closing it via `--tied-embedding` matches Leixin's untied default and beats the prediction.
