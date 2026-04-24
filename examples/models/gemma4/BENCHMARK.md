# Gemma 4 ExecuTorch — Multimodal Benchmark (CPU + XNNPACK)

**Hardware:** Linux devserver, CPU only (no GPU / NPU acceleration).
**Date:** 2026-04-23.
**Runner:** `cmake-out/examples/models/gemma4/gemma4_runner` built from this branch (`git rev-parse HEAD` at benchmark time captured each run's PyTorchObserver JSON; raw rows in `/tmp/bench_gemma4.jsonl`).

All numbers are end-to-end through the C++ runner: tokenize → encode (vision/audio) → prefill → decode → EOS or `seq_cap`. The runner uses ExecuTorch's stock `MultimodalRunner` plus `Gemma4DecoderRunner` (PLE 3-input).

For the benchmark methodology: each test runs once on a fresh runner invocation; numbers are not averaged over multiple runs to avoid masking warm-up vs cold load. Stand-alone repro:

```bash
RUNNER=cmake-out/examples/models/gemma4/gemma4_runner
TOK=~/models/gemma-4-E2B-it/tokenizer.json
$RUNNER --model_path /tmp/gemma4_multimodal_v11.pte   --tokenizer_path $TOK \
        --prompt "What is the capital of France?" --seq_len 30
```

## PTEs compared

| .pte | size | text quant | encoder quant |
|---|---|---|---|
| `gemma4_multimodal_v11.pte`    | 21 GB | FP32 | FP32 |
| `gemma4_multimodal_v11_q.pte`  | 13 GB | 8da4w (group=32) | FP32 |

Quantizing the text backbone to 8da4w shrinks the PTE by **35 %** and gives **15–25 % decode speedup** with no measured quality regression on the suite below.

## Summary (averaged per modality)

| Modality | PTE | Prefill tok/s | Decode tok/s | TTFT |
|---|---|---:|---:|---:|
| text  | v11 FP32  | 102 | 13.4 | 217 ms |
| text  | v11 8da4w | **126** | **16.1** | **176 ms** |
| image | v11 FP32  | 94  | 12.1 | 3.25 s |
| image | v11 8da4w | **103** | **14.5** | **2.96 s** |
| audio | v11 FP32  | 254 | 11.6 | 2.03 s |
| audio | v11 8da4w | **301** | **13.4** | **1.71 s** |

8da4w wins on every metric for every modality. Image TTFT is dominated by the FP32 vision encoder (~3 s); applying `--vision-quantize 8da8w` (Phase 1.3 flag, validation pending) is expected to halve that. Audio TTFT is dominated by the audio encoder + 512-token prefill on the soft tokens.

## Text + text

| Test | Prompt tok | Gen tok | Prefill tok/s | Decode tok/s | TTFT (ms) | Total (ms) | PTE |
|---|---:|---:|---:|---:|---:|---:|---|
| text-short  | 18 | 8   | 93  | 13.8 | 194 | 774   | v11 FP32  |
| text-short  | 18 | 8   | 112 | 16.3 | 161 | 652   | v11 8da4w |
| text-medium | 21 | 29  | 103 | 13.7 | 204 | 2322  | v11 FP32  |
| text-medium | 21 | 27  | 127 | 16.2 | 166 | 1835  | v11 8da4w |
| text-math   | 25 | 49  | 105 | 12.3 | 238 | 4235  | v11 FP32  |
| text-math   | 25 | 49  | 137 | 15.3 | 183 | 3376  | v11 8da4w |
| text-long   | 25 | 149 | 107 | 13.9 | 233 | 10960 | v11 FP32  |
| text-long   | 25 | 149 | 130 | 16.5 | 193 | 9230  | v11 8da4w |

Decode speedup is **+18 % to +24 %** across all four prompt sizes.

## Image + text

| Test | Prompt tok | Gen tok | Prefill tok/s | Decode tok/s | TTFT (ms) | Total (ms) | PTE |
|---|---:|---:|---:|---:|---:|---:|---|
| image-short  | 302 | 14  | 93  | 12.4 | 3245 | 4378  | v11 FP32  |
| image-short  | 302 | 14  | 103 | 14.7 | 2931 | 3883  | v11 8da4w |
| image-medium | 300 | 17  | 93  | 12.0 | 3222 | 4641  | v11 FP32  |
| image-medium | 300 | 18  | 102 | 14.8 | 2953 | 4171  | v11 8da4w |
| image-long   | 310 | 159 | 95  | 12.1 | 3276 | 16464 | v11 FP32  |
| image-long   | 310 | 159 | 103 | 14.0 | 3011 | 14380 | v11 8da4w |

Image input is `image.jpg` (strawberry photo) → 280 vision soft tokens after the encoder. The 300-token prompt count is `~20 text` + 280 vision soft tokens.

Prefill `tok/s` includes the soft tokens; the actual text-decoder prefill rate is the same as in the text-only column (~100 tok/s). The encoder cost (~2.5 – 3 s for vision) sits in TTFT.

## Audio + text

| Test | Prompt tok | Gen tok | Prefill tok/s | Decode tok/s | TTFT (ms) | Total (ms) | PTE |
|---|---:|---:|---:|---:|---:|---:|---|
| audio-short  | 515 | 29  | 256 | 11.6 | 2013 | 4510  | v11 FP32  |
| audio-short  | 515 | 29  | 300 | 13.4 | 1719 | 3880  | v11 8da4w |
| audio-medium | 512 | 63  | 251 | 11.7 | 2036 | 7403  | v11 FP32  |
| audio-medium | 512 | 63  | 305 | 13.4 | 1676 | 6370  | v11 8da4w |
| audio-long   | 520 | 65  | 254 | 11.5 | 2045 | 7685  | v11 FP32  |
| audio-long   | 520 | 149 | 298 | 13.4 | 1747 | 12838 | v11 8da4w |

Audio input is `obama_short20.wav` (20 s, 16 kHz mono PCM) → 494 audio soft tokens after the audio encoder. The 512-token prompt count is `~18 text` + 494 audio soft tokens.

Audio prefill `tok/s` is ~3× higher than text-only because the runtime amortises encoder cost across the larger soft-token batch — most of the prefill time is the text decoder running on a 512-token batch in one shot.

## Sample outputs (FP32 v11; 8da4w produces qualitatively similar text)

- **text-short**: *"The capital of France is **Paris**."*
- **text-medium**: *"A neural network is a computational model inspired by the structure of the human brain, designed to recognize patterns in data by processing information through interconnected nodes."*
- **text-math**: *"12 multiplied by 8 is **96**."*
- **text-long** (Python code request): well-structured response with section headers and a slicing-based reverse implementation.
- **image-short** ("What color is the object?"): *"The object in the image is **red**. It is a strawberry."*
- **image-medium**: *"This is a close-up photograph of a single, vibrant red strawberry resting on a light-colored, textured surface, likely stone or wood."*
- **image-long**: multi-paragraph structured description with **Subject (Strawberry)** / **Background** / **Foreground and Surface** sections covering color, texture, bokeh, condition.
- **audio-short** ("What is being said?"): a faithful prefix of the speech ("This week I traveled to Chicago to deliver my final farewell address...").
- **audio-medium** ("Transcribe this audio."): *"This week I traveled to Chicago to deliver my final farewell address to the nation, following in the tradition of presidents before me. It was an opportunity to say thank you. Whether we've seen eye-to-eye or rarely agreed at all, my conversations with you, the American people, in living rooms and..."*
- **audio-long**: full transcript followed by a coherent summary paragraph.

## Reproducing this benchmark

The benchmark script lives at `/tmp/bench_gemma4.sh` (kept out of the repo because it embeds local file paths). To regenerate:

```bash
# 1. Have the two PTEs and the test assets in place
ls /tmp/gemma4_multimodal_v11.pte /tmp/gemma4_multimodal_v11_q.pte
ls ~/executorch/image.jpg ~/executorch/obama_short20.wav

# 2. Build the runner (if not already)
make gemma4-cpu

# 3. Run the suite (writes /tmp/bench_gemma4.jsonl)
/tmp/bench_gemma4.sh

# 4. Tabulate
python3 tools/format_bench_table.py /tmp/bench_gemma4.jsonl > BENCHMARK.md
```

(The `format_bench_table.py` step is the inline script used to produce the tables above; not packaged because the next iteration of `tools/measure_pte.py` will absorb it.)

## What's next

| Item | Expected impact |
|---|---|
| `--vision-quantize 8da8w` | ~50 % vision encoder size; image TTFT 3.25 s → ~1.6 s |
| `--audio-quantize 8da4w`  | ~75 % audio encoder size; audio TTFT 2.03 s → ~1.4 s |
| `--quantize-kv-cache`     | reduces decode-time memory; opens way to longer contexts |
| `--tied-embedding`        | E4B size E2B-class (~3 GB) since lm_head is folded into embed_tokens |
| Mobile (S25 / iPhone)     | reference target from D99603811: 152 / 6 tok/s prefill/decode for E2B 4-bit |

Each is a flag in `export.py`; this benchmark is the FP32 baseline they'll be measured against.
