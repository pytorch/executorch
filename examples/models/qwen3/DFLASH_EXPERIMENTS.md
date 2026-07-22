Written By: Chetan Thotti (cthotti)
Date: 08/16/2026

This is a record of the benchmarking we did on DFlash speculative decoding
for Qwen3-4B, across three Apple Silicon machines. The short
version: **DFlash's speedup depends on GPU architecture generation, not on
how big or fast the chip otherwise is.** A base M4 clearly beats an M2 Pro
here, even though the M2 Pro has more GPU cores and more memory bandwidth.
If you're benchmarking DFlash on new hardware, read this first so you
don't waste time on a chip that was never going to show a speedup.

## The setup

Model was `Qwen/Qwen3-4B` with the `z-lab/Qwen3-4B-DFlash-b16` draft
checkpoint, exported with:
--dflash-layers 1,9,17,25,33 --qlinear 4w --qembedding 4w --use-custom-sdpa --use-custom-kv-cache

We tested three chips: the M2 in a MacBook Air (8 GPU cores), an M2 Pro
rental (16 GPU cores), and a base M4 rental (10 GPU cores). Same `.pte`
files got copied across the M2 Pro and M4 runs rather than re-exported
separately, so any difference we saw was purely hardware, not export
drift.

## What we expected vs. what we found

Going in, the assumption was that a "bigger" chip -- more GPU cores, more
memory bandwidth -- would just be faster across the board, M2 Pro
included. That's not what happened. Baseline (plain, one-token-at-a-time)
decoding did scale the way you'd expect: M2 Pro's extra bandwidth made it
faster than the M4 at baseline decoding, in every category we tested.
But DFlash flipped that around entirely, only the M4 ever beat its own
baseline. The M2 Pro was slower with DFlash turned on than without it,
every single time.

| Chip | Category | Baseline tok/s | DFlash tok/s | Speedup |
|--------|------|-------|-------|-------|
| M2 Air | Math | 25.65 | 19.44 | 0.76x |
| M2 Air | Code | 28.19 | 23.81 | 0.84x |
| M2 Air | Chat | 26.11 | 14.77 | 0.57x |
| M2 Pro | Math | 48.30 | 42.63 | 0.88x |
| M2 Pro | Code | 51.65 | 42.84 | 0.83x |
| M2 Pro | Chat | 51.29 | 18.22 | 0.36x |
| M4     | Math | 31.71 | 51.44 | 1.62x |
| M4     | Code | 31.33 | 53.39 | 1.70x |
| M4     | Chat | 31.42 | 22.73 | 0.72x |

(Math/Code/Chat here are three different prompts, run 3 times each and
averaged.)

## The main difference between M2 and M4

The performance difference comes down to GPU architecture, not the CPU.
While I initially suspected SME2, the MLX backend doesn't use it. Instead,
M3/M4 GPUs (Apple9) introduced Dynamic Caching and an improved SIMD matrix-
multiply pipeline, which are much better suited for DFlash's verification
stage. Dflash verifies an entire block of tokens at once using large matrix-
matrix operations, allowing the M4 to execute this workload far more efficiently
than the M2's Apple8 GPU, which lacks these architectural improvements. 

**Practical takeaway: DFlash is worth using on M3 or M4 generation Macs,
any tier, but not on an M1/M2.**

## The draft model is also just bad at chat

Separately from all the hardware stuff: math and code prompts got a tau
(average tokens accepted per speculative round) around 6.6-6.8 on both
chips. Chat-style prompts landed around 2.9 -- consistently, across three
different chat prompts we tried, not just one unlucky example. Since tau
was identical across both chips for the same prompt, this isn't a
hardware issue at all, it's the draft model itself being noticeably
worse at predicting open-ended conversational text than it is at
structured math or code. Worth knowing if you're deciding whether DFlash
is worth turning on for a particular kind of workload, independent of
what hardware you're running it on.

## What we didn't get to

- **8-bit target quantization** (`--qlinear 8w`) fails to export with
  `RuntimeError: Missing out variants: {'torchao::dequantize_affine'}`.
  That's a real gap in the MLX partitioner's op coverage, not a flag
  mistake.
- **M3-generation chips** were never tested directly. Based on the
  mechanism above they should behave like the M4 (same Apple9 GPU
  family), but that's inference, not something we confirmed ourselves.
