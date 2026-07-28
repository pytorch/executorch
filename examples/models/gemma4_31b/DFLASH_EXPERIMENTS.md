# DFlash Experiments (Gemma4-31B)

**Author:** Chetan Thotti (cthotti)  
**Date:** 07/26/2026

This document summarizes DFlash benchmarking for Gemma4-31B on an M4 Pro (10-core CPU, 20-core GPU, 64GB). DFlash improved math and code generation by **1.5–1.7×**, but was slower on open-ended chat (**0.77×**), highlighting that its effectiveness is highly task-dependent.

## Setup

- **Target:** `SocialLocalMobile/gemma-4-31B-it-HQQ-INT4`
- **Draft:** `z-lab/gemma-4-31B-it-DFlash`
- 6 tapped target layers: `[1,12,23,35,46,57]`
- Block size: `16`
- Baseline and DFlash share the same exported target model.

A Gemma4-specific export path was required because the default MLX export only produced last-token logits, while DFlash needs full-sequence logits and hidden states. Export correctness was verified by inspecting the exported model metadata.

## Results

Three prompts (math, code, chat), three trials each, greedy decoding, 300 max new tokens:

| Category | Baseline | DFlash | τ    | Speedup   |
|----------|----------|--------|------|-----------|
| Math     | 10.41    | 15.23  | 6.52 | **1.46×** |
| Code     | 10.22    | 17.58  | 7.70 | **1.72×** |
| Chat     | 10.48    | 8.02   | 3.48 | **0.77×** |

## Observations

Target verification latency stayed nearly constant across context lengths, indicating a memory-bandwidth-bound workload. Math and code maintained high acceptance rates (τ ≈ 6–8), producing a speedup, while chat averaged τ ≈ 3.5 and fell below the break-even point, matching the task-dependent behavior reported in the DFlash paper.

## Future Work

- Revisit draft-model KV caching if future `torch.export` support improves.