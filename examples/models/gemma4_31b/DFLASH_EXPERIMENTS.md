Written By: Chetan Thotti (cthotti)
Date: 07/24/2026

This is a record of DFlash speculative decoding benchmarking for
Gemma4-31B on a single M4 Pro machine (10-core CPU / 20-core GPU / 64GB
LPDDR5X). The short version: **DFlash gives a real, solid speedup on
math and code prompts (~1.5-1.7x), but is slower than plain baseline
decoding on open-ended chat prompts (~0.77x).** The task-dependence is
the headline finding here, not the raw tok/s numbers -- see the Qwen3
write-up (`../qwen3/DFLASH_EXPERIMENTS.md`) for the hardware-generation
story, which we didn't re-run here (only one machine tested).

## The setup

Target model was the prequantized `SocialLocalMobile/gemma-4-31B-it-HQQ-INT4`
checkpoint (INT4/INT8 mixed, ~21GB, "sensitive" recipe), draft was
`z-lab/gemma-4-31B-it-DFlash` (5 layers, 6 tapped target layers
`[1,12,23,35,46,57]`, block_size 16). Both baseline and DFlash runs reuse
the same target `.pte` (the one with the two-output `(logits, hidden)`
signature) rather than a separately-exported plain baseline model -- same
methodology as the Qwen3 write-up, so the comparison isolates the
algorithm rather than any difference between two export paths.

Getting a correct target export here took two real fixes, not just a
straightforward port from Qwen3:

- The base `Gemma4_31B.forward` (and the MLX-optimized version installed
  by `mlx_source_transformations.py`) is last-token-only by design --
  correct for normal decode, wrong for DFlash verification, which needs
  logits at every position in the drafted block. Using it unmodified
  silently collapsed the exported model back to `(B, 1, V)` with only
  one output, dropping the hidden-state output entirely, no error at any
  stage.
- `mlx_source_transformations.py`'s `_replace_model_forward` does a full
  `types.MethodType` replacement of the top-level forward at export time
  -- it doesn't wrap or call whatever forward the model already had, it
  discards it. So even after fixing `Gemma4_31BWithHidden.forward`
  directly, calling the stock transform function still silently reverted
  the model to last-token-only, single-output behavior. Needed a
  Gemma4-specific `dflash_mlx_source_transformations.py` that reuses the
  per-layer MLX op rewrites (rope/custom_sdpa/kv-cache, which are
  output-shape-agnostic) but installs a DFlash-aware top-level forward
  instead of the stock one.

Both were verified by checking the exported `.pte`'s actual `MethodMeta`
(`num_outputs`, tensor shapes) after export, not just that export
completed without error -- the failure mode here doesn't throw.

## Results

Three prompts (math, code, chat), three trials each, 300 max new tokens,
greedy decoding:

| Category | Baseline tok/s | DFlash tok/s | tau | Speedup |
|----------|---------------|--------------|------|---------|
| Math     | 10.41         | 15.23        | 6.52 | 1.46x   |
| Code     | 10.22         | 17.58        | 7.70 | 1.72x   |
| Chat     | 10.48         | 8.02         | 3.48 | 0.77x   |

(Prompts: math was a word problem requiring algebraic setup and solving;
code was "write binary search + explain time complexity"; chat was
"explain how photosynthesis works in detail". Within-category trials
were tightly consistent -- greedy decoding is deterministic, so this is
mostly a sanity check that nothing was flaky, not real independent
samples.)

## Why math/code work and chat doesn't

Target verification cost (`target_exec`) is close to flat regardless of
context length -- roughly 374ms per round whether verifying a 20-token or
90-token window. That's the expected signature of a memory-bandwidth-
bound model: verifying a block of drafted tokens in one forward pass
costs close to the same as verifying one token, because the dominant
cost is streaming the ~21GB of weights through memory once, not the
compute itself. That flatness is what makes DFlash's speedup possible at
all on this hardware -- the breakeven tau (where DFlash stops being a
net loss) works out to roughly 3.8-3.9 given baseline's ~10.3 tok/s and
target_exec's ~374ms.

Math and code prompts land comfortably above that line (tau 6.5-7.7).
Chat lands just below it (tau 3.48), which is why it's the one category
that's slower than baseline rather than faster. This mirrors the DFlash
paper's own reported numbers almost exactly: their Table 1 shows
MT-Bench (chat/conversational) as the consistently weakest category
across every model they tested, well below math/code, even on much
smaller target models and H200 GPUs. So this isn't a Gemma4-specific
weakness or a bug in our export -- it's the same task-dependent pattern
the paper documents, just reproduced on a bigger target model and
different hardware.

## What we didn't get to

- **Only one machine tested** (M4 Pro). The Qwen3 write-up found DFlash's
  speedup depends heavily on GPU architecture generation (Apple9/M3+
  vs Apple8/M1-M2), not on raw core count or bandwidth. We'd expect the
  same story to hold for Gemma4 -- an M2-class chip likely wouldn't show
  a speedup even on math/code -- but that's inference from the Qwen3
  result, not something confirmed on Gemma4 directly.
- **An incremental-caching optimization for the draft model's context
  K/V was attempted and reverted.** The draft model recomputes
  projections/norm/RoPE over the *entire* accumulated target-hidden
  history every round rather than just the new chunk since the last
  round -- real, measurable waste that gets worse as generations get
  longer. We built and numerically validated (float32-precision-exact)
  an incremental-caching version, but hit a `torch.export` lowering
  failure (`GuardOnDataDependentSymNode` on a dynamic slice of an
  unbacked SymInt) that would have needed a deeper rewrite to
  `mlx::custom_sdpa` to fix properly. Separately, and more importantly:
  the internal DFlash-on-ExecuTorch design doc states the draft cache is
  *intentionally* reset every round ("the draft model re-processes
  context hidden states each round, so rollback is not needed") -- so
  the behavior we were trying to optimize away may be the intended
  reference design, not an oversight. Reverted rather than pushing
  further against that.
- **MLX v0.32.0's small-M GEMM improvements** (flagged in the Qwen3 PR
  review) were never tested against Gemma4 either -- this repo is still
  pinned to v0.31.1.
