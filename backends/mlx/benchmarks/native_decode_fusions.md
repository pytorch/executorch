# Native MLX decode fusion measurements

This draft preserves opt-in FP16 SwiGLU/sigmoid, RoPE, RMSNorm/residual and logit
soft-cap fusions, plus sharing of identical RoPE frequency buffers during Muse
Glimmer MLX export. Packed Q4_K/Q5_K/Q6_K weights and native quantized matmuls
are unchanged. Activations remain FP16; rotation and logit soft-cap arithmetic
preserve the existing FP32 intermediates and rounding boundaries.

Enable during export with `ET_MLX_FUSE_GATES=1 ET_MLX_FUSE_ROPE=1
ET_MLX_FUSE_NORM_ADD=1 ET_MLX_FUSE_SOFT_CAP=1 ET_MLX_SHARE_ROPE_PHASE=1`.
All flags default off. The runtime also caches custom-kernel callables per loaded
program and accepts zero-byte outputs. The phase-sharing hook is MLX-only.

## Serving protocol

Measured November 20–21, 2026 on M5 Max, 64 GiB, macOS 27.2, MLX
`14dc9c673bce9a51dca4e58f5b2ccae80736431d`, ExecuTorch parent
`f9e14fe3817871f34ec8bb6f70b63ecdb112c143` plus the local fusion changes.
Models: Muse-Glimmer-30B-KQuant-17GB-Q4_K_M target, original DFlash Q4_K_M,
and Muse-Glimmer-30B-DFlash2-Q4_K_M.

One active HTTP request/session; six prompts (four short, plus 512/2048 tokens),
64 generated tokens, seed 42, top-k 0, top-p 1, ignore-EOS, visible reasoning,
no prefix reuse. Two counterbalanced baseline/fused blocks with reversed
temperature ordering and excluded warmups. Model loading and cooling are excluded.

Effective decode is pooled `(completion_tokens - 1) / server decode seconds`.
HTTP throughput is pooled `completion_tokens / full client request seconds`,
including prefill. GPU telemetry is sampled every 200 ms. Retained pairs require
active-time-weighted frequencies within 1%, nominal thermals, AC power,
low-power mode off, normal memory pressure and at least 95% telemetry coverage.
Batches start after two seconds with nominal thermals, GPU <=60 C and CPU <=70 C.
These are request-level checks, not hardware frequency locks or separate phase
clock controls. Different rows retain different prompt cohorts; do not infer an
optimal proposal count from this table.

## RoPE, norm/add, soft cap and shared phases

The comparator already has gate fusion enabled. The following gains are
incremental to that build. 192 measured requests and 192 warmups completed;
96/96 paired visible outputs match. 80/96 pairs pass the predeclared gates.
The solo configuration disables speculation using the same target method from
the draft-capable PTE; it still computes feature taps.

| Mode | Proposals | Temperature | Retained pairs | Decode tok/s baseline → fused | Decode gain | Prefill gain | HTTP gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| solo | 0 | 0 | 7/12 | 26.36 → 26.91 | +2.10% | -1.27% | +1.60% |
| solo | 0 | 0.8 | 7/12 | 26.01 → 26.69 | +2.59% | +0.51% | +2.11% |
| dflash | 7 | 0 | 12/12 | 89.96 → 91.71 | +1.95% | +1.92% | +1.91% |
| dflash | 7 | 0.8 | 11/12 | 78.41 → 79.26 | +1.09% | +1.92% | +1.45% |
| dflash2 | 7 | 0 | 11/12 | 99.64 → 101.70 | +2.06% | +1.32% | +1.67% |
| dflash2 | 7 | 0.8 | 11/12 | 88.47 → 90.79 | +2.62% | +1.53% | +2.12% |
| dflash2 | 15 | 0 | 10/12 | 123.51 → 124.85 | +1.08% | +1.43% | +1.28% |
| dflash2 | 15 | 0.8 | 11/12 | 103.82 → 106.64 | +2.72% | +1.41% | +2.06% |

Both retained run-order blocks have positive aggregate decode and HTTP gains
for every configuration, with the following variation (not confidence intervals).
Only two repetitions per prompt were measured.

| Mode | Proposals | Temperature | Decode gain across blocks |
|---|---:|---:|---:|
| solo | 0 | 0 | +0.98% to +3.61% |
| solo | 0 | 0.8 | +1.21% to +3.63% |
| dflash | 7 | 0 | +1.37% to +2.53% |
| dflash | 7 | 0.8 | +0.55% to +1.58% |
| dflash2 | 7 | 0 | +1.68% to +2.50% |
| dflash2 | 7 | 0.8 | +1.56% to +3.80% |
| dflash2 | 15 | 0 | +0.54% to +1.64% |
| dflash2 | 15 | 0.8 | +2.10% to +3.28% |

Prefill is mixed for solo. Greedy 512-token solo has no retained pair; sampled
512-token solo has one pair with decode -0.25% and HTTP -0.28%. Short-prompt
prefill includes -6.55% for sampled solo. Small aggregate gains do not imply every
workload improves. Machine-readable aggregates, counts and per-block gains are
in [native_decode_fusions.json](native_decode_fusions.json).

## Earlier gate-only experiment

This separate run compares SwiGLU/sigmoid fusion with the unfused native path.
144 measured requests and 144 warmups completed; 72/72 paired outputs match,
63/72 pairs pass the same gates. Do not add or compound these percentages with
the later experiment: their retained cohorts and run conditions differ.

| Mode | Proposals | Temperature | Matched pairs | Native decode tok/s | Fused decode tok/s | Decode change | Prefill change | HTTP throughput change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dflash | 7 | 0 | 11/12 | 86.45 | 87.55 | +1.27% | +0.79% | +1.08% |
| dflash | 7 | 0.8 | 11/12 | 76.38 | 77.33 | +1.24% | +1.17% | +1.26% |
| dflash2 | 7 | 0 | 11/12 | 97.74 | 97.83 | +0.09% | +1.77% | +0.88% |
| dflash2 | 7 | 0.8 | 10/12 | 83.49 | 83.79 | +0.36% | +1.49% | +0.88% |
| dflash2 | 15 | 0 | 10/12 | 117.27 | 121.55 | +3.64% | +1.84% | +2.61% |
| dflash2 | 15 | 0.8 | 10/12 | 93.14 | 95.63 | +2.67% | +1.51% | +2.07% |

DFlash2 p7 greedy was effectively unchanged (block gains -0.22% to +0.37%).
DFlash2 p15 improved in both blocks (greedy +3.45–3.84%, sampled +2.16–3.18%).

## Isolated kernel measurements

GPU command-buffer times measure only each operator stage, not quantized matmul
or whole-model latency. Two counterbalanced blocks of 1500 trials per arm,
warmed buffers, identical 256 KiB arithmetic conditioning, exact-byte output
checks. Reported values average block medians in the predeclared 1620 MHz
+/-0.5% subset with nominal thermals, AC and low-power mode off. Every arm/block
retains at least 500 trials and 25% of trials; some whole-window clock gates fail.

| Operator | M | Heads/layout | Native us | Fused us | Speedup |
|---|---:|---|---:|---:|---:|
| rotary | 1 | 32 interleaved | 10.125 | 2.812 | 3.60x |
| rotary | 1 | 2 interleaved | 9.688 | 2.833 | 3.42x |
| rotary | 1 | 32 half-split | 10.000 | 2.875 | 3.48x |
| rotary | 8 | 32 interleaved | 10.208 | 2.958 | 3.45x |
| rotary | 16 | 32 interleaved | 11.646 | 3.000 | 3.88x |
| rotary | 512 | 32 interleaved | 119.687 | 16.229 | 7.37x |
| norm | 1 | — | 4.833 | 4.875 | 0.99x |
| norm | 8 | — | 5.083 | 5.000 | 1.02x |
| norm | 16 | — | 5.125 | 5.125 | 1.00x |
| norm | 512 | — | 36.313 | 28.938 | 1.25x |
| soft_cap | 1 | — | 12.146 | 4.229 | 2.87x |
| soft_cap | 8 | — | 50.000 | 18.000 | 2.78x |
| soft_cap | 16 | — | 120.813 | 34.000 | 3.55x |

RoPE head dimension 128, norm hidden dimension 6656, soft-cap vocabulary 202048.
Standalone Q/K normalization remains unchanged. Shared phase computation is not
included in isolated RoPE timing. Norm/add is effectively flat for M1/8/16.

Earlier SwiGLU measurements (K=19968):

| M | Native µs | Fused µs | Stage speedup |
|---:|---:|---:|---:|
| 8 | 8.333 | 3.750 | 2.22x |
| 128 | 45.083 | 26.208 | 1.72x |
| 512 | 216.938 | 105.135 | 2.06x |

SwiGLU M8 fails the separate whole-window clock gate (1.09% difference), while
its predeclared 1620 MHz subset passes; M128/M512 pass both gates.

## Correctness and scope

25 gate and 44 further exported operator cases pass exact-byte comparisons
against native MLX through the production passes and C++ runner. Coverage
includes empty/dynamic dimensions, strided layouts, extreme finite FP16 values,
large/reset RoPE positions, shared intermediates and BF16 fallback. Existing
offgraph tests: nine passed, one skipped, five subtests passed.

For each experiment, 18 full-vocabulary logit comparisons and 36 speculative
token/acceptance comparisons match exactly at temperatures 0/.8 for original
DFlash p7 and DFlash2 p7/p15. The final target has 78 rotary, 104 norm/add,
one soft-cap and 104 gate kernels; sin/cos pairs decrease from 39 to one. Each
draft has ten rotary, five gate kernels and one sin/cos pair, previously five.
DFlash2 draft soft cap keeps its existing path because the conservative matcher
requires identical cap nodes. Every retained constant is checked bytewise;
only duplicate frequency buffers are removed. Native quantized-matmul and GGUF
fallback counts are unchanged; embedding delegates are byte-identical.

These are saved measurements from the experimental build, not a claim that the
entire draft versus its parent was benchmarked in one combined run. The shared
runtime fixes were present in both measured arms. The isolated draft excludes
activation-quantization experiments, architecture dispatch and their schema,
GGUF, submodule and CI changes. The four Metal fusion bodies are unchanged from
the measured versions; the gate matcher drops its now-unneeded experiment guard.
Full models, raw request records and telemetry remain in the local experiment
artifacts; this draft preserves their result summaries.
