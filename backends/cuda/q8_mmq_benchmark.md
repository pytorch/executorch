# Q8 MMQ prefill: benchmark and optimization handoff

## Code and benchmark provenance

This standalone draft starts from ExecuTorch main `0eba76c3a844c3818a890eb9fdea5c6bbced3bf6`. The CUDA MMQ numerical kernels and launcher have the same Python AST as the optimized implementation used below. Autotune support was isolated into `triton/kernels/q8_mmq_autotune.py` because the experiment used compile helpers from an unmerged PR stack. Current main's existing small-row kernels and fallback policy are retained; in particular, this draft does not require the Q5 M8–M64 changes from #23800.

The performance and full-model output results below were measured on the earlier benchmark checkout: PR #23740 stack plus #23727, with the prior Q5/DFlash fixes. They are not a new main-only full-model benchmark. The three ET reference exports were rerun in the same experiment. The standalone main port is validated separately before publication; its validation log is retained in the workspace.

Experimental source base HEAD: `4c451dc75faa5920bb08791a7877db5de8d15a0a`; the checkout also had the documented benchmark patches applied. Full source hashes and export hashes remain in `results/q8-mmq-opt` and the local reproduction bundle.

## Arithmetic and pipeline

BF16 activations are quantized to signed INT8 with one FP32 scale per K256 and INT32 sums per K32. A preparation kernel expands packed Q4/Q5/Q6 and integer scale codes into low/high signed INT8 byte matrices. The two byte matrices exactly reconstruct the centered integer weight value, without an extra weight quantization. Q4/Q5 additionally prepare a BF16 residual with explicit FMA. GEMM uses two INT8 dots, the Q4/Q5 residual BF16 dot, and FP32 superblock accumulation before BF16 output.

The prepared buffers are transient per linear call; they are not persistent copies of all model weights. Source allocation is `2*N*K + 2*N*(K/32)` bytes. The combined FFN gate/up shape N39936, K6656 uses about 523 MiB. Observed whole-process peak VRAM increased 256–576 MiB versus prior MMQ.

Buckets 128,256,512,1024,2048,4096 serve large-row prefill. Inputs provably within M<=64 retain main's prior dispatch policy. A dynamic large bucket still masks runtime rows, including small runtime M values. Unsupported inputs use the existing dequant + linear fallback.

## Solo prefill

RTX 5090 GPU0, Muse Glimmer 30B Q4_K_M. Seven fresh-process runs per engine/tier; discard runs 1–2 and retain 3–7. Same runner, exact shared input tokens, greedy temperature=0, seed=1, 512 forced output tokens, context131072, prefill chunk4096, CUDA graphs, no cross-request prefix reuse. Engines alternate order; all GPU measurements run serially. Total:105 runs. Units:tokens/s, mean ± sample standard deviation.

| Input tokens | Original BF16 | Prior MMQ | Optimized MMQ | vs prior MMQ | vs BF16 |
|---:|---:|---:|---:|---:|---:|
| 512 | 2311.0 ± 40.3 | 1883.3 ± 45.6 | 2188.5 ± 78.5 | +16.2% | -5.3% |
| 2,048 | 3639.1 ± 21.3 | 2485.5 ± 13.2 | 3239.5 ± 16.0 | +30.3% | -11.0% |
| 8,192 | 3781.8 ± 20.6 | 2497.0 ± 2.5 | 3381.8 ± 5.8 | +35.4% | -10.6% |
| 32,768 | 3597.6 ± 1.8 | 2403.7 ± 1.5 | 3230.1 ± 2.4 | +34.4% | -10.2% |
| 130,048 | 3009.6 ± 0.9 | 2123.9 ± 0.5 | 2745.3 ± 1.0 | +29.3% | -8.8% |

The optimized version improves prior MMQ by 16–35%, but remains 5–11% below the original dequant+BF16 path. There is no new llama.cpp comparison in this experiment. M1 decode takes the existing kernel path and remains approximately unchanged.

| Input tokens | BF16 decode | Prior MMQ decode | Optimized decode | BF16 peak MiB | Prior peak MiB | Optimized peak MiB |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 83.08 | 82.93 | 83.02 | 19018 | 18538 | 19076 |
| 2048 | 82.58 | 82.49 | 82.64 | 19274 | 18980 | 19338 |
| 8192 | 81.73 | 81.76 | 81.71 | 19837 | 19594 | 20010 |
| 32768 | 81.45 | 81.18 | 81.32 | 20042 | 19754 | 20330 |
| 130048 | 76.03 | 75.92 | 76.04 | 21418 | 21354 | 21610 |

Peak VRAM is the mean of the retained process peaks, sampled externally every 200 ms; instantaneous peaks may be missed.

## Profile and remaining work

M4096,N39936,K6656 complete FFN gate/up linear: prior 16.279 ms, optimized 11.636 ms, original BF16 9.915 ms. This includes activation quantization and weight preparation.

Nsight Systems captures the full process: warmup, 8192-token prefill, one output token. These are accumulated GPU kernel milliseconds; they are not the independently measured prefill throughput.

| Engine | MMQ GEMM ms | Weight preparation ms | Activation quantization ms | All kernels ms |
|---|---:|---:|---:|---:|
| et-baseline | 0.00 | 0.00 | 0.00 | 2205.01 |
| et-q8-prior | 3033.27 | 0.00 | 32.57 | 3332.60 |
| et-q8-optimized | 2051.16 | 117.22 | 32.27 | 2462.24 |

GEMM remains 83.3% of optimized total kernel time; preparation 4.8%, activation quantization 1.3%. The main FFN choice is BM128, BN64, BK256, 8 warps, 2 stages. Runtime launch records show 255 registers/thread and 64–84 KiB shared memory for MMQ. These are resource counts, not a measured occupancy or Tensor Core utilization claim.

The next agent should prioritize the GEMM integer representation and data pipeline: the exact low/high representation requires two INT8 dots, plus Q4/Q5 BF16 compensation and K256 scaling. Investigate a packed-weight CUDA/CUTLASS implementation that reduces extra dot work while preserving group scales and centering, and improves asynchronous data movement. Do not assume that improving the small activation quantizer will close the remaining BF16 gap.

Exploratory alternatives already tried include K32 groupwise dots, native warp INT8 MMA variants, raw-code preparation, and Gluon implementations. None beat the selected prepared low/high implementation in the hotspot experiment. One register-prefetch prototype failed output validation and was discarded. The attempted scripts and logs are preserved in the original workspace.

## Numerical and generation validation

Standalone main port: the 126-check CUDA/dispatch/export/legality suite passed 124 checks initially. Two tests still expected M65 to fall back; they were updated to the new M4097 fallback boundary and both passed on targeted rerun. Numerical kernels and launcher remain unchanged. Pinned ufmt/flake8 and git diff --check passed.

The original optimized source passed 16 CUDA tests, including an independent FP64 FMA residual regression, Q4/Q5/Q6 synthetic arithmetic and strict dynamic export/compile with runtime M5,31,65,129,4096. Three real GGUF linear checks against prior MMQ had relative RMS<1e-4; floating point output is not bit-exact.

Full-model teacher forcing sampled 64 positions for each of four inputs. Optimized top-1 agreed 256/256 with both original BF16 and prior MMQ. Relative logits RMS versus BF16 ranges 1.74–5.39%, versus prior MMQ 1.38–5.61%. The initial 0.02 RMS gate failed on long text; the final gate is the prior experiment's finite-output and RMS<0.10 criterion. Passing sampled top-1 does not establish general output equivalence.

| Generation check | Reference | Complete token sequences identical | First token identical |
|---|---|---:|---:|
| 35 repetitive-input runs, 512 output tokens | et-baseline | 35/35 | 35/35 |
| 7 natural tasks, greedy, EOS, max1024 | et-baseline | 3/7 | 7/7 |
| 35 repetitive-input runs, 512 output tokens | et-q8-prior | 35/35 | 35/35 |
| 7 natural tasks, greedy, EOS, max1024 | et-q8-prior | 0/7 | 7/7 |

All three generated prime-count functions passed seven inputs from 0 to 10000; BF16 produced a sieve, MMQ trial division. All math answers correctly compute 60+80×2=220 miles. Exercise JSON lists and name/date/amount extraction match after parsing. All answers were read; wording differs in the natural tasks, and no general quality score is claimed.

Explicit FMA was added after PTX inspection found that a separated multiply/subtract in the first preparation implementation changed residual rounding. The pre-FMA export and partial benchmark were archived, and all 105 final runs were restarted after re-export. FMA restores that residual rounding rule, but does not remove all GEMM floating point differences.

## Workspace handoff and reproduction

Environment: `/venv/executorch/bin/python`; versions `2.14.0+cu132 13.2 3.8.0` (Torch, CUDA runtime, Triton).

Original benchmark checkout: `/workspace/benchmark-5090/q8-mmq/executorch`; standalone main validation checkout: `/workspace/benchmark-5090/mmq-draft-pr/executorch`. Preserve the original checkout and exports for controls.

Original BF16 export: `/workspace/benchmark-5090/exports/pr23727/solo`; prior MMQ: `/workspace/benchmark-5090/exports/q8-mmq/solo`; optimized: `/workspace/benchmark-5090/exports/q8-mmq-opt/solo`. Use `/workspace/benchmark-5090/build-q8-mmq-controls/solo_runner`. Any new kernel revision requires a new export/compiler cache before timing it.

The complete evidence is under `/workspace/benchmark-5090/results/q8-mmq-opt`; experiment scripts are under `/workspace/benchmark-5090/jobs`. Local complete report, raw outputs, source snapshots and reproduction archive are in this chat's `outputs/q8-mmq-opt` and `outputs/q8-mmq-optimization-reproduction.tar.gz`. Large PTE/PTD/NSYS artifacts remain remote; their hashes are recorded.

```bash
cd /workspace/benchmark-5090/mmq-draft-pr/executorch
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD/src" /venv/executorch/bin/python -m pytest -q backends/cuda/tests/test_q8_mmq.py
```

This draft and its report were prepared with OpenAI Codex.
