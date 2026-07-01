# INT5 Pipeline (GGUF Q5_K → CUDA W5A8 dp4a) — Implementation Notes

Genuine 5-bit weight pipeline in `~/executorch-2/executorch` (branch
`cuda-int4-int6-metadata-opt`), mirroring the existing INT4 (asymmetric zero) and
INT6 (bit-plane split) paths and reusing the branch's compact int8-encoded
scale/zero metadata scheme (uint8 codes + per-row bf16 super-scale).

**Scope: generic ET-level INT5 support only.** This migration adds the reusable
GGUF Q5_K decode + CUDA int5 backend to ExecuTorch. No model-specific (e.g. guac)
content is migrated or wired here. A Q5_K weight (such as a Q5_K lm_head) routes
through the shared CUDA packer to a genuine `CudaDp4aPlanarInt5Tensor` — no lossy
Q5_K→bf16→INT4 round-trip. Validation reads the guac `.gguf`'s `output.weight`
purely as a Q5_K *data sample* (it is the only real Q5_K tensor on hand); no guac
model code is imported.

## Files added

| File | Purpose |
| --- | --- |
| `backends/cuda/dp4a_planar_int5_tensor.py` | `CudaDp4aPlanarInt5Tensor` subclass + `pack_int5` / `unpack_int5` + `_encode_uint8_per_row`. Owns the 5-bit ql/qh pack and metadata re-encode. |
| `backends/cuda/quantize_op_dispatch/int5_dispatch.py` | Defines the `executorch_cuda::int5_plain_mm` custom op (Meta + CUDA impls) and the `F.linear` / `aten.linear` dispatch on the subclass. Decode (M≤4) → custom op; prefill (M>4) → inline dequant + `F.linear`. |
| `backends/cuda/runtime/shims/int5_plain_mm.cuh` | W5A8 dp4a matvec kernel + activation int8 quant kernel (`_int5_plain_mm_cuda` entry point). |
| `backends/cuda/runtime/shims/int5_plain_mm.cu` | `extern "C" aoti_torch_cuda_int5_plain_mm` AOTI C shim. |
| `backends/cuda/runtime/shims/int5_plain_mm.h` | Shim header / ABI declaration. |

## Files changed

| File | Change |
| --- | --- |
| `extension/llm/export/gguf.py` | Q5_K support: `_q5_k_fields` (176-byte block decode), `GGML_Q5_K=13`, type/block-size tables, and `to_intx_unpacked_to_int8_tensor` Q5_K branch (centers `[0,31]`→`[-16,15]`, folds affine min into zero-point, `target_dtype=torch.int5`, `block_size=(1,32)`). |
| `backends/cuda/cuda_backend.py` | Registers the `int5_plain_mm` AOTI ABI signature (6 tensor handles + int64 group_size → tensor). |
| `backends/cuda/quantize_op_dispatch/__init__.py` | Imports `int5_dispatch` so the dispatch registers on package import. |
| `examples/models/gemma4_31b/quant/pack_cuda.py` | `pack_linear_for_cuda` routes a Q5_K `ExportableGGUFTensor` → `CudaDp4aPlanarInt5Tensor.from_exportable_gguf`. |
| `backends/cuda/CMakeLists.txt` | Adds `runtime/shims/int5_plain_mm.cu` to `_aoti_cuda_shim_sources`. |

(No model-specific files are added — a Q5_K weight loads as genuine int5 purely
via the shared packer routing in `pack_cuda.py`.)

## GGUF Q5_K block decode (`_q5_k_fields`)

Q5_K super-block = 256 weights in a 176-byte block:
`d(2) + dmin(2) + scales(12) + qh(32) + qs(128)`.
It is Q4_K plus a 1-bit high plane:
- `d`/`dmin` fp16 super-scales, `scales` = 8 sub-blocks' 6-bit scale/min (same
  packing as Q4_K),
- `qs` (128 B) = low nibble of each weight (Q4_K nibble order),
- `qh` (32 B) = one high bit per weight; sub-block `g` (32 consecutive weights)
  reads bit `g` of each `qh` byte.

Reconstruction: `q = qs_nibble | (qh_bit << 4)` ∈ [0,31], with the affine form
`w = eff_scale · q − eff_min` (`eff_scale = d·sc`, `eff_min = dmin·mn`).
**Verified BIT-EXACT vs `gguf.dequantize`** (llama.cpp reference).

## INT5 metadata encoding (`CudaDp4aPlanarInt5Tensor`)

Asymmetric (like INT4 — Q5_K has a `dmin`), so the stored value is the raw
unsigned `u = q` ∈ [0,31] and a per-group **zero point** is subtracted at decode.
The 5 bits are split into two dp4a-friendly planes that mirror the INT4 nibble
layout so the kernel reuses INT4's even/odd extraction:

- `ql` : `(N, K/2)` uint8 — low-nibble plane, nibble-packed even/odd
  (`ql[:,j] = lo[:,2j] | (lo[:,2j+1]<<4)`, `lo = u & 0xF`).
- `qh` : `(N, K/8)` uint8 — high-1-bit plane, 8 values/byte, arranged per
  32-weight chunk as 4 bytes (one per dp4a word); each byte holds the four 1-bit
  highs of its word's even weights in the low nibble and odd weights in the high
  nibble (`hi_even_nib | (hi_odd_nib<<4)`, `hi = (u>>4)&1`).
- `scale` : `(N, n_groups)` uint8 **codes** (per-group scale).
- `zero_point` : `(N, n_groups)` uint8 **codes** (per-group zero).
- `steps` : `(N, 2)` bf16 per-row super-scale `(scale_step, zero_step)`; real
  per-group value = `code · step`. Codes are `round(value / (row_max/255))`,
  clamped to [0,255].

This is the branch's int4/int6 compaction: bf16 scale + bf16 zero (4 B/group,
5.625 bpw) → uint8 scale + uint8 zero + tiny per-row step (2 B/group). Measured
**5.505 bpw** on the guac Q5_K `output.weight` data sample (weight + metadata).

Q5_K group scales/zeros are non-negative (`d`,`dmin` ≥ 0), so the unsigned code
is exact at the row max and ~baseline elsewhere.

## CUDA kernel (`int5_plain_mm.cuh`, W5A8 dp4a)

Activations bf16→int8 in per-32 blocks (even/odd order, identical to INT4/INT6).
dp4a is linear, so the kernel reconstructs the full 5-bit weight byte per dp4a
word (`vfull = vi_lo | (spread1(hi_nibble) << 4)`) and does one dp4a per
even/odd half. Asymmetric decode:
`out += scale · a_scale · (dp − zero · a_sum)` (a_sum via dp4a against
`0x01010101`), exactly the INT4 zero-point handling. `scale_step` is a per-row
constant that factors out of the dp4a sum, so the dot products are bit-identical
to a bf16-metadata kernel. Symbols suffixed `_i5` to avoid ODR clashes with
int4/int6/int8.

## Validation results (et1, GPU 0)

| Check | Result |
| --- | --- |
| **Q5_K decode vs `gguf.dequantize`** (real guac `output.weight`, 1.34B elems) | **BIT-EXACT** — max_abs_diff = 0.0, mismatches = 0, SNR = ∞ dB |
| Q5_K → `IntxUnpackedToInt8Tensor` (int8 stage) SNR | 54.57 dB (max_abs 6.4e-3) |
| **Packed `CudaDp4aPlanarInt5Tensor` dequant vs `gguf.dequantize`** (guac Q5_K `output.weight` data sample) | **46.92 dB** (max_abs 6.39e-3), 5.505 bpw |
| `pack_int5`/`unpack_int5` round-trip | BIT-EXACT (random u∈[0,31]) |
| Eager `F.linear` dispatch vs dense dequant (M=1,4,16) | maxrel = 0.0, maxabs = 0.0 |
| **Standalone CUDA kernel numeric test** (real ET headers, sm_80, on GPU) | **PASS** — MAX_ABS_DIFF = 0.25 (tol 0.5) vs exact dispatch reference (M=4,N=16,K=256, edge values 0/31) |
| CUDA shim library build (`cmake --build cmake-out --target aoti_cuda_shims`) | RC=0 → `cmake-out/backends/cuda/libaoti_cuda_shims.so` |

## End-to-end generation — out of scope (ET-only migration)

Full model end-to-end generation is intentionally **not** performed in
`executorch-2`: this migration is the generic ET int5 backend only, so no model
is wired here. The int5 path is proven at the ET level by the checks above
(bit-exact Q5_K decode, 46.92 dB packed dequant, exact eager dispatch, on-GPU
kernel numeric test). Model-level e2e (e.g. guac temp-0 generation) belongs to
the model's own repo/task, not this migration.

## Status

- INT5 source: complete and git-staged in `~/executorch-2/executorch` (generic
  ET only; no model-specific content).
- Unit + kernel + eager-dispatch validations: pass (numbers above).
- `.pte` runtime build under `cmake-out`: the CUDA shim library
  (`libaoti_cuda_shims.so`, which contains `aoti_torch_cuda_int5_plain_mm`)
  builds cleanly (`cmake --build cmake-out --target aoti_cuda_shims`, RC=0). A
  full model `.pte` export/runtime run is out of scope for this ET migration.

## Reproduce

Validation scripts at repo root (untracked scratch):
`_int5_val_decode.py` (decode bit-exactness), `_int5_val_eager.py` (pack + eager
dispatch), `_int5_val_pack_snr.py` (real-Q5_K packed SNR). Run with
`PYTHONPATH=~/executorch-2/executorch/src CUDA_VISIBLE_DEVICES=0 conda run -n et1 python <script>`.
Standalone kernel test: `/tmp/int5_build/` (`build_and_run.sh`).
