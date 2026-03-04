# Conv2D Output Corruption — XAI FOLD16 Address-Sensitive Bug

**Date:** February 2026  
**Scope:** `backends/cadence/vision/operators/conv/conv_exec_3x3j1d1.c`,
`conv_exec_3x3j2d1.c`  
**Platform:** Xtensa Vision P6 (XTENSA_CORE=Vision_130_64k_iDMA_Cache),
toolchain RJ-2024.4-linux  
**Simulation:** `xt-run --turbo` (TurboXim — fast functional, no cache
modeling)  
**Status:** FIXED (workaround) — Root cause mechanism unconfirmed.

---

## 1. The Bug

Convolution layers 16–19 of quantized ResNet-18 (int8) produced incorrect
output on the Xtensa Vision P6 DSP. The affected layers are 3×3 stride-1
and stride-2 convolutions with **2×2 spatial output tiles** (the smallest
tile size in the network). Expected `max_diff ≤ 1` (single quantization
step) vs. the generic C reference; observed `max_diff` up to 24. Layers
0–15 were unaffected.

---

## 2. What We Know

### 2.1 The Bug Is Address-Dependent

The buffer swap test (§3.1) proved definitively that the bug depends on
the **memory address** of the coefficient and input buffers, not their
contents. Identical data at different addresses produces different results.

### 2.2 Which Layers and Why

Only layers with `inDataPitch1 ≤ 16` are affected. This routes them
through the **FOLD16** variant of the XAI kernel
(`convolved3D_S_3x3j1d1_S8S8IX_MOW_WHD_FOLD16` in
`third-party/libxai/cnn/src/cnn_dilated_conv_MOW.h`, line ~4813).
With `outH ≤ 3`, the main loop (`for y = 0; y < outH - 3`) never
executes; all processing goes through the "leftover rows" code path
(line ~5058).

Earlier layers use FOLD32 or general variants (larger `inDataPitch1`),
which are not affected.

### 2.3 Coefficient Corruption Pattern

Diagnostic dumps showed that coefficient bytes [1] and [2] of ch0 are
read back as **-128** (the input zero-point fill value) instead of
their correct values. Only even-N tiles (using `p_input0`) are affected;
odd-N tiles (using `p_input1`) produce correct results. The output error
is a near-constant offset (~52), consistent with corrupted filter weights.

### 2.4 The Original Buffer Layout

In the original allocation order (input first, coeff second):

```
p_input0 → dram0_pool + 0x0000          (pool base)
p_input1 → dram0_pool + input_size
p_coeff  → dram0_pool + 2×input_size    (pool base + 0x4000)
```

| Property              | Buggy (input-first) | Fixed (coeff-first) |
|-----------------------|---------------------|---------------------|
| `p_input0` address    | `0x3ffc0380`        | `0x3ffc9380`        |
| `p_coeff` address     | `0x3ffc4380`        | `0x3ffc0380`        |
| Offset (coeff−input0) | `0x4000`            | `-0x9000`           |

### 2.5 Cache Is Not the Cause

The initial analysis attributed this to 16KB data cache aliasing +
iDMA incoherency. **This was wrong on three independent grounds:**

1. **Wrong cache geometry.** The core has a **64KB 2-way set
   associative** data cache (`XCHAL_DCACHE_SIZE=65536`,
   `XCHAL_DCACHE_WAYS=2`, `XCHAL_DCACHE_LINESIZE=128`). Way size =
   32KB. Two addresses 0x4000 (16KB) apart do **not** map to the same
   cache set in this geometry.

2. **No cache simulation.** `xt-run --turbo` is TurboXim — fast
   functional simulation. Cache behavior is only modeled with
   `--mem_model`. In turbo mode, all loads/stores go directly to
   memory. There is no stale data, no cache eviction, no coherency
   issue.

3. **Wrong execution path.** The `xthal_dcache_region_invalidate()`
   calls were added to the **non-VQ** function (`conv_exec_3x3j1d1`,
   line 552), but ResNet-18 quantized runs through the **VQ** function
   (`conv_exec_3x3j1d1VQ`, line 31) — which has zero dcache
   invalidation calls. The "fix" was never executed.

### 2.6 Root Cause Is Unconfirmed

The precise mechanism inside the XAI FOLD16 kernel that makes it
address-sensitive in TurboXim functional simulation is **unknown**.

Possible hypotheses (untested):
- TurboXim ISS bug in modeling vector load priming (`IVP_LA2NX8_PP`)
  or variable-length loads (`IVP_LAV2NX8_XP`) when coefficient and
  input pointers have specific relative alignments
- Buffer overlap due to the bump allocator not respecting alignment
  constraints for vector register priming
- XAI kernel internal pointer arithmetic that assumes a minimum
  separation between coefficient and input buffer addresses

---

## 3. Evidence

### 3.1 Address-Dependency Proof (Buffer Swap Test)

After dumping both ping-pong input buffers (`p_input0`, `p_input1`) and
confirming they contained **identical data**, the pointers were swapped:

```c
int8_t* temp = p_input0;
p_input0 = p_input1;
p_input1 = temp;
```

**Result:** The error pattern flipped — previously correct output groups
became buggy and vice versa. This proved the bug is **address-dependent**.

### 3.2 Coefficient Corruption Pattern (Diagnostic Dump)

Added printf diagnostics to dump bias values, coefficient bytes, and
output values per FOLD16 tile. Ran full ResNet-18 under both allocation
orders and diffed the output:

| Data         | Buggy run                        | Fixed run  |
|--------------|----------------------------------|------------|
| Bias values  | Identical                        | Identical  |
| coeff_ch0 [0] | Correct                        | Correct    |
| coeff_ch0 [1] | **Corrupted** (e.g., −128)     | Correct    |
| coeff_ch0 [2] | **Corrupted** (e.g., −109)     | Correct    |
| coeff_ch0 [3..8] | Correct                     | Correct    |
| coeff_ch1    | Identical                        | Identical  |
| Input data   | Identical                        | Identical  |
| Output ch0   | **Shifted by ~52**               | Correct    |
| Output ch1   | Identical                        | Identical  |

Key observations:
- Only **coeff_ch0** bytes [1] and [2] corrupted.
- Corrupted values (−128) match the **input zero-point fill** value,
  suggesting the kernel reads from the wrong memory region.
- Only **even-N tiles** (using `p_input0`) affected; odd-N tiles
  (using `p_input1`) correct.
- Output error is a **near-constant offset**, consistent with corrupted
  filter weights producing a systematically biased convolution result.

---

## 4. Fix (Workaround)

### 4.1 Allocation Reorder

Reorder DRAM buffer allocations so `p_coeff` is allocated **before**
`p_input0` and `p_input1`. This changes the relative addresses of the
coefficient and input buffers, avoiding the address pattern that triggers
the FOLD16 bug:

```c
// FIX: Allocate coeff FIRST to avoid address-sensitive FOLD16 bug.
// See FUNCTIONALITY_FIXES.md §2 for details.
int8_t* p_coeff  = allocate_dram_buffer(config->coeff_buffer_size, ...);
int8_t* p_input0 = allocate_dram_buffer(config->input_buffer_size, ...);
int8_t* p_input1 = allocate_dram_buffer(config->input_buffer_size, ...);
```

### 4.2 Cache Invalidation (For Future Hardware Use)

`xthal_dcache_region_invalidate()` calls were added to the **non-VQ**
executor functions after DMA transfers. These do nothing in `xt-run
--turbo` (no cache modeled) and are not in the VQ execution path used
by ResNet-18 quantized. They are retained as correct practice for
eventual deployment on real hardware with data cache enabled.

### 4.3 Why This Is a Workaround, Not a Root Cause Fix

The allocation reorder eliminates the bug empirically — all 20
convolution layers produce correct output. However, because the root
cause mechanism is unconfirmed, the fix is **address-specific**: it works
for the current buffer sizes and DRAM pool layout. A different model with
different buffer sizes could potentially re-trigger the same issue.

A true fix requires either:
- Identifying the exact FOLD16 kernel bug (in the XAI library source or
  the TurboXim ISS model)
- Or adding cache invalidation to the VQ executor functions as well, for
  hardware deployment

---

## 5. Result (Allocation Reorder Only)

With the allocation reorder alone (original quantization approach),
all 20 convolution layers produce **max_diff ≤ 1** vs. the generic C
reference. However, top-1 classification is **class 644** (incorrect).
Top-5: [644, 530, 920, 815, 818].

With both the allocation reorder and the enhanced quantization (§8),
top-1 classification is **class 920** (correct). Top-5: [920, 530, 644,
916, 818].

---

## 6. Files Modified

| File | Change |
|------|--------|
| `operators/conv/conv_exec_3x3j1d1.c` | Allocation reorder (VQ + non-VQ) + `xthal_dcache_region_invalidate` (non-VQ only) |
| `operators/conv/conv_exec_3x3j2d1.c` | Allocation reorder (VQ + non-VQ) + `xthal_dcache_region_invalidate` (non-VQ only) |
| `operators/quantized_conv2d_nchw_out_per_tensor.cpp` | Enhanced quantization (§8) |

---

## 7. XAI Library — Affected Function

The bug is triggered by the XAI FOLD16 kernel. The library is third-party
and was not modified. The specific function is
`convolved3D_S_3x3j1d1_S8S8IX_MOW_WHD_FOLD16` in
`third-party/libxai/cnn/src/cnn_dilated_conv_MOW.h` (line ~4813).

The FOLD16 "leftover rows" path (line ~5058) processes coefficient and
input data through an `IVP_LAV2NX8_XP`-based inner loop with
`IVP_LA2NX8_PP` priming. The address-dependent failure only occurs when
the `pCoeffData` and `pInData` pointers have a specific relative offset
(0x4000 in this case).

---

## 8. Quantization Precision Fix — accumShift and output_zero_point

### 8.1 The Problem

The XAI kernel pipeline computes:
```
out = ((acc >> accumShift) * outputScale) >> outputShift
```

The kernel internally saturates the shifted accumulator to **int16**
[-32768, 32767] after `accumShift`. This means `accumShift` must be
large enough to keep the worst-case accumulator in range, but every
extra bit of shift **loses one bit of precision** from the accumulator.

The original quantization approach used a **worst-case** bound:
```
max_acc = num_products × 127 × (127 + |in_zero_point|)
```
where `num_products = ic_per_group × kh × kw`. For deep layers with
large `num_products` (e.g., 512×3×3 = 4608), this produces
`accumShift = 10`, discarding the lower 10 bits of every accumulator.

Additionally, `output_zero_point` was added **after** the kernel
returned, element-by-element, introducing an extra clamp boundary
where the kernel output (already quantized) was shifted and re-clamped.

### 8.2 The Fix

Three improvements:

1. **L1-norm accumShift:** Instead of worst-case `127 × num_products`,
   compute the actual maximum L1 weight norm across all output channels:
   ```
   max_acc = max(|kernel_bias|) + max(L1(weights_oc)) × 128
   ```
   This produces tighter `accumShift` values — typically 1–3 bits less
   than the worst-case bound, preserving significantly more precision.

2. **output_zero_point absorption:** Convert `output_zero_point` into
   the accumulator domain (`zp_acc = output_zero_point / effective_scale`)
   and add it to the kernel bias before the kernel runs. This eliminates
   the post-kernel addition loop and avoids the double-clamp.

3. **24-bit bias clamping with residual correction:** The XAI kernel's
   `ACC_INIT_BIAS` instruction loads only the lower 24 bits of the bias.
   The combined bias (input_zero_point correction + output_zero_point
   absorption) can exceed 24 bits. The fix clamps to [-2²³, 2²³-1] and
   tracks the residual per output channel. After the kernel returns,
   the residual (converted back to output scale) is added per-channel.

### 8.3 Empirical Result

Tested on quantized ResNet-18 with allocation reorder applied in both
cases:

| Quantization Approach | Top-1 Class | Top-5 Classes          | Correct? |
|----------------------|-------------|------------------------|----------|
| Original (worst-case) | **644**     | [644, 530, 920, 815, 818] | **No** |
| Enhanced (L1-norm)    | **920**     | [920, 530, 644, 916, 818] | **Yes** |

The original approach's excessive `accumShift` loses enough precision
to change the final classification. The enhanced approach preserves
sufficient precision for correct top-1 output.

### 8.4 Files Modified

| File | Change |
|------|--------|
| `operators/quantized_conv2d_nchw_out_per_tensor.cpp` | Enhanced quantization (active), original commented out |

---

# Appendix A: Investigation Timeline

The bug required 12 investigation steps to diagnose. This appendix
summarizes the hypotheses tested and eliminated.

### A.1 Dual-Path Comparison

Built a dual-path executor running both the generic C reference and the
XAI DMA kernel on the same input, comparing outputs element-by-element.
Confirmed the bug was in the XAI kernel path, not in bias correction,
quantization parameters, or DMA data movement.

### A.2 Hypothesis: FOLD16 Remainder Path — DISPROVED

All affected layers have output channel counts that are exact multiples
of 16 (256 or 512), ruling out a FOLD16 remainder path issue.

### A.3 Hypothesis: 24-bit Accumulator Overflow — DISPROVED

The XAI kernel's `ACC_INIT_BIAS` loads only the lower 24 bits. Tested
with 24-bit-clamped bias + post-correction residual. Error persisted.

### A.4 Hypothesis: DMA Row Count Mismatch — DISPROVED

The config field `in_rows_firstdma` differed from `input_rows` for
affected layers. Adjusting DMA row counts broke other layers sharing
the `height_tiles=1` configuration.

### A.5 Hypothesis: DRAM Bank Placement — DISPROVED

Moved input buffers to different DRAM banks. No effect.

### A.6 Buffer Content Dump

Dumped byte contents of both ping-pong input buffers before the first
convolution call. Both contained **identical data** — proving incorrect
data loading was not the cause.

### A.7 Buffer Address Swap — BREAKTHROUGH

Swapped `p_input0` and `p_input1` pointers after allocation (§3.1 above).
The error pattern flipped, proving the bug is **address-dependent**.

### A.8 FOLD16 Source-Level Analysis

Traced through `cnn_dilated_conv_MOW.h`:
- Entry: `xaiConvolved3D_S_3x3j1d1_S8S8IX_MOW_WHD` (line ~5453)
- Dispatch: `inDataPitch1 ≤ 16` → FOLD16 variant (line ~4813)
- With `outH=2`: main loop skipped, all processing in leftover-rows
  path (line ~5058) using `IVP_LAV2NX8_XP` (variable-length loads)

### A.9 Disassembly Analysis

Disassembled the compiled object (`xt-objdump -d`). The S8 3×3j1d1
function spans 946 lines of VLIW assembly (offsets 0x0–0x1e98). Mapped
4 processing blocks, confirmed the `bltui.w15 a8, 2` branch at 0xb68
always skips the main loop for `outH=2`.

### A.10 IVP_LAV2NX8_XP Standalone Test — DISPROVED

Wrote a standalone test exercising `IVP_LAV2NX8_XP` at 35+ different
alignments and 11 byte counts. **All 6 test suites passed.** The
instruction is functionally correct and address-independent on this core.

> Note: An initial test (v1) showed false failures due to overlapping
> `memcpy` (undefined behavior). After fixing to use separate source
> buffers, all results matched.

### A.11 Coefficient Dump — Address Correlation Found

Added diagnostic dumps for bias, coefficient bytes, and output values.
Diffed results between buggy and workaround allocation orders. Found
coefficient ch0 bytes [1] and [2] corrupted to input zero-point values.
Computed buffer addresses: `p_coeff` was at offset 0x4000 from
`p_input0`. Reordering allocations to place `p_coeff` first (at pool
base) eliminated the corruption.

### A.12 Cache Aliasing Theory — DISPROVED

Initial analysis attributed the bug to 16KB data cache aliasing + iDMA
incoherency. This was disproved:
- Core has 64KB 2-way set associative cache (way size = 32KB), not 16KB
  direct-mapped — 0x4000 offset does not cause set aliasing
- `xt-run --turbo` does not model caches — all memory accesses are direct
- The `xthal_dcache_region_invalidate()` calls were only in the non-VQ
  function, never executed for ResNet-18 quantized

---

# Appendix B: Diagnostic Artifacts

- Diagnostic logs: `/tmp/resnet18_buggy_bias.txt` and
  `/tmp/resnet18_workaround_bias.txt` (5913 lines each)
- Disassembly: `/tmp/cnn_dilated_conv_MOW.dis` (50k lines, S8 3×3j1d1
  function at line 30275)
- Standalone test: `/tmp/test_ivp_lav2.c` (fixed v2).
  Build: `xt-clang -O3 -mcoproc -mlongcalls -LNO:simd`.
  Run: `xt-run /tmp/test_ivp_lav2`

---

# Appendix C: Platform Cache Configuration Reference

From `core-isa.h` for `Vision_130_64k_iDMA_Cache`:

```
XCHAL_DCACHE_SIZE       = 65536   (64 KB)
XCHAL_DCACHE_WAYS       = 2       (2-way set associative)
XCHAL_DCACHE_LINESIZE   = 128     (128-byte lines)
XCHAL_DCACHE_SETWIDTH   = 8       (256 sets)
XCHAL_DCACHE_IS_WRITEBACK = 1
XCHAL_DCACHE_IS_COHERENT  = 0
```

The `xthal_dcache_region_invalidate()` calls in the non-VQ executor are
correct practice for real hardware deployment (iDMA does not maintain
cache coherency), but are irrelevant in TurboXim simulation and not
present in the VQ execution path.
