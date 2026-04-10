# Cadence Vision Backend — Complete Changelog

**Date:** February 2026
**Branch:** `stable-branch`
**Scope:** `backends/cadence/vision/` + `backends/cadence/CMakeLists.txt`

For detailed write-ups of each fix, see:
- **RUNTIME_FIXES.md** — DMA synchronization bugs, buffer management, config issues, optimizations
- **FUNCTIONALITY_FIXES.md** — XAI kernel address-dependent conv2d output corruption

---

## Overview

21 files modified (765 insertions, 1,078 deletions). 2 new documentation files created.
Changes span 10 categories: 4 runtime correctness bugs, 1 XAI kernel workaround,
1 output DMA overflow fix, DMA pipeline rewrite, config tuning, performance timing,
and debug code removal.

### Verification

Both generic (C reference) and optimized (XAI SIMD) builds were rebuilt and
tested end-to-end on quantized ResNet-18 (int8, 20 conv layers, 1000-class
ImageNet output) using `xt-run --turbo`.

| Metric               | Result                                     |
|-----------------------|--------------------------------------------|
| Top-1 classification  | Match (class 920)                         |
| Top-5 overlap         | 4 of 5                                    |
| Max output diff       | 4 quantization levels (0.29 float units)  |
| Mean output diff      | 0.75 quantization levels                  |
| Exact match           | 40.0%                                     |
| Off-by-1              | 47.4%                                     |
| Off-by-2              | 10.8%                                     |
| >2 levels             | 1.8%                                      |
| Conv2d speedup (overall) | 47.2× (generic → optimized)            |
| Conv2d speedup (range)   | 11× (deep 2×2 tiles) – 657× (early layers) |

---

## 1. DMA Synchronization Fixes — Conv Executors

Added `idma_hw_wait_all` at three synchronization points in all five
DMA conv executor functions:

1. After initial coefficient + bias (ch0) and first input tile (ch1) loads
2. Before each convolution in the tiling loop
3. After the loop — wait for final output DMA store before returning

**Files:**
- `operators/conv/conv_exec_1x1j1d1.c`
- `operators/conv/conv_exec_1x1j2d1.c`
- `operators/conv/conv_exec_3x3j1d1.c`
- `operators/conv/conv_exec_3x3j2d1.c`
- `operators/conv/conv_exec_7x7j2d1.c`

---

## 2. Address-Sensitive FOLD16 Bug Fix — Conv2D Output Corruption

**Symptom:** Layers 16–19 (3×3, 2×2 spatial output tiles,
`inDataPitch1 ≤ 16`, FOLD16 code path) produced `max_diff` up to 24
vs. the generic C reference. Coefficient bytes [1] and [2] of ch0 were
corrupted to -128 (input zero-point fill value). Only even-N tiles
(using `p_input0`) affected.

**Root cause:** Unconfirmed. The bug is address-dependent — identical
data at different buffer addresses produces different results. The
precise mechanism inside the XAI FOLD16 kernel is unknown. An earlier
analysis attributed this to 16KB data cache aliasing, which was
disproved (64KB 2-way cache, no cache simulation in `xt-run --turbo`,
and dcache invalidation was never in the VQ code path).

See **FUNCTIONALITY_FIXES.md** for complete analysis, evidence, and
investigation timeline.

**Fix (workaround):**
1. Reordered DRAM buffer allocations — `p_coeff` allocated before
   `p_input0`/`p_input1`, changing the relative buffer addresses.
2. Added `xthal_dcache_region_invalidate()` in non-VQ executor
   functions (correct practice for future hardware deployment,
   but not in the VQ path that ResNet-18 quantized uses).

**Files:**
- `operators/conv/conv_exec_3x3j1d1.c`
- `operators/conv/conv_exec_3x3j2d1.c`

---

## 3. 7×7 Conv Output DMA Overflow Fix

The output DMA store in `conv_exec_7x7j2d1.c` was writing the full tile
height for the last height tile, potentially spilling past the valid output
boundary into the next channel's memory. Fixed to only write valid output
rows for the final tile.

**Files:**
- `operators/conv/conv_exec_7x7j2d1.c`

---

## 4. DMA Pipeline Rewrite — Element-wise Operators

Rewrote all DMA-accelerated element-wise operators from single-channel
`idma_copy_2d_desc` + `idma_desc_done` to dual-channel `dma_1dm` with
proper synchronization:

- Channel 0 for loads, Channel 1 for stores (overlapped operation)
- Replaced inline `idma_init`/`idma_init_loop` (NULL error callback,
  1 descriptor slot) with wrapper `dma_2dm_init(ch)` (proper error
  callback, 2 descriptor slots)

**Files:**
- `operators/quantize_per_tensor.cpp`
- `operators/dequantize_per_tensor.cpp`
- `operators/quantized_relu.cpp`
- `operators/op_softmax.cpp`
- `operators/op_add.cpp`

---

## 5. DRAM Buffer Size Unification

Two independent DRAM size definitions existed (`DRAM0_BUFF_SIZE` from CMake
vs `IDMA_BUFFER_SIZE_DRAM0` in `conv_layer_configs.h`). Replaced all
`DRAM0_BUFF_SIZE`/`DRAM1_BUFF_SIZE` usage with the config header values.
Removed the CMake `add_compile_definitions`.

**Files:**
- `operators/quantize_per_tensor.cpp`
- `operators/dequantize_per_tensor.cpp`
- `operators/quantized_relu.cpp`
- `operators/op_softmax.cpp`
- `operators/op_add.cpp`
- `operators/quantized_conv2d_nchw_out_per_tensor.cpp`
- `backends/cadence/CMakeLists.txt`

---

## 6. Conv2D Quantization Precision Fix

Enhanced quantization for correct classification output:

- **`accumShift` computation:** Uses actual per-layer weight L1
  norms instead of worst-case bounds, preserving 1–3 extra bits of
  accumulator precision
- **`output_zero_point` absorption:** Converted to accumulator domain
  and added to kernel bias before kernel execution, eliminating
  post-kernel addition loop and double-clamp issue
- **24-bit bias clamping:** Kernel bias clamped to 24-bit range
  (ACC_INIT_BIAS takes lower 24 bits), with per-channel residual
  correction applied post-kernel
- **Disabled in-kernel ReLU:** `relu_min=-128, relu_max=127`
  (ReLU handled by separate fused operator)

Without this fix, ResNet-18 quantized misclassifies (top-1: class 644).
With the fix, correct classification (top-1: class 920).
See **FUNCTIONALITY_FIXES.md §8** for full analysis.

Other changes retained:
- **Added `input_zero_point` field** to `conv_layer_config_t` struct
- **Fixed wrong config header include** — replaced all
  `conv_layer_configs1.h` with `conv_layer_configs.h`

**Files:**
- `operators/quantized_conv2d_nchw_out_per_tensor.cpp`
- `operators/conv/kernel_executors.h`
- `operators/conv/conv_layer_configs.h`
- `third-party/include/memory_manager.h`

---

## 7. Conv Layer Config Tuning

Systematic re-tuning of all 29 layer tile configurations in
`conv_layer_configs.h` (294 insertions / 293 deletions):

- DRAM pool sizes increased: 32 KB → 62 KB
- Doubled `n_tile_size` (e.g., 8→16, 16→32, 64→128, 128→256)
- Halved `n_tiles` accordingly
- Doubled `coeff_buffer_size` and `output_buffer_size`
- Adjusted `coeff_dram` placement between DRAM banks 0/1
- Tuned input/output tile dimensions for 1×1 layers

**Files:**
- `operators/conv/conv_layer_configs.h`

---

## 8. Performance Timing Macros

Added `TIME_DECL`/`TIME_START`/`TIME_END`/`TIME_DISPLAY` (PERF_LOG cycle
measurement) to all operators that previously lacked timing:

- `op_max_pool2d_with_indices.cpp`
- `op_softmax.cpp` — all code paths
- `quantized_conv2d_nchw_out_per_tensor.cpp`

**Fixed `TIME_START` macro** in `lib.h`: changed from `XT_WSR_CCOUNT(0)`
(writes/resets the hardware cycle counter) to `XT_RSR_CCOUNT()` (reads
without side effects). The old macro would reset the cycle counter for
every operator, breaking nested or cumulative timing.

**Files:**
- `operators/op_max_pool2d_with_indices.cpp`
- `operators/op_softmax.cpp`
- `operators/quantized_conv2d_nchw_out_per_tensor.cpp`
- `third-party/include/lib.h`

---

## 9. Debug Code Removal

Removed all debug output across the codebase:

- All `printf` statements (error messages, debug logging, config dumps)
- `#include <stdio.h>` from conv executors and kernel dispatcher
- `#include <iostream>` from maxpool operator
- CSV logging infrastructure from conv2d operator (~40 lines)
- Commented-out code blocks from `memory_manager.h`, `dma.c`, `lib.h`
- Diagnostic `[DBG]` printf instrumentation from `conv_exec_3x3j1d1.c`
  (coefficient/bias/output dumps used for DMA-cache coherency investigation)
- `dbg_dump` helper function removed from `conv_exec_3x3j1d1.c`

**Files:**
- All conv executors (`conv_exec_*.c`)
- `operators/conv/conv_kernel_dispatcher.c`
- `operators/op_max_pool2d_with_indices.cpp`
- `operators/quantized_conv2d_nchw_out_per_tensor.cpp`
- `third-party/include/memory_manager.h`
- `third-party/include/lib.h`
- `third-party/library/dma.c`

---

## 10. DMA Library Cleanup

Cleaned up `dma.c` and `dma.h`:

- Removed `MEASURE_DMA_CYCLES` instrumentation blocks
- Removed `IDMA_DEBUG` conditional code
- Removed non-multichannel `#else` fallback paths (assumes multichannel iDMA)
- Removed unused `dma_2dm_schd` function
- Silenced error callback (`printf` → `(void) error`)
- Formatting cleanup in `dma.h`

**Files:**
- `third-party/library/dma.c`
- `third-party/include/dma.h`

---

## Complete File List

| # | File | Categories |
|---|------|------------|
| 1 | `operators/conv/conv_exec_1x1j1d1.c` | §1 DMA sync, §9 Debug removal |
| 2 | `operators/conv/conv_exec_1x1j2d1.c` | §1 DMA sync, §9 Debug removal |
| 3 | `operators/conv/conv_exec_3x3j1d1.c` | §1 DMA sync, §2 XAI workaround, §9 Debug removal |
| 4 | `operators/conv/conv_exec_3x3j2d1.c` | §1 DMA sync, §2 XAI workaround, §9 Debug removal |
| 5 | `operators/conv/conv_exec_7x7j2d1.c` | §1 DMA sync, §3 Output overflow fix, §9 Debug removal |
| 6 | `operators/conv/conv_kernel_dispatcher.c` | §9 Debug removal |
| 7 | `operators/conv/conv_layer_configs.h` | §6 Conv2D rework, §7 Config tuning |
| 8 | `operators/conv/conv_layer_configs1.h` | §6 Conv2D rework (include replaced) |
| 9 | `operators/conv/kernel_executors.h` | §6 Conv2D rework |
| 10 | `operators/quantize_per_tensor.cpp` | §4 DMA rewrite, §5 DRAM unification |
| 11 | `operators/dequantize_per_tensor.cpp` | §4 DMA rewrite, §5 DRAM unification |
| 12 | `operators/quantized_relu.cpp` | §4 DMA rewrite, §5 DRAM unification |
| 13 | `operators/op_softmax.cpp` | §4 DMA rewrite, §5 DRAM unification, §8 Timing |
| 14 | `operators/op_add.cpp` | §4 DMA rewrite, §5 DRAM unification |
| 15 | `operators/op_max_pool2d_with_indices.cpp` | §8 Timing, §9 Debug removal |
| 16 | `operators/quantized_conv2d_nchw_out_per_tensor.cpp` | §5 DRAM unification, §6 Conv2D rework, §8 Timing, §9 Debug removal |
| 17 | `third-party/include/dma.h` | §10 DMA cleanup |
| 18 | `third-party/include/lib.h` | §8 Timing fix, §9 Debug removal |
| 19 | `third-party/include/memory_manager.h` | §6 Conv2D rework, §9 Debug removal |
| 20 | `third-party/library/dma.c` | §9 Debug removal, §10 DMA cleanup |
| 21 | `third-party/library/memory_manager.c` | §9 Debug removal |
| 22 | `../../backends/cadence/CMakeLists.txt` | §5 DRAM unification |

---

## Known Deferred Issues

1. **`allocate_dram_buffer` never returns NULL on overflow** — The DRAM
   buffer allocator does not check whether the requested allocation exceeds
   remaining pool space. Silently returns a pointer past the pool boundary.
