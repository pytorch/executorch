# Cadence Vision Backend — Runtime Bug Fixes & Optimizations

**Date:** February 2026  
**Scope:** `backends/cadence/vision/` — DMA operators, conv executors, configuration headers

---

## Summary

The vision backend had multiple issues causing stalls, wrong output, and severe performance degradation. This document covers **runtime correctness bugs** (DMA synchronization, buffer management, configuration) and **optimization/consistency improvements**.

For the convolution output mismatch investigation and XAI kernel bug fix, see the companion document: **FUNCTIONALITY_FIXES.md**.

---

# Part 1: Runtime Correctness Bugs

---

## Bug 1: Missing `idma_hw_wait_all` in Conv Executors

**Symptom:** Conv executor stalling or producing corrupt output.

**Root Cause:** The DMA conv executor functions (`conv_exec_*`) issued asynchronous DMA transfers via `dma_1dm`, `dma_2dm`, `dma_3dm` but never waited for completion before using the destination buffers. The CPU could begin convolution on partially-transferred input/coefficient data.

**Fix:** Added `idma_hw_wait_all` at three synchronization points in all five DMA conv executor functions:

1. **After initial loads** — wait for coefficients + bias (ch0) and first input tile (ch1) before the first convolution call.
2. **Before each convolution in the loop** — wait for the previous iteration's output store / coefficient prefetch (ch0) and the current input prefetch (ch1).
3. **After the loop exits** — wait for the final output DMA store (ch0) to complete before returning.

**Files Modified:**
- `operators/conv/conv_exec_1x1j1d1.c`
- `operators/conv/conv_exec_1x1j2d1.c`
- `operators/conv/conv_exec_3x3j1d1.c`
- `operators/conv/conv_exec_3x3j2d1.c`
- `operators/conv/conv_exec_7x7j2d1.c`

---

## Bug 2: Unsafe Inline DMA Initialization in Quantize/Dequantize/ReLU

**Symptom:** Potential stalls or silent DMA errors.

**Root Cause:** `quantize_per_tensor.cpp`, `dequantize_per_tensor.cpp`, and `quantized_relu.cpp` initialized iDMA with inline calls to `idma_init` + `idma_init_loop` using a **NULL error callback** and only **1 descriptor slot**. The NULL callback means DMA errors are silently swallowed. The 1-slot descriptor ring provides no pipelining.

**Fix:** Replaced inline init with the wrapper `dma_2dm_init(ch)` from `dma.c`, which uses a proper error callback (`err_cb_func`) and 2 descriptor slots (`CHL_MAX`).

**Files Modified:**
- `operators/quantize_per_tensor.cpp`
- `operators/dequantize_per_tensor.cpp`
- `operators/quantized_relu.cpp`

---

## Bug 3: DRAM Buffer Size Mismatch

**Symptom:** Operators could overrun DRAM pool boundaries, corrupting adjacent memory.

**Root Cause:** Two independent size definitions existed:
- `DRAM0_BUFF_SIZE` / `DRAM1_BUFF_SIZE` — passed from CMake via `add_compile_definitions`
- `IDMA_BUFFER_SIZE_DRAM0` / `IDMA_BUFFER_SIZE_DRAM1` — hardcoded in `conv_layer_configs.h`

The DRAM pool arrays are sized using `IDMA_BUFFER_SIZE_DRAM0/1`, but some operators used `DRAM0_BUFF_SIZE` for chunk sizing. If these values disagreed, operators could attempt to use more memory than allocated.

**Fix:** Replaced all uses of `DRAM0_BUFF_SIZE` → `IDMA_BUFFER_SIZE_DRAM0` and removed the CMake `add_compile_definitions`. Single source of truth: `conv_layer_configs.h`.

**Files Modified:**
- `operators/quantize_per_tensor.cpp`, `dequantize_per_tensor.cpp`, `quantized_relu.cpp`
- `operators/op_softmax.cpp`, `operators/op_add.cpp`
- `CMakeLists.txt`

---

## Bug 4: Wrong Config Header — Conv Layers Not Matched

**Symptom:** ~142M cycles for 12,288-element conv layer (~11,637 cycles/element). **447× slower than expected.**

**Root Cause:** Two config header files existed:
- `conv_layer_configs.h` — correct configs for the actual model (64×64 input)
- `conv_layer_configs1.h` — configs for a different model (224×224 ResNet-18)

Three files included `conv_layer_configs1.h`. Because configurations didn't match the model's layer shapes, `get_layer_config_by_params()` returned NULL for every layer, causing fallback to the generic scalar C implementation.

**Fix:** Replaced all `conv_layer_configs1.h` includes with `conv_layer_configs.h`. Added the `input_zero_point` struct field to the config struct.

**Result:** Conv2d performance improved from **142,989,317 → 319,992 cycles** (447× speedup).

**Files Modified:**
- `operators/quantized_conv2d_nchw_out_per_tensor.cpp`
- `operators/conv/kernel_executors.h`
- `third-party/include/memory_manager.h`
- `operators/conv/conv_layer_configs.h`

---

# Part 2: Optimization & Consistency Improvements

---

## Optimization 1: Dual-Channel DMA — Load/Store Overlap

**Before:** All element-wise operators used a single DMA channel. Loads and stores fully serialized.

**After:** Implemented dual-channel DMA (channel 0 for loads, channel 1 for stores). Enables overlapping load of next chunk with store of previous result.

**Files Modified:**
- `operators/quantize_per_tensor.cpp`, `dequantize_per_tensor.cpp`
- `operators/quantized_relu.cpp`, `operators/op_softmax.cpp`, `operators/op_add.cpp`

---

## Consistency 1: Raw iDMA Calls Replaced with Wrapper Functions

Replaced all raw `idma_copy_2d_desc` → `dma_1dm` and `idma_desc_done` → `idma_hw_wait_all`. Centralizes DMA management through `dma.c` with proper error callbacks.

---

## Consistency 2: Misleading DRAM Pool Size Comments

Removed "40 KB" comments from `memory_manager.h` / `memory_manager.c` where actual pool sizes were 64,000 bytes.

---

# Complete File Modification List

| File | Category | Issues |
|------|----------|--------|
| `operators/conv/conv_exec_1x1j1d1.c` | Runtime | Bug #1 |
| `operators/conv/conv_exec_1x1j2d1.c` | Runtime | Bug #1 |
| `operators/conv/conv_exec_3x3j1d1.c` | Runtime | Bug #1 |
| `operators/conv/conv_exec_3x3j2d1.c` | Runtime | Bug #1 |
| `operators/conv/conv_exec_7x7j2d1.c` | Runtime | Bug #1 |
| `operators/quantize_per_tensor.cpp` | Both | Bug #2, #3, Opt #1, Con #1 |
| `operators/dequantize_per_tensor.cpp` | Both | Bug #2, #3, Opt #1, Con #1 |
| `operators/quantized_relu.cpp` | Both | Bug #2, #3, Opt #1, Con #1 |
| `operators/op_softmax.cpp` | Both | Bug #3, Opt #1, Con #1 |
| `operators/op_add.cpp` | Both | Bug #3, Opt #1, Con #1 |
| `operators/conv/conv_layer_configs.h` | Runtime | Bug #4 |
| `operators/conv/kernel_executors.h` | Runtime | Bug #4 |
| `operators/quantized_conv2d_nchw_out_per_tensor.cpp` | Runtime | Bug #3, #4 |
| `third-party/include/memory_manager.h` | Both | Bug #4, Con #2 |
| `third-party/library/memory_manager.c` | Consistency | Con #2 |
| `../../CMakeLists.txt` | Runtime | Bug #3 |

---

## Known Deferred Issues

1. **`allocate_dram_buffer` never returns NULL on overflow** — The DRAM buffer allocator does not check whether the requested allocation exceeds remaining pool space. Silently returns a pointer past the pool boundary.
