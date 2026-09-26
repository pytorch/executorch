---
title: Arm Backend Known Issues
category: DEBUGGING
backends: [Arm]
last_validated: 2026-04-15
source_issues: [1004, 1110, 1161, 1163, 1230, 11913, 11999, 12237, 10899, 12270, 12959, 12991, 13022, 13399, 13557, 13842, 13901, 15805, 15870, 16090, 16225, 16374, 16426, 16541, 16629, 16739, 16779, 16784, 16864, 16899, 16902, 17241, 17397, 17437, 17489, 17667, 17668, 17753, 17902, 18306, 18319, 18491, 18500, 18873]
---

# Arm Backend Known Issues

## Submodule / Setup Issues

### git.mlplatform.org SSL and availability

The Arm backend's submodule (`ethos-u-core-driver`) is hosted on `git.mlplatform.org` which has recurring issues (note: `serialization_lib` has been removed from the repo):

- **SSL certificate verification failures** — `fatal: unable to access ... server certificate verification failed`
- **HTTP 500 errors** — server outages
- These failures block ALL submodule init, not just Arm submodules [Source: #1004, #1163]

**Fix:** Remove the Arm submodule if not using the Arm backend:
```bash
git submodule deinit backends/arm/third-party/ethos-u-core-driver/
```
Or disable SSL verification (not recommended): `git config --global http.sslVerify "false"` [Source: #1004]

### install_executorch.sh failures on macOS

Build failures during pip wheel build on macOS may be caused by CMake version conflicts. Some users report that downgrading CMake to 3.25, re-running the install script (which then upgrades CMake), resolves the issue. This is likely a caching/state issue. [Source: #10151]

**Best fix:** Use a clean environment and v0.6+. [Source: #10151]

## Operator / Compilation Issues

### Dynamic shapes not supported

The Arm backend cannot handle models with dynamic shapes. `SymFloat` or `SymInt` objects in the graph cause assertion failures in `get_first_fake_tensor()`.

```
AssertionError: Found zuf38 in meta["val"] of _local_scalar_dense_2, expected to find FakeTensor
```
or:
```
TypeError: Expected a FakeTensor ... but got SymFloat
```

**Workaround:** Fix all input shapes at export time. For YOLO models, remove the dynamic anchor generation. [Source: #12237]

### Attribute mutation during export

Models that mutate attributes (like YOLO's `self.anchors`) fail with strict export:
```
AssertionError: Mutating module attribute anchors during export.
```

**Fix:** Use `strict=False` in `torch.export.export_for_training()`. [Source: #12237]

### NHWC memory format conversion

TOSA requires channel-last (NHWC) format. The `Permute_Memory_Format_Pass` handles this, but was historically WIP with incomplete shape updates for neighbor operators. [Source: #1110]

### Vela compiler internal errors

Early versions had issues with Vela rejecting TOSA output:
- `AttributeError: 'ReshapeAttribute' object has no attribute 'NewshapeAsNumpy'` — case sensitivity bug in Vela
- Linear layers could fail until the TOSA-to-Vela mapping was revised [Source: #1161]

### Missing quantized op kernels

Running quantized models without delegation requires linking the quantized op library:
```
RuntimeError: Missing out variants: {'quantized_decomposed::dequantize_per_tensor', ...}
```

**Fix:** Build and link `quantized_ops_lib`. Performance without NPU delegation will be poor. [Source: #1161]

## Build Issues

### c10/macros/cmake_macros.h not found

When building backends as separate CMake projects (e.g., MediaTek LLaMA runner), you may see:
```
fatal error: 'c10/macros/cmake_macros.h' file not found
```

**Fix:** Define `C10_USING_CUSTOM_GENERATED_MACROS` in the CMakeLists.txt. This is needed whenever a separate CMake project sets up ExecuTorch include paths directly rather than using the `executorch_core` target's public compile definitions. [Source: #11999]

### Selective build for baremetal

`libportable_kernels` for Arm baremetal may not include selective build by default. Use CMake flags to enable:
```bash
-DEXECUTORCH_SELECT_OPS_FROM_MODEL="<model>.pte"
-DEXECUTORCH_DTYPE_SELECTIVE_BUILD=ON
```
[Source: #11913]

## Performance Profiling

### Vela estimator vs FVP profiling

The Vela compiler includes a performance estimator, but its estimates can differ significantly from actual FVP (Fixed Virtual Platform) profiling results. Always validate performance on FVP or real hardware. [Source: #18319]

### Non-delegated performance

Running quantized models on Cortex-M CPU without Ethos-U delegation has "tragic" performance (as noted by core team). Always use delegation for production workloads. [Source: #1161]

## Preserved Ops API

Cadence and Arm backends need `to_edge_with_preserved_ops` (experimental) to prevent decomposition of ops like `aten.rms_norm`. This API is being promoted to official status:
- `preserve_ops` will be added to `EdgeCompileConfig`
- View/mutation ops can be preserved if consumed by a delegate backend
- View/mutation ops should NOT be preserved if they remain in the portable graph [Source: #12306]

## Quantizer Issues

### Observer sharing bug at Conv-ReLU + residual junctions

The Arm Ethos quantizer incorrectly shares observers across `add`, `permute`, `relu` at residual connections. This causes quantization errors in models with skip connections (e.g., ResNet, MobileNet). Root cause: `quantization_annotator.py` doesn't properly handle shared quantization specs at add nodes. [Source: #12959]

### SharedQuantizationSpec infinite recursion

Using `SharedQuantizationSpec` with certain topologies (e.g., `minimum → eq` chains) causes `RecursionError`. Fixed upstream in pytorch/ao#3011. [Source: #13842]

### LeakyReLU fails with device mismatch

ARM quantizers (VGF, Ethos-U) fail on `nn.LeakyReLU` because the `negative_slope` constant gets placed on wrong device. XNNPACK quantizer doesn't have this. Root cause: kwargs removal in `quantization_annotator.py`. [Source: #16541]

### ReLU(inplace=True) with 16-bit activation

`ReLU(inplace=True)` with `a16w8` quantization config fails at `to_edge_transform_and_lower` with `Expected tensor aten_convolution_default in aten.clamp`. Fixed on main branch. [Source: #16629]

### FuseQuantizedActivationPass INT16 failure

`FuseQuantizedActivationPass` does not handle INT16 symmetric quantization correctly in some cases. [Source: #17437]

### aot_arm_compiler.py Conv2d quantization failure

`aot_arm_compiler.py` may not quantize `Conv2d` for `cortex-m55+int8` target in certain configurations. [Source: #17902]

### Name filter doesn't match nodes correctly

`arm_quantizer.py`'s `module_name_filter` assumes names start with `"L['self']."`, which may not be present. Fixed on main. [Source: #15870]

### GroupNorm decomposition failure

`DecomposeGroupNormPass(ArmPass)` fails when running `prepare_pt2e` on models with `torch.nn.GroupNorm`. May be related to dynamic shape handling. [Source: #16090]

## Vela Compiler Issues

### Custom config file crashes with trailing spaces

Custom `[System_Config.*]` sections crash Vela with `IndexError` if config lines have trailing spaces. Fixed in Vela 4.5.0. [Source: #15805]

### `--optimise Size` produces incorrect results

Vela with `--optimise Size` flag can produce different (wrong) results compared to default optimization. [Source: #16864]

### reduce_mean not fully delegated

Operator support checks for views/reshapes are overly pessimistic — they reject view nodes with axis-product > 65536 even when no transpose is needed. This prevents full delegation of reduce_mean to the NPU. [Source: #16779]

### Vela internal errors on certain models

Vela may crash internally on certain model structures. The Vela team is actively investigating. [Source: #13022]

## Delegation Issues

### conv→relu→permute→reshape(5D) crashes partitioner

This specific graph pattern crashes during `to_edge_transform_and_lower` for Ethos-U. [Source: #16739]

### PReLU unsupported on Ethos-U

`torch.nn.PReLU` decomposes to `torch.where(x>0, x, weights*x)` which isn't supported by the Ethos-U backend. No workaround exists. [Source: #16902]

### BatchNorm2d without preceding Conv not delegated

Standalone `BatchNorm2d` (not fused with Conv) fails Ethos-U delegation, though it works in TFLite→Vela flow. Workaround: manually decompose to `mul + add`. [Source: #17241, #17397]

### GRU / RNN layers not supported

GRU decomposition fails during Ethos-U lowering. LSTM support via CMSIS-NN is planned but not yet implemented. [Source: #12270, #17753]

### RewriteConvPass crashes on non-fuseable conv→relu branches

**Symptom**:
```
ValueError: RewriteConvPass: No output quantization parameter found in node tosa_conv2d_default
original_aten=aten.convolution.default
```
Occurs during `to_edge_transform_and_lower` when a delegated `conv → relu/clamp` branch has an activation whose output quantization has `zero_point != qmin` (non-fuseable). [Source: #18491]

**Root Cause**: `FoldAndAnnotateQParamsPass` places `output_qparams` on the downstream `clamp` node rather than the `conv` node in the non-fuseable case. `RewriteConvPass` unconditionally calls `get_output_qparams(conv)` which crashes because the conv doesn't own its output quantization.

**Fix**: Fixed by PR #18778. The fix makes `RewriteConvPass` check for `output_qparams` on successor activation nodes when the conv itself has no output qparams. [Source: #18491]

### Quantized sigmoid TABLE generation bug with qmin=-127

**Symptom**: Quantized `aten.sigmoid.default` produces incorrect outputs when lowered to TOSA TABLE with `qmin=-127, qmax=127, dtype=torch.int8`. The generated 256-entry LUT has duplicate entries and off-by-one shifts. [Source: #18873]

**Root Cause**: `InsertTableOpsPass.generate_8bit_table_values()` uses `torch.linspace(start=-127, end=127, steps=256, dtype=torch.int8)` which cannot produce 256 distinct values in a 255-code range, causing code `0` to be duplicated.

**Status**: Open issue. The fix should use the full int8 domain `[-128, 127]` as table input regardless of `qmin/qmax`, or use explicit integer range instead of `torch.linspace`. [Source: #18873]

### ConvTranspose2d fallback failure

`ConvTranspose2d` fails to fall back to CPU when it can't run on the NPU, producing "Non-passthrough operation could not run on NPU" error. [Source: #17668]

### Ethos-U base_addr mismatch

The Ethos-U backend may use `base_addr` values that don't match ExecuTorch's planned memory pool, causing output buffers to remain unchanged on real hardware despite reported successful execution. Works on FVP but fails on real MCUs. [Source: #16784]

## Performance Issues

### Softmax decomposition slow on NPU

Softmax decomposition uses `aten::amax` which runs on the elementwise engine (not MACs). The Vela performance estimator is unreliable for cycle counts — always validate on FVP or real hardware. [Source: #18319]

### LayerNorm quantization accuracy

LayerNorm quantization is sensitive to epsilon values. For transformer models (DeiT-tiny, etc.), accuracy drops in TOSA quantized pipeline may be caused by epsilon sensitivity. Use `--stable_softmax` flag for numerically stable algorithm. [Source: #16426, #18306, #18316]

### amax support added for U55

`amax` op support was added for Ethos-U55 (via Vela update). To use it, set `ArmPassPipelineConfig` in compile spec with `stable_softmax=True`. [Source: #17211]

## Setup / Build Issues

### Dependency conflicts in setup.sh

`examples/arm/setup.sh` has known dependency conflicts between ethos-u-vela (flatbuffers==24.12.23) and tosa-tools (flatbuffers==23.5.26). These are known and the backend still works. [Source: #10899, #12991]

### No module named 'tosa' after pip install

`pip install executorch` does not install tosa dependencies. Run `examples/arm/setup.sh` after pip install. Future: `pip install executorch[ethos-u]`. [Source: #13901]

### ARM GitLab access issues (resolved)

`git.gitlab.arm.com` had recurring access issues. Resolved with improved IP access management. [Source: #13557]

### Cross-compilation flatc issues

Remove manual `FLATBUFFERS_FLATC_EXECUTABLE` args — newer ExecuTorch builds handle host flatc automatically. [Source: #10964]

### strided_copy in output graph

When sample inputs are transposed (e.g., NHWC numpy arrays), `aten.as_strided_copy` appears in the graph. This is inserted by `ExportedProgram.run_decompositions()` and is often a no-op that can be removed. [Source: #16374]

## Runner Issues

### Object lifetime bug in arm_executor_runner.cpp

`BufferCleanup` used `free()` on memory from `ArmMemoryAllocator` (static pools). Hidden by FVP, crashes on real hardware. Fixed in PR #16339. [Source: #16225]

### FVP log format issues

ARM GNU compiler may not support C99 format specifiers (`%zd`) by default, causing garbled FVP output. Use `%ld` instead. [Source: #13038]

### int8 I/O with ML Toolkit

When using `QuantizeInputs`/`QuantizeOutputs` passes, the PTE expects int8 I/O. The ML Toolkit (MLEK) preprocessing may feed float data, causing type mismatches. [Source: #16899]

### Cortex-M quantization operators incorrect results

When using the Arm backend without Ethos-U delegation, Cortex-M quantization operators (`cortex_m_dequantize`, etc.) can produce incorrect results if calibration data is not representative. The default calibration in `aot_arm_compiler` uses `torch.randn(32, 2, 2)` which may not be appropriate. [Source: #13399]
