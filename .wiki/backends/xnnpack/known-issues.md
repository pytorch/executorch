---
title: "XNNPACK Backend Known Issues and Workarounds"
category: DEBUGGING
backends: [XNNPACK]
last_validated: 2026-04-05
source_issues: [1263, 1287, 1306, 1330, 1340, 1350, 2163, 3636, 3696, 4005, 4504, 5068, 5264, 5265, 5381, 7748, 7775, 7880, 8177, 8369, 8508, 8539, 8700, 8830, 8884, 8924, 10297, 10602, 10663, 11355, 11523, 11738, 12271, 12804, 12817, 14321, 14644, 14735, 14741, 14809, 14831, 14987, 15914, 16406, 17301, 17482, 17669, 18487, 18562]
---

# XNNPACK Backend Known Issues and Workarounds

## Operator Support Gaps

### Batch Norm Without Preceding Conv

**Symptom**: `RuntimeError: For aten__native_batch_norm_legit_no_training_default` during partitioning.

**Cause**: XNNPACK only supports batch_norm when it follows a convolution (for conv+BN fusion). Standalone batch_norm is not supported. [Source: #1340]

**Workaround**: Either restructure the model to ensure BN follows conv, or let the op fall back to portable ops (which happens automatically via the partitioner).

### Missing Operators

Common unsupported ops that fall back to portable:

| Operator | Status | Notes |
|----------|--------|-------|
| `aten::native_dropout.out` | Not supported | Dropout should be disabled in eval mode [Source: #1287] |
| `aten::unfold.default` | Not in Aten Canonical | Requires decomposition [Source: #5381] |
| `torch.mm` with two dynamic inputs | Not delegated | XNNPACK requires at least one constant weight tensor for matmul [Source: #10297] |
| `torch.topk` | Partial | May fail to allocate temp memory [Source: #8700] |

### Quantized Ops Not Lowered

**Symptom**: `RuntimeError: Missing out variants: {'quantized_decomposed::dequantize_per_tensor', ...}`

**Cause**: Quantized ops (Q/DQ patterns) from `XnnpackQuantizer` were not delegated to XNNPACK. Running them on portable ops triggers this error. [Source: #1263, #7775, #8369]

**Fix**: Ensure the model is partitioned with `XnnpackPartitioner()` after quantization. Use `to_edge_transform_and_lower` which handles this correctly:

```python
et_program = to_edge_transform_and_lower(
    exported_program,
    partitioner=[XnnpackPartitioner()]
)
```

### Static Slice Quantization Mismatch

**Symptom**: `Failed to create static slice node with code: xnn_status_invalid_parameter`

**Cause**: Mismatching zero point quantization parameter across input and output of a slice operation. [Source: #12271]

**Fix**: Exclude the slice operation from quantization by configuring the `XnnpackQuantizer` with per-operator configs, or build ExecuTorch in Debug mode to see the detailed error message revealing the exact mismatch.

### YOLO Models

**Symptom**: Export errors like `'float' object has no attribute 'node'` when exporting Ultralytics YOLO models.

**Cause**: Ultralytics model wrappers introduce Python constructs that are not `torch.export`-compatible. [Source: #14644]

**Fix**: Access the inner model (`model.model`) and ensure eval mode. Use `strict=False` in `torch.export.export()` if needed:

```python
yolo_model = YOLO("yolo11n.pt").model
yolo_model.eval()
exported = torch.export.export(yolo_model, sample_inputs, strict=False)
```

### Unbacked SymInts Block Delegation

**Symptom**: Operators not being delegated to XNNPACK despite being supported. Log shows `arg tensor has free unbacked symbols or numel == 0`.

**Cause**: Data-dependent shapes (e.g., from masking operations) introduce unbacked symbolic integers that XNNPACK cannot handle. [Source: #14987]

**Workaround**: Restructure model code to avoid data-dependent views/reshapes, or accept that those subgraphs will run on portable ops.

### Const Tensors with Non-Default Dim Order

**Symptom**: `xnn_status_invalid_parameter` at runtime. Model partitions successfully but fails during execution with `Internal Error: Propagating input`.

**Cause**: Constant tensors with non-default dim order (e.g., from a `permute` call) are consumed by the XNNPACK partitioner but fail the runtime parameter validation. [Source: #14735]

**Workaround**: Currently none — this is an open bug. Affected models include YOLO11 when quantized and exported.

### Matrix Multiply Weight Not Recognized as Parameter

**Symptom**: `torch.mm` or matmul ops not delegated to XNNPACK despite being supported, causing significant slowdown (e.g., 50% of inference time in Whisper models).

**Cause**: The partitioner checks if the weight tensor is a model parameter. Computed intermediate tensors used as weights are not recognized, so the op falls back to portable. [Source: #15914]

**Workaround**: Enable internal debug logs in the partitioner to diagnose. A fix is being worked on to expose better controls for this check.

### Dynamic Quantization Requires per_channel=True

**Symptom**: `XnnpackBackend init failed` when running a dynamically quantized model.

**Cause**: XNNPACK dynamic quantization requires per-channel quantization. Using `get_symmetric_quantization_config(is_dynamic=True)` without `per_channel=True` produces a model that fails at runtime. [Source: #8830]

**Fix**:
```python
get_symmetric_quantization_config(is_dynamic=True, per_channel=True)
```

### Non-Contiguous Input Tensors Produce Silent Wrong Results

**Symptom**: Model produces incorrect outputs with no error or warning.

**Cause**: `Method.execute()` ignores tensor strides and reads `data_ptr` as if the tensor were contiguous. Non-contiguous tensors (e.g., from slicing, transposing) are misinterpreted. [Source: #18562]

**Fix**: Always call `.contiguous()` on input tensors before passing to ExecuTorch:
```python
input_tensor = input_tensor.contiguous()
```

### LSTM Dynamic-Shape Export Fails

**Symptom**: `Attempted to resize a static tensor` when using `register_lstm_while_loop_decomposition` with `to_edge_transform_and_lower()`.

**Cause**: Using `to_edge_transform_and_lower()` outside the LSTM decomposition context manager causes all symbolic shapes to vanish. Using it inside the context manager also fails due to shape propagation issues. [Source: #18487]

**Status**: Under investigation. The interaction between LSTM decomposition and XNNPACK's static shape requirements is being worked on.

### Partitioner Reorders Graph Inputs

**Symptom**: Backend receives inputs in a different order than the original exported program (e.g., `[input_ids, attention_mask]` becomes `[attention_mask, input_ids]`).

**Cause**: `fuse_as_graphmodule` in the partitioner can change input ordering during graph fusion. Root cause is in `torch.export`'s `fuse_conv_bn` and related passes. [Source: #14741]

**Status**: Open issue. Workaround is to track the input reordering in backend preprocessing.

## Platform-Specific Issues

### MediaTek SoC SIGSEGV in Weights Cache

**Symptom**: `SIGSEGV` crash in `XNNWeightsCache::look_up_or_insert` during `memcmp` on MediaTek Dimensity 6100+ (Samsung Galaxy M15). [Source: #17669]

**Details**: The crash occurs during weight cache lookup, specifically in `memcmp` when comparing packed weight data. This appears to be a **memory alignment issue** specific to certain MediaTek SoCs. The same model works on other MediaTek devices (Helio G99).

**Status**: Under investigation. As a workaround, disable weights caching if possible.

### iOS KleidAI/SME Crash

**Symptom**: Crash at `kai_get_sme_vector_length_u32` when loading a model on iOS.

**Cause**: KleidAI kernels use ARM SME instructions not available on all iOS devices. [Source: #17482]

**Fix**: Explicitly enable KleidAI with the `-DENABLE_XNNPACK_KLEIDI` CMake flag, or use the non-KleidAI XNNPACK build. Confirmed working on iPhone 16 Pro and iPhone 15 Pro. [Source: #17482]

### iOS SwiftPM Error 32 (NotFound)

**Symptom**: `Error 32 (NotFound)` when loading the "forward" method from exported PTE models via SwiftPM binary distribution.

**Cause**: Custom ops (e.g., `llama::custom_sdpa.out`) are not registered in the SwiftPM binary. The `-all_load` linker flag can cause 88 duplicate symbols. [Source: #14809]

**Fix**: Use ExecuTorch v1.0+ and build from source with proper custom op registration. The user confirmed it working with v1.0 after resolving custom op registration.

### Pthreadpool OOB When Reducing Thread Count

**Symptom**: Out-of-bounds read (ASan) or native crash when reducing threadpool size via `_unsafe_reset_threadpool` on macOS.

**Cause**: Version mismatch between libtorch's pthreadpool and ExecuTorch's pthreadpool can cause ODR violations when both libraries are loaded. [Source: #14321]

**Fix**: Use PR #14838 for a minimal workaround. The underlying issue is different pthreadpool commits in libtorch vs ET.

### AArch64 _Float16 Build Failure

**Symptom**: `error: '_Float16' is not supported on this target` when building on AArch64 Linux. [Source: #6844, #8924]

**Fix**: Use a newer compiler (GCC 12+) or Clang that supports `_Float16` on ARM targets.

### Android armv8.2-a Build Error

**Symptom**: `unsupported architecture 'armv8.2-a+dotprod+fp16'` when building optimized kernels for Android. [Source: #8508]

**Fix**: Ensure NDK version is r25+ which supports these architecture extensions.

### AVX-512 Intrinsic Errors on Older GCC

**Symptom**: `implicit declaration of function '_mm_loadu_si64'` when building XNNPACK on x86 with older GCC. Error comes from `qs8-vpreluc/gen/qs8-vpreluc-avx2-u16.c`. [Source: #12817]

**Cause**: Not an AVX-512 requirement — it's a GCC bug where `_mm_loadu_si64` / `_mm_storeu_si64` intrinsics are not declared even in GCC 12.4.0. See GCC bug #78782.

**Workaround**: Add compiler flags to redefine the intrinsics:
```bash
CFLAGS="-D_mm_loadu_si64=_mm_loadl_epi64 -D_mm_storeu_si64=_mm_storel_epi64"
```

## Threading and Workspace Issues

### Workspace Lock in Disabled Mode

**Symptom**: XNNPACK acquires `XNNWorkspace` mutex even when `WorkspaceSharingMode::Disabled` is set, causing blocking in real-time audio callbacks. [Source: #17301]

**Details**: When workspace sharing is disabled, each delegate instance creates its own workspace, but the `execute()` path still acquires the global lock.

**Fix**: A patch is available in the issue (skip lock acquisition when workspace sharing is disabled).

### Thread Count Not Set

**Symptom**: XNNPACK inference unexpectedly slow on multi-core devices.

**Cause**: Default thread count may be 1. [Source: #10297]

**Fix**:
```cpp
#include <executorch/extension/threadpool/threadpool.h>
torch::executorch::threadpool::get_threadpool()->set_num_threads(4);
```

## Dynamic Shape Handling

XNNPACK does **not** support dynamic shapes within delegated subgraphs. [Source: #3636, #8539]

**Symptoms**:
- `Attempted to resize a static tensor to a new shape at dimension 0` [Source: #1350]
- `Symbol undefined error in to_out_var_pass by inputs with dynamic dims` [Source: #8539]

**Workarounds**:
1. Use fixed input shapes and pad inputs to a maximum size
2. Remove XNNPACK delegation for dynamic-shape subgraphs (they fall back to portable ops)
3. Use multiple entry points with different fixed shapes

## Convolution Issues After Save/Load

**Symptom**: XNNPACK fails on convolution operations after `export.save()` -> `export.load()` cycle. [Source: #5265]

**Cause**: Serialization/deserialization of exported programs can alter tensor metadata that XNNPACK relies on.

**Fix**: Apply XNNPACK partitioning after loading the saved program, not before saving.

## Multi-Entry Point Issues

### No Shared Mutable State

Weight sharing across entry points works by default, but **mutable state** (buffers like hidden states) cannot be shared across entry points. [Source: #11738, #12804]

**Current state**: Shared constant weights across methods are supported and enabled by default. Shared mutable state is under active development. [Source: #12804]

## SDK and Profiling

### Inconsistent Time Format in ETDump

**Symptom**: Time metrics from XNNPACK delegate in ETDump use different units than non-delegated ops. [Source: #4504]

**Note**: When profiling XNNPACK-delegated models, be aware that delegate-level timing may not break down individual op times within the delegate blob.

### ETDump Generation Fails with XNNPACK Delegation

**Symptom**: Bundled program file (.bp) generated from an XNNPACK-delegated model outputs "Terminated" when executed — no error message or stack trace. [Source: #8177]

**Fix**: Use `to_edge_transform_and_lower()` (not the older API) and build in debug mode to get crash logs. The ETDump interaction with different API surfaces is being investigated.

## Configuration Complexity

The XNNPACK partitioner has many configuration options that can be difficult to get right. Common mistakes include: [Source: #8884]

- Not configuring per-op quantization settings
- Missing operator configs for specific operator patterns
- Not handling operator-specific constraints (e.g., channel alignment)

**Recommendation**: Start with the default `XnnpackPartitioner()` configuration and only customize when needed.

## Linking and Registration Issues

### Backend XnnpackBackend is Not Registered

**Symptom**: `Backend XnnpackBackend is not registered` at runtime.

**Cause**: The XNNPACK backend library is not properly linked. Common when using separate build trees or pre-built binaries. [Source: #3696, #8196]

**Fix**:
1. Set `EXECUTORCH_BUILD_XNNPACK=ON` in CMake
2. Link with `target_link_libraries(your_runner PRIVATE xnnpack_backend)`
3. If using pre-built static libraries, use `--whole-archive` to force static initialization: [Source: #10602]
```cmake
target_link_libraries(my_app
    -Wl,--whole-archive libvulkan_backend.a -Wl,--no-whole-archive
)
```

### libtorch XNNPACK Conflict

**Symptom**: Unexpected behavior when both ExecuTorch and libtorch are linked.

**Cause**: libtorch brings its own XNNPACK dependency with a global struct for initialization state. When both are loaded, the global state from libtorch's XNNPACK dep can incorrectly interfere with ExecuTorch's. [Source: #3696]

**Workaround**: Avoid linking both libtorch and ExecuTorch's XNNPACK in the same binary.

## Quantization Platform Differences

### Different Results on Intel vs ARM with Quantization

**Symptom**: Quantized model produces large loss on ARM (e.g., Raspberry Pi) while converging to zero on Intel x86.

**Cause**: `XnnpackQuantizer` with `get_symmetric_quantization_config()` (static quantization) can introduce platform-specific numerical differences due to architecture-specific quantization kernel implementations. [Source: #16406]

**Fix**: Remove global symmetric quantization (`quantizer.set_global(get_symmetric_quantization_config())`) if cross-platform numerical consistency is required. The model works correctly on both platforms without quantization.
