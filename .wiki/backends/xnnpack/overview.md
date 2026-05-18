---
title: "XNNPACK Backend Overview"
category: BACKEND_CONSTRAINT
backends: [XNNPACK]
last_validated: 2026-04-05
source_issues: [1231, 1330, 1340, 3497, 3586, 3636, 3696, 3919, 4005, 4873, 5068, 5265, 8476, 8558, 8830, 8884, 8932, 9027, 10066, 10297, 10663, 11523, 11738, 12134, 12248, 12804, 13629, 13732, 13787, 14221, 14644, 14987, 15914, 16123, 17301, 17669]
---

# XNNPACK Backend Overview

## What Is XNNPACK

XNNPACK is a highly optimized neural network inference library developed by Google that serves as the primary CPU backend for ExecuTorch. It accelerates floating-point and quantized (int8) inference on ARM (NEON), x86 (SSE/AVX), and WebAssembly (SIMD) architectures. [Source: #3497]

Within ExecuTorch, XNNPACK works as a **delegate backend**: during model export, supported operators are partitioned and lowered into XNNPACK subgraphs that execute as opaque delegate blobs at runtime.

## Supported Platforms

| Platform | Status | Notes |
|----------|--------|-------|
| Android (arm64-v8a) | Fully supported | Primary target, CI-tested |
| iOS (arm64) | Fully supported (as of ExecuTorch v1.0+) | Via SwiftPM or CMake; KleidAI kernels may need explicit flag [Source: #17482] |
| Linux (x86_64, aarch64) | Supported | Used in development; aarch64 may need `cstdint` fix on older compilers [Source: #6844] |
| macOS (Apple Silicon) | Supported | Included in `pip install executorch` since v0.6 [Source: #10066] |
| Windows | Partial | Native builds possible but not CI-covered |
| WebAssembly | Theoretically possible | XNNPACK supports wasm-simd; no official ET integration yet [Source: #3497, #8216] |

## Key Capabilities

- **FP32 and INT8 quantized inference**: Full support for PT2E quantization flow with `XnnpackQuantizer` [Source: #1330]
- **Operator fusion**: Conv+BN fusion, quantized operator fusion (Q/DQ patterns lowered directly) [Source: #1230, #1340]
- **Multi-threaded execution**: Threadpool-based parallelism; set thread count explicitly for best performance [Source: #10297, #8932]
- **Weight sharing across entry points**: Constant weights are shared across multiple methods in a single PTE file [Source: #12804]
- **Weights cache**: Packing and caching weights for repeated inference; uses `memcmp` for cache lookup [Source: #17669]
- **Sparse kernels (experimental)**: XNNPACK has `XNN_ENABLE_SPARSE` for SpMM (sparse matrix multiply), but this is not exposed as a stable API in ExecuTorch. SpMM is auto-invoked by conv2d when build/runtime conditions are met (NCHW-compatible, non-quantized). [Source: #13787]
- **Dynamic weight update**: Supported only for linear layers via a slower kernel path (no packed weights). Requires setting a flag on `XnnpackPartitioner` — see `xnnpack_config.py:ConfigPrecisionType`. Not supported on QNN. [Source: #16123]

## Export Flow

The recommended export flow uses `to_edge_transform_and_lower`:

```python
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower

exported = torch.export.export(model, example_inputs)
et_program = to_edge_transform_and_lower(
    exported,
    partitioner=[XnnpackPartitioner()]
)
```

The older `to_edge().to_backend()` flow still works but `to_edge_transform_and_lower` is preferred for better optimization. [Source: #10297]

## Quantization

Use the `XnnpackQuantizer` for PT2E quantization:

```python
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    XnnpackQuantizer,
    get_symmetric_quantization_config,
)

quantizer = XnnpackQuantizer()
quantizer.set_global(get_symmetric_quantization_config())
```

**Important**: Quantized ops (Q/DQ nodes) should be lowered to XNNPACK — running them on portable ops will be extremely slow. If you see `Missing out variants` errors for `quantized_decomposed` ops, it usually means the model was not properly delegated. [Source: #1263, #7775, #8369]

**Dynamic quantization** must use `per_channel=True`:
```python
get_symmetric_quantization_config(is_dynamic=True, per_channel=True)
```
Without `per_channel=True`, dynamic quantization will fail at runtime with `XnnpackBackend init failed`. [Source: #8830]

**Calibration required**: When using dynamic quantization, you must calibrate the model by running sample inputs through the prepared graph before exporting. Skipping calibration causes `Failed loading of method forward` at runtime. [Source: #11355]

**HuggingFace models with `padding_idx`**: Embeddings with `padding_idx` are not recognized by `XnnpackQuantizer`'s quant patterns, causing `Missing out variants: {'torchao::dequantize_affine'}`. Strip `padding_idx` with a custom pass (e.g., `RemoveEmbeddingPaddingIdxPass` that replaces `aten.embedding.default` with a version without `padding_idx`). [Source: #10663]

## Performance Baseline

Without XNNPACK delegation, ExecuTorch runs on portable ops which are **not optimized for performance** — inference can be 10-100x slower. Always delegate to XNNPACK for CPU inference. [Source: #1231, #3919]

For competitive performance with PyTorch Mobile:
1. Build with **Release** mode (CMake `-DCMAKE_BUILD_TYPE=Release`) [Source: #4005]
2. Set thread count: `torch::executorch::threadpool::get_threadpool()->set_num_threads(4)` [Source: #10297]
3. Ensure the model is actually delegated (check that nodes appear under XNNPACK delegate in the exported program) [Source: #10297]

## Limitations

- **No dynamic shape support within delegate**: XNNPACK subgraphs require static tensor shapes. Dynamic shapes cause fallback to portable ops. [Source: #3636, #8539]
- **Batch norm only fused with conv**: Standalone `batch_norm` is not supported; it must follow a `conv` for fusion [Source: #1340]
- **Tensors limited to rank <= 4 or 5**: Higher-dimensional tensors may not be supported for all ops [Source: #15441]
- **No shared mutable state across entry points**: Weight sharing works, but shared mutable buffers (hidden states) across multiple entry points are not yet supported [Source: #11738]
- **NHWC layout considerations**: Some ops require NHWC (channel-last) layout for optimal performance; dim order tagging in the partitioner is evolving [Source: #4873, #8476]
- **`.module()` not sound after `to_executorch()`**: Calling `.module()` on the program after `to_executorch()` may produce wrong results due to internal invariant violations. Use `.module()` only after `to_edge()`, or run the model through the ET runtime after `to_executorch()`. [Source: #5068]
- **`torch.mm` weight must be recognized as parameter**: If a matmul's weight tensor is not recognized as a model parameter (e.g., computed intermediates in Whisper-like architectures), it won't be delegated. This can cause 50%+ inference slowdown. Enable internal debug logs in the partitioner to diagnose. [Source: #15914]
- **Non-contiguous input tensors silently produce wrong results**: `Method.execute()` ignores tensor strides and reads `data_ptr` as contiguous. Always call `.contiguous()` on inputs before passing to ExecuTorch. [Source: #18562]

## Multi-Backend Usage

XNNPACK is commonly used as a CPU fallback alongside GPU backends:

- **XNNPACK + CoreML**: CoreML typically consumes the whole graph on iOS; little left for XNNPACK [Source: #13732]
- **XNNPACK + Vulkan**: Use `partitioner=[VulkanPartitioner(), XnnpackPartitioner()]` for GPU-first with CPU fallback [Source: #15441]
- **XNNPACK + QNN**: For heterogeneous execution on Qualcomm devices [Source: #13629]

## Building

### Python Export (pip install)

```bash
pip install executorch  # Includes XNNPACK export support since v0.6
```

### C++ Runtime (CMake)

```bash
cmake -DEXECUTORCH_BUILD_XNNPACK=ON \
      -DCMAKE_BUILD_TYPE=Release \
      ...
cmake --build cmake-out -j$(nproc)
```

### Android

```bash
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
      -DANDROID_ABI=arm64-v8a \
      -DEXECUTORCH_BUILD_XNNPACK=ON \
      ...
```

### Debug Logging

Set XNNPACK to debug mode for detailed operator-level logs:

In `backends/xnnpack/cmake/Dependencies.cmake`, change the XNNPACK build type to Debug. This reveals parameter validation failures (e.g., quantization zero point mismatches). [Source: #1330, #12271]
