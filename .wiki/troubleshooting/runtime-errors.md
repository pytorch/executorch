---
title: "Runtime Errors Troubleshooting"
category: DEBUGGING
backends: []
last_validated: 2026-04-15
source_issues: [10297, 10451, 10549, 10226, 10179, 1020, 11050, 1340, 10188, 18573, 18832, 3515, 3528, 1350]
---

# Runtime Errors Troubleshooting

## Missing Operator Errors

### Symptom: Op Not Registered

If an operator is not included in the build, you'll get a runtime error. Use selective build to include exactly the ops your model needs:

```cmake
gen_selected_ops(
  LIB_NAME "my_ops"
  ROOT_OPS "aten::add.out;aten::mul.out"
  INCLUDE_ALL_OPS "OFF"
)
```

Or use `INCLUDE_ALL_OPS "ON"` during development to include everything. [Source: #10297]

### Ops Falling Through to CPU

When a backend doesn't support an op, it "falls through" to the portable CPU kernel. This is silent (no error) but can cause significant performance degradation.

**How to detect:**
```python
from executorch.devtools import Inspector
inspector = Inspector(etdump_path="./etdump.etdp")
inspector.print_data_tabular()
# Look at occurrences_in_non_delegated_graphs column
```

**Common ops that fall through:**
| Op | Reason | Backend |
|----|--------|---------|
| `aten.mm.default` | Both inputs dynamic | XNNPACK |
| `aten.native_layer_norm` | Not supported | XNNPACK |
| `aten.embedding` | Integer inputs | XNNPACK |
| `aten.batch_norm` | Only supported after conv (fusion) | XNNPACK |
| `_to_dim_order_copy` | Not recognized | CoreML, XNNPACK |
| BF16 ops | Only fp16/fp32 supported | XNNPACK |

[Source: #10297, #10451, #1340, #10188]

### "Failed to load method: error 20" (Missing Operators)

This error means the PTE file requires operators not included in the build. Common on embedded platforms (RISC-V, Cortex-M). [Source: #18573]

**Fix:** Use `EXECUTORCH_SELECT_OPS_MODEL` instead of manually listing ops -- it auto-detects all required ops from the .pte file:
```cmake
set(EXECUTORCH_SELECT_OPS_MODEL path/to/model.pte)
```

### LLM Native Runner: Missing custom_ops_aot_lib or llama::update_cache

**Error (first)**:
```
AssertionError: Expected 1 library but got 0
```
at `extension/llm/custom_ops/custom_ops.py` when importing `sdpa_with_kv_cache`.

**Error (after building custom_ops)**:
```
kernel 'llama::update_cache.out' not found
```

**Cause**: The Python-based native LLM runner (`python -m executorch.examples.models.llama.runner.native`) requires `libcustom_ops_aot_lib.so` to be built and placed in `extension/llm/custom_ops/`. Building from source does not always produce this library by default. [Source: #18832]

**Fix**: Build with the pybind preset which enables all required LLM AOT libraries:
```bash
cmake -B cmake-out \
  -DEXECUTORCH_BUILD_KERNELS_LLM_AOT=ON \
  -DEXECUTORCH_BUILD_EXTENSION_CUSTOM_OPS=ON \
  -DEXECUTORCH_BUILD_EXTENSION_LLM=ON
cmake --build cmake-out --target custom_ops_aot_lib -j
cp cmake-out/executorch/extension/llm/custom_ops/libcustom_ops_aot_lib.so extension/llm/custom_ops/
```

Or use the pybind CMake preset which handles this automatically:
```bash
cmake --preset pybind
cmake --build --preset pybind -j
```

**Note**: The Android/C++ runner path (via `examples/models/llama/README.md`) does not have this issue — it links the custom ops statically. [Source: #18832]

### "Overriding output data pointer allocated by memory plan is not allowed"

This error/warning occurs when you try to set output buffers for tensors that are already memory-planned. It is typically NOT fatal -- the model may still execute correctly. [Source: #3515, #3528]

**Fix:** Check `method_meta` to determine which tensors are memory-planned before attempting to set data pointers.

### Static Tensor Resize Error

**Error:** `Attempted to resize a static tensor to a new shape at dimension 0`

ExecuTorch plans memory at export time. Runtime inputs must match the shapes used during export. [Source: #1350]

**Fix:** Use consistent input shapes or export with dynamic shapes support.

## Memory Issues

### MPS Backend: Metal Device Initialization Failure

**Error:**
```
assert failed: _mtl_device != nil
```

**Causes:**
1. Running on Intel Mac (MPS requires Apple Silicon or AMD GPU) [Source: #1020]
2. Running in a headless/SSH environment without GPU access

**Fix:** Ensure you're on Apple Silicon Mac with macOS Sonoma+, Xcode 15+. [Source: #1020]

### Model Loading: MmapUseMlockIgnoreErrors

For large models, use memory-mapped loading to avoid OOM:

```cpp
Module model(model_path, Module::LoadMode::MmapUseMlockIgnoreErrors);
```

This memory-maps the model file and ignores mlock failures (which happen when the model is larger than available RAM for locking). [Source: #10297]

## Model Loading Failures

### Delegate Errors at Runtime

If a model exports successfully but crashes at runtime, common causes include:

1. **Scalar vs Tensor mismatch:** CoreML wraps scalar inputs as rank-1 tensors at compile time, but ExecuTorch passes them as scalars at runtime. This causes shape mismatch errors. [Source: #10451]

2. **Missing backend library:** Ensure the backend library (e.g., `xnnpack_backend`, `coreml_backend`) is linked to your application. [Source: #10297]

3. **Version mismatch:** PTE files are not guaranteed backward-compatible. Ensure the runtime version matches the export version.

### CoreML Runtime Crash After Successful Export

**Symptom:** Model exports without error but crashes during inference.

**Root cause:** `_to_dim_order_copy` ops cause CoreML to receive scalar inputs that it wraps as rank-1 tensors at compile time, but ExecuTorch passes the original scalars at runtime. [Source: #10451]

**Workaround:** Disable dim order or use the older `to_backend()` API (which uses `_to_copy` instead of `_to_dim_order_copy`). [Source: #10451]

## Performance-Related Runtime Issues

### Inference Slower Than Expected

If ExecuTorch inference is slower than PyTorch on the same hardware:

1. **Check build type:** Must be `Release`, not `Debug` [Source: #10297]
2. **Check thread count:** Set explicitly for XNNPACK:
   ```cpp
   #include <executorch/extension/threadpool/threadpool.h>
   ::executorch::extension::threadpool::get_threadpool()
       ->_unsafe_reset_threadpool(4);
   ```
   [Source: #10297]
3. **Check delegation rate:** If most ops are non-delegated, performance will be poor. Use profiling to identify bottlenecks. [Source: #10297]
4. **Check for non-delegated mm ops:** `aten.mm.default` between two dynamic tensors is NOT delegated by XNNPACK. Replace with `aten.mul` if applicable, or use quantization. [Source: #10297]

### cpuinfo Misreporting Core Count

`executorch::extension::cpuinfo::get_num_performant_cores()` may report all cores instead of just performance cores on some devices (iPhone 14, Pixel 8). This can lead to suboptimal thread scheduling. [Source: #10549]

### ETDump Profiling Warning

**Warning:** `No delegate mapping found for delegate with instruction id`

This means the profiler cannot map delegate instructions back to the original graph. The profiling data for delegated ops may be incomplete, but non-delegated op timings are still accurate. [Source: #10297]

## Debugging Methodology

### Step 1: Profile

```cpp
// Enable ETDump tracing
auto etdump_gen = std::make_unique<ETDumpGen>();
Module model(model_path, load_mode, std::move(etdump_gen));
// ... run inference ...
auto result = etdump_gen->get_etdump_data();
// Write to file
```

### Step 2: Analyze

```python
from executorch.devtools import Inspector
inspector = Inspector(etdump_path="./etdump.etdp")
inspector.print_data_tabular()
```

### Step 3: Identify Bottlenecks

Look for:
- Non-delegated ops with high execution time
- Memory copy operations (`_to_dim_order_copy`, `expand_copy`)
- Repeated small kernel launches (overhead-dominated)

### Step 4: Advanced Profiling

For deeper analysis, use system-level profiling:
- **FlameGraph:** `perf record` + FlameGraph tools for CPU-level call stack analysis [Source: #10297]
- **Runtime profiling docs:** Search for "runtime profiling" in ExecuTorch documentation [Source: #10297]
