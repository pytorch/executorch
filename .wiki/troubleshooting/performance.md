---
title: "Performance Troubleshooting"
category: PERFORMANCE
backends: []
last_validated: 2026-04-05
source_issues: [10297, 10549, 10226, 11034, 10451, 10188, 1340]
---

# Performance Troubleshooting

## Profiling Methodology

### ETDump Operator-Level Profiling

The primary profiling tool for ExecuTorch is ETDump, which captures per-operator timing:

**C++ Runtime Setup:**
```cpp
#include <executorch/devtools/etdump/etdump_flatcc.h>

auto etdump_gen = std::make_unique<ETDumpGen>();
Module model(model_path, Module::LoadMode::MmapUseMlockIgnoreErrors,
             std::move(etdump_gen));

// Run inference
auto result = model.forward(inputs);

// Save trace
ETDumpGen* gen = static_cast<ETDumpGen*>(model.event_tracer());
ETDumpResult dump = gen->get_etdump_data();
FILE* f = fopen("etdump.etdp", "w+");
fwrite((uint8_t*)dump.buf, 1, dump.size, f);
fclose(f);
free(dump.buf);
```

**Python Analysis:**
```python
from executorch.devtools import Inspector
inspector = Inspector(etdump_path="./etdump.etdp")
inspector.print_data_tabular()
```

The output shows per-op timing, delegation status, and occurrence counts. [Source: #10297]

### FlameGraph for System-Level Profiling

For deeper CPU-level analysis, use `perf` + FlameGraph:

```bash
# Record
perf record -g -F 99 ./my_executorch_app
# Generate flamegraph
perf script | stackcollapse-perf.pl | flamegraph.pl > flamegraph.svg
```

FlameGraphs help identify:
- Whether the bottleneck is in XNNPACK kernels or portable kernels
- Thread scheduling issues (GEMM vs GEMV kernel selection)
- Memory allocation overhead

See https://github.com/brendangregg/FlameGraph [Source: #10297]

### Profiling Docs

Search for "runtime profiling" in the ExecuTorch documentation for the full profiling guide. [Source: #10297]

## Common Performance Bottlenecks

### 1. Non-Delegated Matrix Multiplication

**Impact:** Can account for 68%+ of inference time [Source: #10297]

`aten.mm.default` with two dynamic inputs is NOT delegated by XNNPACK. XNNPACK requires at least one input to be a constant (weight) tensor.

**Detection:** In profiling output, look for `aten_mm_default` in `occurrences_in_non_delegated_graphs`.

**Fixes:**
- If one operand is constant, reshape the model to use `aten.linear` instead
- If both operands are truly dynamic, consider replacing `mm` with `mul` for element-wise operations where applicable [Source: #10297]
- Quantization can also help by making ops more delegation-friendly

### 2. Debug Build Instead of Release

**Impact:** 5-10x slowdown

```bash
# WRONG
cmake .. -DCMAKE_BUILD_TYPE=Debug

# CORRECT
cmake .. -DCMAKE_BUILD_TYPE=Release
```
[Source: #10297]

### 3. Incorrect Thread Count

**Impact:** 2-4x slowdown on multi-core devices

XNNPACK defaults to single-threaded execution unless explicitly configured:

```cpp
#include <executorch/extension/threadpool/threadpool.h>
::executorch::extension::threadpool::get_threadpool()
    ->_unsafe_reset_threadpool(4);
```

Note: `cpuinfo::get_num_performant_cores()` may misreport on some devices (iPhone 14, Pixel 8), so consider hardcoding the thread count for known devices. [Source: #10549]

**Android (Java/Kotlin):** Thread count control via Java API is not yet fully documented. Use native JNI calls to set thread count. [Source: #10297]

### 4. BF16 Ops Not Delegated

XNNPACK only supports fp16 and fp32 activations. BF16 (`-d bf16`) causes all linear ops to fall through to CPU, losing all XNNPACK acceleration. [Source: #10188]

**Fix:** Use fp16 instead of bf16 for XNNPACK delegation. BF16 dynamically-quantized delegation is being developed. The portable + optimized kernel libraries do support bf16 for CPU-only execution. [Source: #10188]

### 5. Non-Delegated layer_norm

`native_layer_norm` is not delegated by XNNPACK but appears frequently in LLM architectures.

**Fix:** Enable optimized kernels:
```cmake
option(EXECUTORCH_BUILD_KERNELS_OPTIMIZED "" ON)
```
Link `optimized_native_cpu_ops_lib` to your application. This accelerates CPU-side execution of non-delegated ops. [Source: #10297]

### 6. Standalone batch_norm Not Delegated

XNNPACK only delegates `batch_norm` when it follows a convolution (conv+bn fusion). Standalone `batch_norm` falls through to CPU. [Source: #1340]

### 7. Static Shape Export Missing GEMM Optimization

When exporting with `--disable_dynamic_shape`, the model uses a single static shape. This means:
- Prefill (batch>1) uses GEMV instead of GEMM, losing parallelism
- TorchScript can dynamically switch between GEMM/GEMV based on input shape, but `torch.export` cannot

**Workaround:** Use separate PTE files for prefill and decode with different batch sizes. [Source: #10297]

## Backend-Agnostic Optimization Tips

### Use to_edge_transform_and_lower

Always use `to_edge_transform_and_lower()` instead of the older `to_edge()` + `to_backend()` pipeline. The newer API applies important graph optimizations. In one benchmark, switching from `to_backend()` to `to_edge_transform_and_lower()` reduced inference time from 27s to 16s. [Source: #10297]

### Selective Build

Only include the operators your model actually needs:
```cmake
gen_selected_ops(LIB_NAME "my_ops" ROOT_OPS "..." INCLUDE_ALL_OPS "OFF")
```

This reduces binary size and can improve load time. [Source: #10297]

### Check Delegation Rate

After lowering, verify what percentage of ops are delegated:

```python
# Print delegation statistics
edge = to_edge_transform_and_lower(
    exported_program, partitioner=[XnnpackPartitioner()])
print(edge.exported_program().graph_module)
```

A high delegation rate (>80% of compute-heavy ops) is needed for good performance. [Source: #10297]

## Benchmarking Best Practices

1. **Always use Release builds** for benchmarking
2. **Warm up:** Run 2-3 warmup iterations before measuring
3. **Measure end-to-end:** Include model loading, inference, and output processing in timing
4. **Compare apples-to-apples:** When comparing with PyTorch, use the same threading configuration and hardware
5. **Use ETDump for operator breakdown:** Overall timing can be misleading; per-op timing reveals the actual bottleneck
6. **Watch for misleading profiling:** ETDump's `execute` time may only show one forward pass, while your measured time includes the full generation loop (multiple forward passes for LLMs) [Source: #10297]

## Expected Performance Characteristics

ExecuTorch should generally be faster than PyTorch on edge devices because:
- Smaller runtime overhead
- Backend delegation (XNNPACK, CoreML, QNN) uses hardware-optimized kernels
- Memory-mapped model loading

If ExecuTorch is slower than PyTorch on CPU, investigate:
1. Delegation rate (many non-delegated ops?)
2. Build configuration (Release mode?)
3. Thread count (matches PyTorch?)
4. Dynamic vs static shapes (losing GEMM?)

[Source: #10297]
