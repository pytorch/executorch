# PTE Generation to Runtime Kernel Selection Pipeline

## Overview

This document traces the complete pipeline from a PyTorch model to runtime kernel execution on Cadence Vision DSP, using `resnet18_quantized.pte` as a concrete example.

---

## Stage 1: PTE Generation (Python, host machine)

```
PyTorch ResNet18 model
        │
        ▼
① torch.export.export()          ── Traces model → ExportedProgram (ATen IR)
        │                            Ops: aten.conv2d, aten.relu, aten.linear, etc.
        ▼
② prepare_pt2() + convert_pt2()  ── Quantizes using CadenceDefaultQuantizer
        │                            Ops: quantized_decomposed.quantize_per_tensor, etc.
        ▼
③ fuse_pt2()                     ── Fuses quantized patterns into Cadence custom ops
        │                            quantized_decomposed.* → cadence::quantized_conv2d_nchw
        │                            aten.relu → cadence::quantized_relu
        │
        │  ops_registrations.py provides @register_fake kernels
        │  so the compiler can infer output shapes/dtypes
        │
        ▼
④ _lower_ep_to_edge()            ── Converts to Edge IR (standardized op set)
        │
        ▼
⑤ apply_exir_ops_passes()        ── Cadence-specific graph optimizations
        │                            (op fusion, layout transforms, etc.)
        ▼
⑥ to_executorch()                ── Memory planning + serialization → PTE flatbuffer
```

### Key Files

| File | Role |
|------|------|
| `backends/cadence/aot/export_example.py` | Entry point: `export_model()` orchestrates the full pipeline |
| `backends/cadence/aot/compiler.py` | Core functions: `prepare_pt2()`, `convert_pt2()`, `fuse_pt2()`, `_lower_ep_to_cadence_gen_etrecord()` |
| `backends/cadence/aot/ops_registrations.py` | `@register_fake` kernels for shape/dtype inference during export |
| `backends/cadence/aot/quantizer/quantizer.py` | `CadenceDefaultQuantizer` — quantization configuration |

### What's IN the PTE (ResNet18 example)

From the exported graph log:

| Op | Count |
|----|-------|
| `cadence.quantized_conv2d_nchw.per_tensor_out` | 20 |
| `cadence.dequantize_per_tensor.out` | 18 |
| `cadence.quantized_relu.per_tensor_out` | 17 |
| `cadence.quantize_per_tensor.out` | 11 |
| `aten.add.out` | 8 |
| `aten.max_pool2d_with_indices.out` | 1 |
| `aten.mean.out` | 1 |
| `cadence.quantized_linear.per_tensor_out` | 1 |

The PTE stores **only op names + tensor metadata** (shapes, dtypes, quantization params). It knows nothing about SIMD, DMA, or which C++ function to call.

---

## Stage 2: Build Time (CMake, cross-compilation)

```
functions_vision.yaml
        │
        ▼
gen_selected_ops()              ── Extracts op names → selected_operators.yaml
generate_bindings_for_kernels() ── Generates C++ registration code:
        │
        │  RegisterCodegenUnboxedKernelsEverything.cpp:
        │    "cadence.quantized_linear.per_tensor_out"
        │       → impl::vision::quantized_linear_per_tensor_out()
        │    "aten.add.out"
        │       → impl::vision::add_out()
        │    "aten.mean.out"
        │       → impl::vision::mean_dim_out()
        │
        ▼
gen_operators_lib()             ── Compiles into cadence_ops_lib
        │
        ▼
executor_runner links cadence_ops_lib ── Final binary with kernel registry
```

### Key Files

| File | Role |
|------|------|
| `backends/cadence/aot/functions_vision.yaml` | Op dispatch table: maps op names → C++ kernel functions |
| `backends/cadence/vision/operators/CMakeLists.txt` | Invokes codegen to process the YAML |

### Namespace Dispatch (current state)

| Namespace | Count | Description |
|-----------|-------|-------------|
| `impl::vision::` | 22 ops | Hardware-optimized (SIMD, DMA) |
| `impl::generic::` | 38 ops | Generic Cadence fallback |
| `torch::executor::` | 29 ops | Portable reference implementations |

---

## Stage 3: Runtime Execution (DSP hardware)

```
executor_runner loads resnet18_quantized.pte
        │
        ▼
For each node in the graph, dispatcher looks up the op name
in the compiled kernel registry:

Example: "cadence.quantized_linear.per_tensor_out"
        │
        ▼  Registry lookup (from YAML → codegen)
        │
impl::vision::quantized_linear_per_tensor_out()   ← always called
        │
        ▼  Runtime decision #1 (dtype + dimension check)
        │
        ├─ src=int8, weight=int8, out=int8, in_dim≥16?
        │   YES → use_optimized = true (SIMD path)
        │   │
        │   ▼  Runtime decision #2 (buffer + size check)
        │   │
        │   ├─ in_dim≥512 && DRAM buffers available?
        │   │   YES → DMA tiling path (prefetch weight tiles into local DRAM)
        │   │         Uses: dma_2dm_init(), idma_copy_2d_desc(), rvdot_zeropt()
        │   │
        │   └─ NO  → Direct SIMD path (no DMA, process from system memory)
        │             Uses: rvdot_zeropt() only
        │
        └─ NO  → Generic fallback (scalar C++ loops)
                  Uses: quantized_linear_per_tensor_generic_<T>()
```

### Concrete Trace: ResNet18 Linear Layer

The PTE has 1 linear layer: `cadence.quantized_linear.per_tensor_out` with `in_dim=512`, `out_dim=1000`, dtype=int8.

At runtime:
1. **Registry** → `impl::vision::quantized_linear_per_tensor_out` (from YAML)
2. **int8 + in_dim=512 ≥ 16** → `use_optimized = true`
3. **in_dim=512 ≥ 512 + DRAM available** → **DMA tiling path**: weight matrix (512×1000) is too large for local memory, so it's loaded in tiles via iDMA while SIMD computes dot products using `rvdot_zeropt()`

---

## Decision Summary

| Layer | What Decides | When |
|-------|-------------|------|
| Op name in PTE | Python compiler passes (`fuse_pt2`) | PTE generation |
| Op → C++ function | `functions_vision.yaml` | Binary build time |
| SIMD vs generic | Tensor dtype + dimensions | Runtime |
| DMA vs no-DMA | Buffer availability + size thresholds | Runtime |

---

## Dual Registration: Python vs YAML

| Aspect | `functions_vision.yaml` | `ops_registrations.py` |
|--------|------------------------|----------------------|
| **When** | CMake build time | AOT export time |
| **Purpose** | Generate C++ runtime dispatcher code | Enable graph optimization & shape inference |
| **Compiled into** | executor_runner binary (runtime) | Not compiled — used during PTE generation only |
| **Usage** | Kernel dispatch at execution | Graph transformation during export |

These two must stay in sync — the op signatures defined in Python (for export) must match the kernel names registered via the YAML (for runtime).
