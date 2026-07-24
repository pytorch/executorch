# ExecuTorch WebGPU Backend

Run ExecuTorch models on the GPU through
[WebGPU](https://www.w3.org/TR/webgpu/). The backend compiles delegated
subgraphs into WGSL compute shaders and executes them through
[Dawn](https://dawn.googlesource.com/dawn) and its Tint WGSL compiler. The same
graph execution design targets Metal and Vulkan in native builds. Browser
builds compile the same execution path with Emscripten and `emdawnwebgpu`.

> **Status: under active development.** The backend has more than 100
> registered operator symbols and has demonstrated optimized browser artifacts
> for Llama and Qwen models.

## Performance

Measured in July 2026 on an Apple M4 Pro in Chrome Canary, using WebGPU
GPU-timestamp queries and a hash-pinned, optimized dynamic-shape 4-bit Llama
3.2 1B artifact. Results are medians from five measured runs after three
warmup runs.

| Metric | Result |
|---|---:|
| Prefill (1,024-token prompt) | 2,176 tokens/s |
| Decode (2,048-token context) | 151 tokens/s |

These figures describe the accepted hash-pinned artifact, not the performance
of every model or hardware configuration.

## Optimization Techniques

| Category | Optimizations |
|---|---|
| Prefill | Shared-memory tiled "Steel" GEMM with fp16 multiplication and fp32 accumulation, shape-routed kernel selection, scoped output suppression, and QKV fusion |
| Decode | Cooperative GEMV with bi-column batching, FlashDecoding with split KV, and register-tiled SDPA |
| SDPA | Register-tiled QK and AV, coalesced KV reads, causal-tile skipping, and aligned loads |
| Quantization | Packed-word dequantization, 4-bit weight-only kernels, dynamic 8da4w quantization, and int8 q8ta kernels |
| Memory | f16 KV cache and scratch-pool reuse |
| Dispatch | Two-dimensional folded dispatch, runtime-configurable workgroup sizes, and SwiGLU fusion |

## Demonstrated Model Artifacts

| Model | Validation status |
|---|---|
| Llama 3.2 1B | Browser-validated, hash-pinned optimized 4-bit artifact |
| Qwen2.5-0.5B | Demonstrated browser artifact with token match and numeric drift |
| Qwen3-0.6B | Demonstrated browser artifact with token match and numeric drift |

The registered operator surface is broader than this table, but registry
coverage alone does not guarantee that an arbitrary model will run end to end.

## Operator Support

The backend registers more than 100 operator symbols. Representative groups
are listed below; this is not an exhaustive registry listing.

| Category | Representative operators |
|---|---|
| Quantized linear and embedding | `linear_q4gsw`, `embedding_q4gsw`, `linear_qcs4w`, `linear_dq8ca_q4gsw`, `choose_qparams_affine` |
| Int8 quantized (q8ta) | `q8ta_linear`, `q8ta_conv2d`, `q8ta_add`, `q8ta_relu`, `q8ta_pixel_shuffle`, per-tensor quantize/dequantize |
| Training primitives | `adamw_step`, `fused_ce`, `linear_q4gsw_backward`, `linear_dW`, `q4gsw_requant` |
| Attention and position | `sdpa_with_kv_cache`, `update_cache`, `apply_rotary_emb` (default, interleaved, and Hugging Face layouts) |
| Elementwise | `add`, `mul`, `sub`, `div`, `pow`, `minimum`, `sigmoid`, `relu`, `gelu`, `tanh`, `abs`, `neg`, `exp`, `sqrt`, `rsqrt`, `sin`, `cos`, `clamp` |
| Comparison and boolean | `eq`, `ne`, `le`, `ge`, `lt`, `gt`, `where`, logical and bitwise operators |
| Shape and memory | `view_copy`, `slice_copy`, `select_copy`, `cat`, `permute_copy`, `squeeze_copy`, `unsqueeze_copy`, `clone`, `expand_copy`, `fill`, `repeat`, `flip`, `pixel_shuffle` |
| Reduction and normalization | `rms_norm`, `layer_norm`, `batch_norm`, `group_norm`, `softmax`, `log_softmax`, `sum`, `mean`, `argmax`, `argmin`, `amax`, `amin`, pooling |
| Convolution and spatial | `conv1d`, `conv_with_clamp`, `grid_sampler_2d`, `grid_priors`, `upsample_bilinear2d` |
| Matrix multiplication | `mm`, `bmm`, `linear` |
| Indexing | `index.Tensor`, `gather`, `embedding`, `index_select` |
| Infrastructure | `prepack`, `select_as_symint`, `_to_copy` |

## Architecture

```text
PyTorch model
    │  torch.export
    ▼
Exported Program
    │  VulkanPartitioner (tags supported ops)
    ▼
Edge Dialect IR
    │  VulkanBackend.preprocess (builds a Vulkan FlatBuffer)
    ▼
.pte file (with VH00/VK00 delegate blob)
    │
    ├── Native runtime (Dawn/Tint → Metal or Vulkan)
    │   └── WebGPUGraph::build   → creates GPU buffers, pipelines, bind groups
    │       WebGPUGraph::execute → encodes and submits compute passes
    │
    └── Browser runtime (Emscripten + emdawnwebgpu)
        └── Same graph execution path, compiled to WebAssembly
```

Key design choices:

- **Reuses Vulkan serialization.** The delegate blob is a Vulkan FlatBuffer
  (`VK00`) with a `VH00` header. The WebGPU runtime ignores Vulkan texture
  storage annotations and allocates tensors as buffers.
- **Built-in WGSL shaders.** Shader source is embedded as C++ string constants,
  with build-time code generation and drift validation.
- **No Python AOT layer.** The backend directly consumes `.pte` files exported
  with `VulkanPartitioner`.
- **Dynamic shapes.** Tensors allocate at their maximum shape, with SymInt
  arithmetic and per-operator resize hooks for runtime dimensions.
- **Shape-routed dispatch.** Runtime tensor dimensions select specialized
  kernels, such as tiled GEMM for prefill and cooperative GEMV for decode.

## Infrastructure and Testing

- **CI:** Dawn/Tint and SwiftShader provide headless GPU execution on Linux/x86.
- **Operator tests:** Python operator-test modules exercise export and
  delegation. The code-generated op-test catalog and native runners compare
  GPU output with eager-generated goldens.
- **GPU profiling:** WebGPU timestamp queries provide per-kernel GPU timings on
  devices that expose the feature.
- **Shader code generation:** `gen_wgsl_headers.py` generates embedded
  `*_wgsl.h` headers; source/header drift fails validation.

## Linux Native Quick Start

The `test_build_webgpu.sh` flow sources
`.ci/scripts/setup-webgpu-linux-deps.sh` to install Dawn and SwiftShader
prebuilts on Linux. On macOS, provide a configured Dawn installation instead
of using this script.

### 1. Export an illustrative model

```python
import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
    VulkanPartitioner,
)
from executorch.exir import to_edge_transform_and_lower


class AddOne(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1.0


ep = torch.export.export(AddOne(), (torch.randn(4, 4),))
et_program = to_edge_transform_and_lower(
    ep, partitioner=[VulkanPartitioner()]
).to_executorch()

with open("model.pte", "wb") as file:
    file.write(et_program.buffer)
```

This snippet demonstrates the export path. The validation script below exports
and runs its own native reference models; it does not consume `model.pte`.

### 2. Build and run native validation

```bash
bash backends/webgpu/test/test_build_webgpu.sh
```

The script exports a `.pte`, builds the native runtime, and validates GPU
output.

## Directory Structure

```text
backends/webgpu/
├── CMakeLists.txt
├── README.md
├── runtime/
│   ├── WebGPUBackend.h/cpp          # BackendInterface (init/execute)
│   ├── WebGPUGraph.h/cpp            # GPU graph: buffers, pipelines, dispatch
│   ├── WebGPUDelegateHeader.h/cpp   # VH00 header parser
│   ├── WebGPUDevice.h/cpp           # Dawn device abstraction
│   ├── WebGPUUtils.h                # Workgroup-size helpers
│   └── ops/                         # Operator implementations
│       ├── OperatorRegistry.h/cpp   # Operator dispatch table
│       ├── add/
│       │   ├── BinaryOp.cpp
│       │   ├── binary_add.wgsl
│       │   └── binary_add_wgsl.h
│       └── ...
├── scripts/
│   ├── gen_wgsl_headers.py          # Generate embedded WGSL headers
│   └── test_webgpu_native_ci.sh      # CI entry point for native tests
└── test/
    ├── conftest.py
    ├── tester.py                    # Partitioner and supported-op list
    ├── test_build_webgpu.sh         # End-to-end build and test
    ├── test_webgpu_native.cpp       # Native C++ test runner
    ├── test_wgsl_codegen.py         # Shader drift check
    ├── native/                      # Native C++ operator tests
    ├── op_tests/                    # Codegen case catalog and generator
    └── ops/                         # Python operator-test modules
```

## Requirements

- **macOS:** Metal-capable GPU
- **Linux:** Vulkan-capable GPU and drivers
- **Browser:** A WebGPU-enabled browser; the benchmark harness uses Chrome
  Canary
- **Build:** CMake 3.19+ and a Python environment with ExecuTorch installed
