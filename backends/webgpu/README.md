# WebGPU Backend

Run ExecuTorch models on the GPU via [WebGPU](https://www.w3.org/TR/webgpu/). The backend compiles delegated subgraphs into WGSL compute shaders executed natively through [Dawn](https://dawn.googlesource.com/dawn), whose Tint compiler is the reference WGSL implementation (Metal on macOS, Vulkan on Linux/Windows).

> **Status: Prototype, under active development.** The backend runs the core of transformer inference today — `add`, `rms_norm`, fused scaled-dot-product attention with KV cache, and 4-bit weight-only quantized linear — plus quantized embedding, rotary embedding, and constant prepacking. See [Progress](#progress) for shipped milestones.

## Progress

Milestones landed on `main`:

| Date | Milestone | Pull Request |
|---|---|---|
| 2026-04 | Made it possible to run ExecuTorch models on the GPU through WebGPU — built the backend from the ground up, including the runtime delegate that builds the GPU graph (buffers, pipelines, bind groups) and runs the model on Metal and Vulkan | [#18808](https://github.com/pytorch/executorch/pull/18808) |
| 2026-06 | Grew model support beyond element-wise operators — added the root-mean-square normalization operator (`rms_norm`) and named-data weight loading | [#19963](https://github.com/pytorch/executorch/pull/19963) |
| 2026-06 | Made sure every change is automatically tested — added WebGPU to ExecuTorch's standard backend test suite, running on Linux/x86 in CI | [#19964](https://github.com/pytorch/executorch/pull/19964) |
| 2026-06 | Removed a class of bugs and manual upkeep — the WGSL shaders are now generated automatically, with a build-time check that fails the build on shader/source drift | [#19981](https://github.com/pytorch/executorch/pull/19981) |
| 2026-06 | Got the test suite to actually run work on the GPU — added operator-allowlist delegation (unsupported operations fall back to the CPU) and a process-wide GPU device context, so models execute on the GPU during testing | [#20036](https://github.com/pytorch/executorch/pull/20036) |
| 2026-06 | Made testing match the WebGPU standard exactly — switched the native runtime and tests to Google's Dawn shader compiler (Tint, the source-of-truth WGSL implementation) running on SwiftShader for headless GPU execution | [#20079](https://github.com/pytorch/executorch/pull/20079) |
| 2026-06 | Strengthened correctness for models that run in several GPU passes — added dispatch-ordering and scratch-buffer (temporary GPU memory) support and tests | [#20080](https://github.com/pytorch/executorch/pull/20080) |
| 2026-06 | Added the attention core of transformer inference — fused scaled-dot-product attention (`sdpa_with_kv_cache`) with an `update_cache` operator for autoregressive decode | [#20086](https://github.com/pytorch/executorch/pull/20086), [#20087](https://github.com/pytorch/executorch/pull/20087) |
| 2026-06 | Added on-GPU kernel timing via WebGPU timestamp queries, for true GPU-side profiling | [#20201](https://github.com/pytorch/executorch/pull/20201) |
| 2026-06 | Added the dominant compute in quantized LLMs — 4-bit weight-only quantized linear (`linear_q4gsw`), a dequantize-and-matmul kernel | [#20226](https://github.com/pytorch/executorch/pull/20226), [#20227](https://github.com/pytorch/executorch/pull/20227) |

In review:

| Milestone | Pull Request |
|---|---|
| Adds 4-bit quantized embedding (`embedding_q4gsw`) | [#20263](https://github.com/pytorch/executorch/pull/20263) |
| Adds rotary position embedding / RoPE (`apply_rotary_emb`) | [#20264](https://github.com/pytorch/executorch/pull/20264) |
| Adds constant prepacking (`prepack`) for end-to-end model weight handling | [#20265](https://github.com/pytorch/executorch/pull/20265) |

## Architecture

```
PyTorch model
    │  torch.export
    ▼
Exported Program
    │  VulkanPartitioner (tags supported fp32 ops)
    ▼
Edge Dialect IR
    │  VulkanBackend.preprocess (builds Vulkan FlatBuffer, buffer-only storage)
    ▼
.pte file (with VH00/VK00 delegate blob)
    │
    ▼
Native runtime (Dawn/Tint → Metal / Vulkan)
    │  WebGPUGraph::build  → creates GPU buffers, pipelines, bind groups
    │  WebGPUGraph::execute → encodes + submits compute passes
    ▼
GPU output (mapped back to CPU via wgpuDevicePoll)
```

Key design choices:
- **Reuses Vulkan serialization** — the delegate blob is a Vulkan FlatBuffer (`VK00`) with a `VH00` header. All tensor storage is forced to `BUFFER` (WebGPU has no 3D storage textures).
- **Built-in WGSL shaders** — shader source is compiled as C++ string constants. Future work will embed fused shaders in the FlatBuffer for compile-time mega-kernel fusion.
- **No Python AOT code** — directly consumes .pte files exported via `VulkanPartitioner`.

## Operator Support

| Operator | WGSL Shader | Notes |
|---|---|---|
| `aten.add.Tensor` | `binary_add.wgsl` | Element-wise with alpha: `out = in1 + alpha * in2` |
| `et_vk.rms_norm.default` | `rms_norm.wgsl` | Root-mean-square normalization |
| `sdpa_with_kv_cache.default` | `sdpa_compute_attn_weights.wgsl`, `sdpa_softmax.wgsl`, `sdpa_compute_out.wgsl` | Fused scaled-dot-product attention (QK / softmax / AV) with KV cache |
| `llama.update_cache.default` | `update_cache.wgsl` | In-place KV cache update for autoregressive decode |
| `et_vk.linear_q4gsw.default` | `q4gsw_linear.wgsl` | 4-bit weight-only quantized linear (dequantize + matmul) |

**In review:** quantized embedding (`embedding_q4gsw`), rotary embedding (`apply_rotary_emb`), and constant prepacking (`prepack`).

**Planned:** `mul`, `sigmoid`, shape ops (`view`, `permute`, `slice`, `select`, `cat`, `squeeze`/`unsqueeze`), and `index` — the remaining ops needed for end-to-end Llama 3.2 1B.

## Quick Start

### 1. Setup

```bash
bash backends/webgpu/scripts/setup-wgpu-native.sh
```

This downloads prebuilt wgpu-native binaries for your platform.

### 2. Export a model

```python
import torch
from executorch.backends.vulkan import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower

class AddModule(torch.nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b

ep = torch.export.export(AddModule(), (torch.randn(4, 4), torch.randn(4, 4)))
et_program = to_edge_transform_and_lower(
    ep, partitioner=[VulkanPartitioner()]
).to_executorch()

with open("add.pte", "wb") as f:
    f.write(et_program.buffer)
```

### 3. Build and run

```bash
bash backends/webgpu/test/test_build_webgpu.sh
```

This runs Python export tests, exports a .pte, builds the native runtime, and validates GPU output.

## Directory Structure

```
backends/webgpu/
├── CMakeLists.txt
├── README.md
├── runtime/
│   ├── WebGPUBackend.h/cpp        # BackendInterface (init/execute)
│   ├── WebGPUGraph.h/cpp          # GPU graph: buffers, pipelines, dispatch
│   ├── WebGPUDelegateHeader.h/cpp # VH00 header parser
│   ├── WebGPUDevice.h/cpp         # Dawn device abstraction
│   ├── WebGPUUtils.h              # Workgroup-size helpers
│   └── ops/
│       ├── OperatorRegistry.h/cpp # Op dispatch table
│       ├── add/
│       │   ├── BinaryOp.cpp       # aten.add.Tensor implementation
│       │   ├── binary_add.wgsl    # WGSL shader source
│       │   └── binary_add_wgsl.h  # Shader as C++ string constant
│       └── rms_norm/
│           ├── RmsNorm.cpp        # et_vk.rms_norm implementation
│           ├── rms_norm.wgsl      # WGSL shader source
│           └── rms_norm_wgsl.h    # Shader as C++ string constant
├── scripts/
│   ├── setup-wgpu-native.sh      # Download wgpu-native binaries
│   └── gen_wgsl_headers.py       # Generate the embedded *_wgsl.h shader headers
└── test/
    ├── conftest.py
    ├── tester.py                  # Partitioner stages + supported-op list
    ├── test_build_webgpu.sh       # End-to-end build + test
    ├── test_webgpu_native.cpp     # C++ native test runner
    ├── test_wgsl_codegen.py       # Shader codegen check
    ├── native/                    # C++ operator tests
    └── ops/                       # Python export tests
        ├── add/
        │   └── test_add.py        # add export tests
        └── rms_norm/
            └── test_rms_norm.py   # rms_norm export tests
```

## Requirements

- **macOS**: Metal-capable GPU
- **Linux**: Vulkan-capable GPU + drivers
- **Build**: CMake 3.19+, conda environment with ExecuTorch installed
