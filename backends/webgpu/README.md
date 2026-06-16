# WebGPU Backend

Run ExecuTorch models on the GPU via [WebGPU](https://www.w3.org/TR/webgpu/). The backend compiles delegated subgraphs into WGSL compute shaders executed natively through [wgpu-native](https://github.com/gfx-rs/wgpu-native) (Metal on macOS, Vulkan on Linux/Windows).

> **Status: Prototype.** The backend supports `add` and `rms_norm` today and is under active development. See [Progress](#progress) for shipped milestones.

## Progress

Milestones landed on `main`:

| Date | Milestone | Pull Request |
|---|---|---|
| 2026-04 | Made it possible to run ExecuTorch models on the GPU through WebGPU — built the backend from the ground up, including the runtime delegate that builds the GPU graph (buffers, pipelines, bind groups) and runs the model on Metal and Vulkan | [#18808](https://github.com/pytorch/executorch/pull/18808) |
| 2026-06 | Grew model support beyond element-wise operators — added the root-mean-square normalization operator (`rms_norm`) and named-data weight loading | [#19963](https://github.com/pytorch/executorch/pull/19963) |
| 2026-06 | Made sure every change is automatically tested — added WebGPU to ExecuTorch's standard backend test suite, running on Linux/x86 in CI | [#19964](https://github.com/pytorch/executorch/pull/19964) |
| 2026-06 | Removed a class of bugs and manual upkeep — the WGSL shaders are now generated automatically, with a build-time check that fails the build on shader/source drift | [#19981](https://github.com/pytorch/executorch/pull/19981) |
| 2026-06 | Got the test suite to actually run work on the GPU — added operator-allowlist delegation (unsupported operations fall back to the CPU) and a process-wide GPU device context, so models execute on the GPU during testing | [#20036](https://github.com/pytorch/executorch/pull/20036) |

In review:

| Milestone | Pull Request |
|---|---|
| Makes testing match the WebGPU standard exactly — switches the tests to Google's Dawn shader compiler (Tint, the source-of-truth WGSL implementation) running on SwiftShader for headless GPU execution | [#20079](https://github.com/pytorch/executorch/pull/20079) |
| Strengthens correctness for models that run in several GPU passes — adds dispatch-ordering and scratch-buffer (temporary GPU memory) tests | [#20080](https://github.com/pytorch/executorch/pull/20080) |

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
Native runtime (wgpu-native → Metal / Vulkan)
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

**Planned:** scaled-dot-product attention (KV cache), quantized linear (4-bit weight-only and 8da4w post-training quantization), quantized embedding, RoPE, `mul`, `sigmoid`, and shape ops (`view`, `permute`, `slice`, `select`, `cat`, `squeeze`/`unsqueeze`).

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
│   ├── WebGPUDevice.h/cpp         # wgpu-native device abstraction
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
