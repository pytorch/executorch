# WebGPU Backend

Run ExecuTorch models on the GPU via [WebGPU](https://www.w3.org/TR/webgpu/). The backend compiles delegated subgraphs into WGSL compute shaders executed natively through [wgpu-native](https://github.com/gfx-rs/wgpu-native) (Metal on macOS, Vulkan on Linux/Windows).

> **Status: Prototype.** The backend supports a single operator today and is under active development. See [TODO.md](TODO.md) for the roadmap.

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

**Planned:** `sub`, `mul`, `relu`, `linear` (matmul), `softmax`, `layer_norm`

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
├── TODO.md
├── runtime/
│   ├── WebGPUBackend.h/cpp        # BackendInterface (init/execute)
│   ├── WebGPUGraph.h/cpp          # GPU graph: buffers, pipelines, dispatch
│   ├── WebGPUDelegateHeader.h/cpp # VH00 header parser
│   ├── WebGPUDevice.h/cpp         # wgpu-native device abstraction
│   └── ops/
│       ├── OperatorRegistry.h/cpp # Op dispatch table
│       └── add/
│           ├── BinaryOp.cpp       # aten.add.Tensor implementation
│           ├── binary_add.wgsl    # WGSL shader source
│           └── binary_add_wgsl.h  # Shader as C++ string constant
├── scripts/
│   └── setup-wgpu-native.sh      # Download wgpu-native binaries
└── test/
    ├── conftest.py
    ├── test_build_webgpu.sh       # End-to-end build + test
    ├── test_webgpu_native.cpp     # C++ native test runner
    └── ops/
        └── add/
            └── test_add.py        # Python export tests
```

## Requirements

- **macOS**: Metal-capable GPU
- **Linux**: Vulkan-capable GPU + drivers
- **Build**: CMake 3.19+, conda environment with ExecuTorch installed
