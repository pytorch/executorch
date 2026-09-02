# WebGPU Backend

The ExecuTorch WebGPU backend runs delegated model graphs on GPUs through the
[WebGPU API](https://www.w3.org/TR/webgpu/). It uses
[Dawn](https://dawn.googlesource.com/dawn) and Tint for native execution and
Emscripten's `emdawnwebgpu` port for browser builds. Operators are implemented
as in-tree WGSL compute shaders.

::::{note}
The WebGPU backend is experimental and under active development.
::::

## Features

- Native GPU execution on macOS and Linux.
- Browser execution through Emscripten and `emdawnwebgpu`.
- Dynamic tensor shapes backed by maximum-size GPU buffer allocations.
- FP32 execution and device-dependent FP16 paths.
- Quantized linear, embedding, and convolution kernels, including 4-bit
  weight-only and dynamic 8-bit activation paths.
- LLM-focused kernels for SDPA, KV-cache updates, rotary embeddings, and
  fused SwiGLU and QKV projections.
- GPU timestamp-query profiling on devices that expose the feature.

The backend contains more than 100 registered operators.
See [Operator Support](webgpu-op-support.md) for details.

## Architecture

WebGPU currently reuses the Vulkan backend's export and serialization path:

```text
PyTorch model
    │  torch.export
    ▼
ExportedProgram
    │  WebGPUPartitioner (thin wrapper over VulkanPartitioner)
    ▼
Vulkan FlatBuffer delegate payload (VH00/VK00) inside a .pte file
    │
    ├── Native: Dawn/Tint → Metal or Vulkan
    └── Browser: Emscripten + emdawnwebgpu → browser WebGPU implementation
```

`WebGPUPartitioner` is a thin wrapper over `VulkanPartitioner`. It forwards
partitioning unchanged and preserves Vulkan serialization and the
`VulkanBackend` delegate identifier. It does not independently validate the
narrower WebGPU runtime capability set. The runtime ignores Vulkan
texture-storage annotations and executes the graph with WebGPU buffers and
WGSL compute pipelines.

::::{important}
The WebGPU runtime currently registers with ExecuTorch as `VulkanBackend` so it
can consume Vulkan delegate payloads. Do not link the Vulkan and WebGPU runtime
backends into the same application. They provide the same backend identifier.
::::

## Target Requirements

| Target | Requirements |
|---|---|
| macOS native | A Metal-capable GPU and a Dawn installation built for macOS |
| Linux native | A Vulkan-capable GPU and driver, plus Dawn |
| Browser | A browser with WebGPU enabled and an Emscripten build using `emdawnwebgpu` |

The exact shader features available, including FP16 and timestamp queries,
depend on the selected WebGPU adapter.

## Development Requirements

- CMake 3.19 or later.
- A Python environment with ExecuTorch installed for model export.
- Dawn's CMake package for native builds.
- Emscripten for browser builds.

On Linux, the end-to-end validation script installs pinned Dawn and
SwiftShader dependencies automatically:

```bash
bash backends/webgpu/test/test_build_webgpu.sh
```

On macOS, provide a configured Dawn installation and set `Dawn_DIR` to its
CMake package directory.

## Exporting a Model

Use `WebGPUPartitioner` to produce a `.pte` file that the WebGPU runtime can
consume. This example delegates a supported `aten.add.Tensor` operation:

```python
import torch

from executorch.backends.webgpu.partitioner import WebGPUPartitioner
from executorch.exir import to_edge_transform_and_lower


class AddOne(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1.0


example_inputs = (torch.randn(4, 4),)
exported_program = torch.export.export(AddOne(), example_inputs)

et_program = to_edge_transform_and_lower(
    exported_program,
    partitioner=[WebGPUPartitioner()],
).to_executorch()

with open("add_webgpu.pte", "wb") as file:
    file.write(et_program.buffer)
```

Because `WebGPUPartitioner` delegates unchanged to `VulkanPartitioner`, it can
select operators that do not yet have a WebGPU implementation. Validate the
delegated graph against the WebGPU registry and operator tests. A
Vulkan-supported but WebGPU-unsupported operator can fail when the WebGPU
runtime builds the graph instead of falling back automatically.

## Runtime Integration

Configure a native source build with the WebGPU backend and Dawn:

```bash
# Dawn_DIR must point to Dawn's CMake package directory.
cmake -B cmake-out-webgpu \
    -DCMAKE_BUILD_TYPE=Release \
    -DEXECUTORCH_BUILD_WEBGPU=ON \
    -DDawn_DIR="${Dawn_DIR}"
cmake --build cmake-out-webgpu --target webgpu_backend
```

Link the `webgpu_backend` target into the application. The target exports the
link options needed to retain its static backend and operator registrations.
An exported Vulkan delegate payload will then be handled by WebGPU at runtime.

When building with Emscripten, the backend automatically adds
`--use-port=emdawnwebgpu` to its compile and link options instead of finding a
native Dawn package.

## Profiling

Build with timestamp-query support and enable it at runtime:

```bash
# Dawn_DIR must point to Dawn's CMake package directory.
cmake -B cmake-out-webgpu \
    -DEXECUTORCH_BUILD_WEBGPU=ON \
    -DEXECUTORCH_BUILD_WEBGPU_PROFILING=ON \
    -DDawn_DIR="${Dawn_DIR}"
cmake --build cmake-out-webgpu

export WEBGPU_TIMESTAMP_QUERY=1
```

Run the WebGPU-enabled application normally after exporting the environment
variable. Timestamp queries require adapter support and cannot be used with
chunked execution.

## Reference

**→{doc}`/backends/webgpu/webgpu-op-support` — Operator coverage and limitations.**

```{toctree}
:maxdepth: 2
:hidden:
:caption: WebGPU Backend

webgpu-op-support
```
