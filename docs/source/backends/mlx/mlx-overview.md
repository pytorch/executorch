# MLX Backend

The MLX delegate is the ExecuTorch backend for Apple Silicon GPUs via the [MLX](https://github.com/ml-explore/mlx) framework. It compiles PyTorch models into a custom FlatBuffer bytecode format at export time and executes them using MLX GPU primitives at runtime.

::::{note}
The MLX delegate is experimental and under active development.
::::

## Features

- GPU acceleration on Apple Silicon (M1 and later) via MLX.
- INT2/INT4/INT8 weight quantization via [TorchAO](https://github.com/pytorch/ao).
- Dynamic shape support.
- Mutable buffers for persistent state across inference calls (e.g., KV cache).
- Zero-copy constant loading on unified memory.

## Target Requirements

- Apple Silicon Mac (M1 or later)
- [macOS](https://developer.apple.com/macos) >= 14.0

## Development Requirements

- [macOS](https://developer.apple.com/macos) on Apple Silicon (M1 or later)
- [Xcode](https://developer.apple.com/xcode/) (full installation, not just Command Line Tools — the Metal compiler is required)

Verify the Metal compiler is available:

```bash
xcrun -sdk macosx --find metal
```

If this prints a path (e.g., `/Applications/Xcode.app/.../metal`), you're set. If it errors, install Xcode from [developer.apple.com](https://developer.apple.com/xcode/), then switch the active developer directory:

```bash
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
```

----

## Using the MLX Backend

To target the MLX backend during export and lowering, pass an instance of `MLXPartitioner` to `to_edge_transform_and_lower`. The MLX backend also provides a set of graph optimization passes via `get_default_passes()` that should be passed as `transform_passes`. The example below demonstrates this process using MobileNet V2:

```python
import torch
import torchvision.models as models
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from executorch.backends.mlx import MLXPartitioner
from executorch.backends.mlx.passes import get_default_passes
from executorch.exir import to_edge_transform_and_lower

mobilenet_v2 = models.mobilenetv2.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()
sample_inputs = (torch.randn(1, 3, 224, 224), )

et_program = to_edge_transform_and_lower(
    torch.export.export(mobilenet_v2, sample_inputs),
    transform_passes=get_default_passes(),
    partitioner=[MLXPartitioner()],
).to_executorch()

with open("mv2_mlx.pte", "wb") as file:
    et_program.write_to_file(file)
```

`get_default_passes()` includes RMSNorm fusion, consecutive view/permute/dtype-cast collapsing, no-op removal, and common subexpression elimination. These are recommended for all models and required for optimal LLM performance.

::::{note}
The MLX backend is primarily designed for LLM and generative AI workloads on Apple Silicon. The MobileNet V2 example above is shown for simplicity, but in practice you would use this backend for models like Llama, Whisper, and other transformer-based architectures. See [LLM example](https://github.com/pytorch/executorch/tree/main/backends/mlx/examples/llm) for a more representative use case.
::::

See [Partitioner API](mlx-partitioner.md) for a reference on available partitioner options.

----

## Quantization

The MLX backend supports INT4, INT8, and NVFP4 weight quantization via TorchAO for both linear and embedding layers. This is particularly useful for LLM inference. See [MLX Quantization](mlx-quantization.md) for details.

----

## Runtime Integration

### Python (pybindings)

The simplest way to get started is to install ExecuTorch with Python bindings. From the repo root:

```bash
python install_executorch.py
```

On Apple Silicon, when the Metal compiler is available, the MLX backend is automatically included. You can then export models in Python using the MLX partitioner and run them via the ExecuTorch Python API.

### C++ (CMake preset)

To build the C++ runtime with the MLX delegate, use the `mlx-release` CMake workflow preset from the repo root:

```bash
cmake --workflow --preset mlx-release
```

This configures and builds a Release build of the ExecuTorch runtime with the MLX delegate and installs artifacts into `cmake-out/`. The preset enables the MLX delegate along with commonly needed extensions (module, data loader, flat tensor, LLM runner, etc.).

Downstream C++ apps can then `find_package(executorch)` and link against `mlxdelegate` and `mlx`. The `executorch_target_link_options_shared_lib` utility handles whole-archive linkage (required for static initializer registration) cross-platform, and `executorch_target_copy_mlx_metallib` copies the Metal kernel library next to the binary so MLX can find it at runtime:

```cmake
# CMakeLists.txt
find_package(executorch REQUIRED)

# Link MLX delegate (with whole-archive for static initializer registration)
target_link_libraries(my_target PRIVATE mlxdelegate mlx)
executorch_target_link_options_shared_lib(mlxdelegate)

# Copy mlx.metallib next to the binary for runtime
executorch_target_copy_mlx_metallib(my_target)
```

No additional steps are necessary to use the backend beyond linking the target. An MLX-delegated `.pte` file will automatically run on the registered backend.

There is also an `mlx-debug` preset useful during development:

```bash
cmake --workflow --preset mlx-debug
```

## Reference

**→{doc}`/backends/mlx/mlx-troubleshooting` — Debug common issues.**

**→{doc}`/backends/mlx/mlx-partitioner` — Partitioner options.**

**→{doc}`/backends/mlx/mlx-quantization` — Supported quantization schemes.**

**→{doc}`/backends/mlx/mlx-op-support` — Supported operators.**

```{toctree}
:maxdepth: 2
:hidden:
:caption: MLX Backend
mlx-troubleshooting
mlx-partitioner
mlx-quantization
mlx-op-support
```
