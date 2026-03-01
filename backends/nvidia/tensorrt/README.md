# ExecuTorch TensorRT Delegate

This subtree contains the TensorRT Delegate implementation for ExecuTorch.
TensorRT is NVIDIA's high-performance deep learning inference optimizer and
runtime library. The delegate leverages TensorRT to accelerate model execution
on NVIDIA GPUs.

## Getting Started

### Clone and Install

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/pytorch/executorch.git
cd executorch

# Create virtual environment and install ExecuTorch
python3 -m venv .venv && source .venv/bin/activate && pip install --upgrade pip
./install_executorch.sh --editable

# Install TensorRT (x86_64 only - Jetson has it pre-installed)
pip install tensorrt>=10.3
```

> **Note for Jetson users:** TensorRT is pre-installed via JetPack SDK. The
> ExecuTorch TensorRT backend automatically detects Jetson hardware and uses
> the system TensorRT installation - no `pip install tensorrt` needed.

### Export and Run Models

```bash
# Export a single model
python -m executorch.examples.nvidia.tensorrt.export -m add

# Export all supported models (default when no -m specified)
python -m executorch.examples.nvidia.tensorrt.export

# Build C++ runner
cmake -B cmake-out -DEXECUTORCH_BUILD_TENSORRT=ON
cmake --build cmake-out --target tensorrt_executor_runner -j$(nproc)

# Run inference
./cmake-out/backends/nvidia/tensorrt/tensorrt_executor_runner --model_path=add_tensorrt.pte
```

## Prerequisites

### TensorRT Installation

TensorRT is required for both ahead-of-time (AOT) compilation and runtime
execution. The installation method depends on your platform:

#### NVIDIA Jetson (aarch64)

TensorRT is **pre-installed** via JetPack SDK. No additional installation is
required.

To verify your TensorRT installation:
```bash
dpkg -l | grep -i tensorrt
```

> **Note:** Ensure you are using JetPack 6.x or later for TensorRT 10.x support.

#### Linux x86_64

Install TensorRT via pip:
```bash
pip install tensorrt>=10.3
```

Alternatively, download and install from the
[NVIDIA TensorRT Download Page](https://developer.nvidia.com/tensorrt).

#### Windows x86_64

Download and install from the
[NVIDIA TensorRT Download Page](https://developer.nvidia.com/tensorrt).

### Additional Requirements

- **CUDA Toolkit**: TensorRT requires a compatible CUDA installation
- **cuDNN**: Required for certain layer optimizations
- **NVIDIA GPU**: Compute capability 7.0 or higher recommended

## Supported Platforms

| Platform | Architecture | TensorRT Source |
|----------|-------------|-----------------|
| Linux | x86_64 | pip or NVIDIA installer |
| Linux (Jetson) | aarch64 | Pre-installed via JetPack |
| Windows | x86_64 | NVIDIA installer |

## Configuration Options

`TensorRTCompileSpec` supports the following options:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `workspace_size` | int | 1GB | TensorRT builder workspace size |
| `precision` | TensorRTPrecision | FP32 | Inference precision (FP32, FP16, INT8) |
| `strict_type_constraints` | bool | False | Enforce strict type constraints |
| `max_batch_size` | int | 1 | Maximum batch size |
| `device_id` | int | 0 | CUDA device ID |
| `dla_core` | int | -1 | DLA core ID (-1 = disabled) |
| `allow_gpu_fallback` | bool | True | Allow GPU fallback when using DLA |

## Version Compatibility

- **Minimum TensorRT version**: 10.3 (required for serialization format
  compatibility)
- **Recommended**: TensorRT 10.6 or later for best performance and feature
  support

## Quick Start

### Export a Model

```python
import torch
from torch.export import export
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.nvidia.tensorrt import TensorRTPartitioner

# Define your model
class MyModel(torch.nn.Module):
    def forward(self, x, y):
        return x + y

model = MyModel()
example_inputs = (torch.randn(2, 3), torch.randn(2, 3))

# Export and lower to TensorRT
with torch.no_grad():
    exported = export(model, example_inputs)

edge_program = to_edge_transform_and_lower(
    exported,
    partitioner=[TensorRTPartitioner()],
)

# Save the .pte file
exec_prog = edge_program.to_executorch()
with open("model_tensorrt.pte", "wb") as f:
    exec_prog.write_to_file(f)
```

### Using the Export Script

```bash
python -m executorch.examples.nvidia.tensorrt.export --model mv3 --output mv3_tensorrt.pte
```

### Run Inference

```bash
./tensorrt_executor_runner --model_path=model_tensorrt.pte
```

## Supported Operations

| Category | Operations |
|----------|-----------|
| Elementwise | add, sub, mul, div, floor_divide, rsub |
| Matrix | mm, addmm, bmm, linear |
| Convolution | conv2d |
| Normalization | batch_norm, layer_norm |
| Pooling | avg_pool2d, adaptive_avg_pool2d |
| Activations | relu, sigmoid, tanh, gelu, silu, hardswish, hardsigmoid, softmax, log_softmax, clamp |
| Reshape | view, reshape, squeeze, unsqueeze, permute, transpose, flatten, unflatten, contiguous, clone |
| Reduction | mean, any |
| Concat/Split | cat, split, chunk, stack |
| Comparison | eq, ne, lt, le, gt, ge, where, logical_not |
| Slicing | slice, select, index |
| Other | embedding, expand, repeat, upsample, pixel_shuffle, scaled_dot_product_attention, full |

## Jetson Deployment

### Prerequisites

On your Jetson device:
- JetPack 5.x or 6.x (includes TensorRT and CUDA)
- Python 3.8+
- CMake 3.19+

### Build on Jetson

```bash
git clone --recurse-submodules https://github.com/pytorch/executorch.git
cd executorch
mkdir build && cd build

# JetPack pre-installs TensorRT and CUDA - auto-detected by CMake
cmake .. -DEXECUTORCH_BUILD_TENSORRT=ON -DCMAKE_BUILD_TYPE=Release

cmake --build . --target tensorrt_backend tensorrt_executor_runner -j$(nproc)
```

### Export and Run

```bash
# Export model on Jetson
python -m executorch.examples.nvidia.tensorrt.export --model mv3 --output mv3_tensorrt.pte

# Run inference
./tensorrt_executor_runner --model_path=mv3_tensorrt.pte --num_executions=100
```

### Jetson Notes

1. **Unified Memory**: Jetson uses unified memory (CPU and GPU share memory), so no explicit
   data transfers are needed.

2. **DLA Support**: For models that support it, you can use NVIDIA's Deep Learning Accelerator:
   ```python
   compile_spec = TensorRTCompileSpec(
       dla_core=0,
       allow_gpu_fallback=True,
   )
   ```

3. **Power Modes**: Set appropriate power mode for your use case:
   ```bash
   sudo nvpmodel -m 0  # Max performance
   sudo jetson_clocks  # Lock clocks for consistent benchmarking
   ```

## Blob Format

The TensorRT delegate uses a custom binary blob format:

```
┌─────────────────────────────────────┐
│ Header (32 bytes)                   │
│  - magic: "TR01" (4 bytes)          │
│  - metadata_offset (4 bytes)        │
│  - metadata_size (4 bytes)          │
│  - engine_offset (4 bytes)          │
│  - engine_size (8 bytes)            │
│  - reserved (8 bytes)               │
├─────────────────────────────────────┤
│ Metadata JSON (variable)            │
│  - I/O binding information          │
│  - Tensor names, dtypes, shapes     │
├─────────────────────────────────────┤
│ Padding (16-byte alignment)         │
├─────────────────────────────────────┤
│ TensorRT Engine (variable)          │
│  - Serialized TensorRT engine       │
└─────────────────────────────────────┘
```

## Requirements

- NVIDIA GPU with CUDA Compute Capability 5.0+
- **TensorRT 10.3+** (required for serialization compatibility)
- CUDA Toolkit 11.x or 12.x
- cuDNN 8.x
- PyTorch 2.x with CUDA support (for export)

## Build Instructions

```bash
cd executorch
mkdir -p cmake-out && cd cmake-out

cmake .. -DEXECUTORCH_BUILD_TENSORRT=ON

cmake --build . --target tensorrt_backend tensorrt_executor_runner
```

## Troubleshooting

### Common Issues

1. **"No CUDA devices available"**: Ensure CUDA drivers are installed and GPU is accessible.

2. **TensorRT engine build takes long time**: This is expected for complex models. The engine
   is cached in the `.pte` file, so subsequent runs are fast.

3. **Model has multiple partitions**: Some operations may not be supported. Check the
   `SUPPORTED_OPS` in `partitioner/operator_support.py`.

### Debugging

```bash
# Check TensorRT version
python -c "import tensorrt; print(tensorrt.__version__)"

# Check CUDA
nvidia-smi

# Run with verbose logging
./tensorrt_executor_runner --model_path=model.pte --verbose
```

## Dynamic Shape Support

The TRT backend has two layers of dynamic shape infrastructure:

**Python (AOT):** When the backend detects symbolic dimensions in input
tensors, it uses `-1` for those dims in the TRT network and creates an
`IOptimizationProfile` with min/opt/max bounds from `range_constraints`.
Shape-dependent ops with symbolic scalar arguments (e.g., `arange` with a
dynamic bound) are excluded from TRT delegation by the partitioner and fall
back to portable execution.

**C++ (runtime):** The executor calls `setInputShape()` before `enqueueV3()`
to set actual input dimensions, computes actual-size copies (not max-buffer
copies), and resizes output tensors based on TRT-inferred shapes.

**Current limitation:** ExecuTorch's memory planner pre-allocates fixed-size
buffers for intermediate tensors. When a method mixes TRT delegates with
portable fallback ops (e.g., encoder with delegated convolutions but portable
`arange` for masking), the fallback ops cannot resize their outputs for
variable-length inputs. Until ExecuTorch supports dynamic intermediate
buffers, models with dynamic shapes should use static export shapes for TRT
and pad inputs at runtime.
