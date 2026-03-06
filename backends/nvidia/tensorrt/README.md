# ExecuTorch TensorRT Delegate

This subtree contains the TensorRT Delegate implementation for ExecuTorch.
TensorRT is NVIDIA's high-performance deep learning inference optimizer and
runtime library. The delegate leverages TensorRT to accelerate model execution
on NVIDIA GPUs.

## Getting Started

### Clone and Install

```bash
git clone --recurse-submodules https://github.com/pytorch/executorch.git
cd executorch

python3 -m venv .venv && source .venv/bin/activate && pip install --upgrade pip

# Install TensorRT (x86_64 only — Jetson has it pre-installed via JetPack)
pip install tensorrt>=10.3

# Install ExecuTorch (auto-detects TensorRT and links it into pybindings)
./install_executorch.sh --editable
```

> **Note for Jetson users:** TensorRT is pre-installed via JetPack SDK. The
> backend automatically detects Jetson hardware and uses the system TensorRT
> and other system/user-installed packages (e.g. `onnx`) — no additional
> `pip install` needed.

### Build C++ Components

```bash
cmake -B cmake-out \
  -DEXECUTORCH_BUILD_TENSORRT=ON \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
  -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build cmake-out --target tensorrt_executor_runner benchmark -j$(nproc)
```

### Export and Run Models

```bash
# Export a single model
python -m executorch.examples.nvidia.tensorrt.export -m add

# Export all supported models
python -m executorch.examples.nvidia.tensorrt.export

# Export with ONNX baseline (for benchmarking)
python -m executorch.examples.nvidia.tensorrt.export --onnx

# Run inference with the C++ runner
./cmake-out/backends/nvidia/tensorrt/tensorrt_executor_runner --model_path=add_tensorrt.pte
```

> **Jetson memory tip:** On memory-constrained devices (e.g., Orin Nano 8GB),
> export models one at a time to avoid OOM during TRT engine building:
> ```bash
> for m in add mv3 resnet18; do
>     python -m executorch.examples.nvidia.tensorrt.export -m "$m"
> done
> ```

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

### Additional Requirements

- **CUDA Toolkit**: TensorRT requires a compatible CUDA installation
- **cuDNN**: Required for certain layer optimizations
- **NVIDIA GPU**: Compute capability 7.0 or higher recommended

## Supported Platforms

| Platform | Architecture | TensorRT Source |
|----------|-------------|------------------|
| Linux | x86_64 | pip or NVIDIA installer |
| Linux (Jetson) | aarch64 | Pre-installed via JetPack |

## Configuration Options

`TensorRTCompileSpec` supports the following options:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `workspace_size` | int | 1GB | TensorRT builder workspace size |
| `precision` | TensorRTPrecision | FP32 | Inference precision (FP32, FP16, BF16, INT8) |
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

## Python API

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

## Supported Operations

| Category | Operations |
|----------|------------|
| Elementwise | add, sub, mul, div, floor_divide, rsub, pow, abs, ceil, sqrt |
| Unary Math | cos, sin, exp, erf, log |
| Matrix | mm, addmm, bmm, linear |
| Convolution | conv2d |
| Normalization | batch_norm, layer_norm |
| Pooling | avg_pool2d, adaptive_avg_pool2d |
| Activations | relu, sigmoid, tanh, gelu, silu, hardswish, hardsigmoid, softmax, log_softmax, clamp |
| Reshape | view, reshape, squeeze, unsqueeze, permute, transpose, flatten, unflatten, contiguous, clone |
| Reduction | mean, any |
| Concat/Split | cat, split, chunk, stack |
| Comparison | eq, ne, lt, le, gt, ge, where, logical_not |
| Slicing | slice, select, index, arange |
| Padding | constant_pad_nd |
| Other | embedding, expand, repeat, upsample, pixel_shuffle, scaled_dot_product_attention, full, dropout |

## Jetson Deployment

### Performance Tuning

```bash
sudo nvpmodel -m 0    # Max performance mode
sudo jetson_clocks     # Lock clocks for consistent benchmarking
```

### DLA Support

For models that support it, you can use NVIDIA's Deep Learning Accelerator:
```python
compile_spec = TensorRTCompileSpec(
    dla_core=0,
    allow_gpu_fallback=True,
)
```

### Memory Management

On memory-constrained Jetson devices, free RAM before export:
```bash
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches && echo 1 > /proc/sys/vm/compact_memory'
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

### Correctness Tests

```bash
# Run all correctness tests
python -m pytest examples/nvidia/tensorrt/tests/test_export.py -v

# Run a single model's test
python -m pytest examples/nvidia/tensorrt/tests/test_export.py -v -k test_mv3
```

Each test exports a model with TensorRT, runs inference via ExecuTorch
pybindings, and compares outputs against eager PyTorch (atol=1e-3, rtol=1e-3)
across 3 random seeds.

### Benchmark

```bash
# Benchmark all exported models in the current directory
./cmake-out/examples/nvidia/tensorrt/benchmark

# Benchmark with options
./cmake-out/examples/nvidia/tensorrt/benchmark -d DIR -m MODEL -n 100 -w 5
```

The benchmark reports three formats per model:
- **pte** — ExecuTorch end-to-end (includes framework overhead)
- **pte-raw** — Raw TRT engine execution extracted from the .pte
- **onnx-trt** — ONNX → TRT engine (baseline, when .onnx files are present)

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `Backend TensorRTBackend is not registered` | Ensure TensorRT is installed, then re-run `./install_executorch.sh --editable` |
| `ModuleNotFoundError: tensorrt` | On x86_64: `pip install tensorrt>=10.3`. On Jetson: auto-detected |
| CMake can't find TensorRT | Set `-DTENSORRT_HOME=/path/to/tensorrt` |
| OOM during TRT engine building | Export models one at a time; free memory first |
| `extension_module not found` | Add `-DEXECUTORCH_BUILD_EXTENSION_MODULE=ON` to cmake |
