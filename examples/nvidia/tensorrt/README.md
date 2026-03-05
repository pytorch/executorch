# TensorRT Delegate Examples

This directory contains examples for exporting and running models with the
TensorRT delegate on NVIDIA GPUs.

## Overview

The TensorRT delegate accelerates model inference by converting ExecuTorch
models to TensorRT engines that run optimized on NVIDIA GPUs.

## Quick Start

### Export a Model

Export a supported model to ExecuTorch format with TensorRT delegation:

```bash
# Export the add model
python -m executorch.examples.nvidia.tensorrt.export -m add

# Export with validation test
python -m executorch.examples.nvidia.tensorrt.export -m add --test

# Export to a specific directory
python -m executorch.examples.nvidia.tensorrt.export -m add -o ./output
```

### Run with C++ Runner

Run the exported model using the C++ executor runner:

```bash
# Build with CMake
cmake -DEXECUTORCH_BUILD_TENSORRT=ON ...
cmake --build . --target tensorrt_executor_runner

# Run the model
./tensorrt_executor_runner --model_path=add_tensorrt.pte
```

Note: The C++ runner requires TensorRT and CUDA which are only available
on systems with NVIDIA GPUs.

### Supported Models

Currently supported models:
- `add` - Simple element-wise addition

More models will be added as converters are implemented.

Run `--help` to see all available options:

```bash
python -m executorch.examples.nvidia.tensorrt.export --help
./tensorrt_executor_runner --help
```

## Files

- `export.py` - Main export script for converting models to TensorRT format
- `runner.py` - Python utilities for running and testing exported models
- `tensorrt_executor_runner.cpp` - C++ executor runner for TensorRT models
- `__init__.py` - Package initialization

## Usage

### Python Export

```python
from executorch.examples.nvidia.tensorrt.export import main
```

Or run directly:

```bash
python -m executorch.examples.nvidia.tensorrt.export -m add
```

### C++ Runner Options

```
--model_path=PATH    Path to .pte model file (default: model.pte)
--num_executions=N   Number of times to run inference (default: 1)
--verbose            Enable verbose output
--help               Show help message
```

### Validation Testing

The `--test` flag runs the exported model through the ExecuTorch runtime
and compares outputs against the PyTorch reference model:

```bash
python -m executorch.examples.nvidia.tensorrt.export -m add --test
```

## Adding New Models

To add support for a new model:

1. Ensure the required operator converters exist in
   `executorch/backends/nvidia/tensorrt/converters/`
2. Add the model name to `TENSORRT_SUPPORTED_MODELS` in `export.py`
3. Verify the model is registered in `executorch/examples/models/`

## Architecture

```
examples/nvidia/tensorrt/
├── export.py                    # CLI export script using MODEL_NAME_TO_MODEL registry
├── runner.py                    # Python runtime utilities for testing
├── tensorrt_executor_runner.cpp # C++ executor runner binary
├── __init__.py                  # Package exports
└── README.md                    # This file
```

The export script follows the same pattern as other backends (XNNPACK, Vulkan,
CoreML) by using the `MODEL_NAME_TO_MODEL` registry and `EagerModelFactory`
for model instantiation.
