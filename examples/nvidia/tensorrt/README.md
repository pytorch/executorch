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

# Export all supported models to a directory
python -m executorch.examples.nvidia.tensorrt.export -o /tmp/trt

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
- `benchmark.cpp` - C++ benchmark runner for performance measurement
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

## Benchmarking

Export models then benchmark with the C++ runner:

```bash
# Step 1: Export models
python -m executorch.examples.nvidia.tensorrt.export -o /tmp/trt

# Step 2: Benchmark all exported models
./benchmark -d /tmp/trt

# Benchmark a specific model
./benchmark -d /tmp/trt -m mv3

# Benchmark with custom iterations
./benchmark -d /tmp/trt -n 200 -w 5
```

**Benchmark Options:**
```
-d, --model_dir DIR    Directory with .pte files (default: current dir)
-m, --model_name NAME  Run only NAME_tensorrt.pte from the directory
-n, --num_executions N Number of timed iterations (default: 100)
-w, --warmup N         Number of warmup runs (default: 3)
-v, --verbose          Enable verbose logging
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
├── benchmark.cpp                # C++ benchmark runner binary
├── tensorrt_executor_runner.cpp # C++ executor runner binary
├── tests/                      # Correctness tests
│   └── test_export.py           # Export + inference verification
├── __init__.py                  # Package exports
└── README.md                    # This file
```

The export script follows the same pattern as other backends (XNNPACK, Vulkan,
CoreML) by using the `MODEL_NAME_TO_MODEL` registry and `EagerModelFactory`
for model instantiation.
