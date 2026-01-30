# AGENTS.md

This file provides guidance to coding agents, such as Claude Code (claude.ai/code) when working with code in this repository.

## Overview

ExecuTorch is PyTorch's unified solution for deploying AI models on-device (smartphones to microcontrollers). It uses ahead-of-time (AOT) compilation to prepare PyTorch models for edge deployment via export → compile → execute workflow, producing `.pte` (PyTorch ExecuTorch) files for on-device execution.

**Production Status**: Powers billions of users at Meta across Instagram, WhatsApp, Quest 3, Ray-Ban Meta Smart Glasses.

## Build System

### Initial Setup

```bash
# Install dependencies
./install_requirements.sh
./install_executorch.sh

# Configure CMake (one-time setup after cloning/pulling)
rm -rf cmake-out
mkdir cmake-out
cd cmake-out
cmake ..

# Build (use core count + 1 for -j)
cmake --build cmake-out -j9
```

### Common CMake Build Options

Configure with `-D<OPTION>=ON/OFF`:
- `EXECUTORCH_BUILD_XNNPACK` - XNNPACK backend (CPU optimization)
- `EXECUTORCH_BUILD_COREML` - CoreML backend (Apple Neural Engine)
- `EXECUTORCH_BUILD_MPS` - Metal Performance Shaders (Apple GPU)
- `EXECUTORCH_BUILD_QNN` - Qualcomm backend
- `EXECUTORCH_BUILD_VULKAN` - Vulkan backend (cross-platform GPU)
- `EXECUTORCH_BUILD_KERNELS_OPTIMIZED` - Optimized kernels (vs portable)
- `EXECUTORCH_BUILD_KERNELS_QUANTIZED` - Quantized kernel support
- `EXECUTORCH_BUILD_EXTENSION_MODULE` - Module wrapper (simplified C++ API)
- `EXECUTORCH_BUILD_EXTENSION_DATA_LOADER` - Data loader implementations
- `EXECUTORCH_BUILD_PYBIND` - Python bindings
- `EXECUTORCH_BUILD_TESTS` - Build tests
- `EXECUTORCH_OPTIMIZE_SIZE` - Size optimization (-Os vs -O2)

### Building Models

Use Makefile targets for common model builds:

```bash
# Build model runners with specific backends
make llama-cpu           # Llama with CPU
make voxtral-cuda        # Voxtral with CUDA backend
make whisper-metal       # Whisper with Metal backend (macOS)
make gemma3-cuda         # Gemma3 with CUDA
make help                # Show all available targets

# Clean build artifacts
make clean
```

Model binaries output to `cmake-out/examples/models/<model>/`

## Testing

### Python Tests

```bash
# Run all Python tests from repo root
pytest

# Test-specific paths are configured in pytest.ini
# Tests cover: backends, codegen, devtools, examples, exir, kernels, runtime
```

### C++ Tests

```bash
# Build and run all C++ tests
./test/run_oss_cpp_tests.sh

# Run specific test directory
./test/run_oss_cpp_tests.sh runtime/core/test/

# Build size test (runtime + portable kernels)
sh test/build_size_test.sh
```

### Single Test Execution

For focused testing during development:
```bash
# Python: Run specific test file
pytest path/to/test_file.py

# Python: Run specific test function
pytest path/to/test_file.py::test_function_name

# C++: Build and run single directory (see run_oss_cpp_tests.sh)
```

## Code Architecture

### Core Runtime (`runtime/`)

**Minimal C++ runtime** - Loads and executes `.pte` programs. No operators/backends included.

- `runtime/core/` - Core types: `Tensor`, `EValue`, `Error`, `Result`, `ArrayRef`
  - `portable_type/` - Portable C++ types compatible with bare-metal (no stdlib dynamic allocation)
  - `exec_aten/` - Dual-mode header: uses either ATen types or ExecuTorch portable types
- `runtime/executor/` - `Program` and `Method` execution interfaces
- `runtime/kernel/` - Kernel registration and management
- `runtime/platform/` - Platform Abstraction Layer (PAL) for architecture-specific code
- `runtime/backend/` - Backend delegate runtime APIs

### Export and Lowering (`exir/`)

**EXIR (EXport Intermediate Representation)** - AOT library for model capture and lowering.

- `exir/capture/` - Program capture via `torch.export`
- `exir/dialects/` - Op sets for different IR dialects (ATen, Edge, Backend)
- `exir/passes/` - Compiler passes for optimization and transformation
- `exir/backend/` - Backend delegate AOT APIs for partitioning and lowering
- `exir/emit/` - Converts ExportedProgram to ExecuTorch execution instructions
- `exir/_serialize/` - Serializes to final `.pte` artifact

**Key Concepts**:
- **ATen dialect**: Full PyTorch operator set
- **Edge dialect**: Reduced operator set (Core ATen) for edge devices
- **Backend dialect**: Device-specific lowered representation

### Kernels (`kernels/`)

Operator implementations registered with runtime:

- `kernels/portable/` - **Reference implementations** of Core ATen ops (portable C++, no SIMD)
  - `functions.yaml` - Defines portable kernel signatures and metadata
- `kernels/optimized/` - **Optimized implementations** (SIMD, architecture-specific)
- `kernels/quantized/` - Quantized kernel implementations
- `kernels/prim_ops/` - Primitive ops for control flow and symbolic operations
- `kernels/aten/` - ATen-compatible kernels (when building with PyTorch)

**Selective Build**: Use YAML files to specify only needed operators, reducing binary size.

### Backends (`backends/`)

Hardware-specific delegate implementations. Each backend has:
1. **Partitioner** - Splits graph into device-compatible subgraphs
2. **Quantizer** - Backend-specific quantization (optional)
3. **Runtime** - Executes lowered graph on target hardware

**Available Backends**:
- `backends/xnnpack/` - XNNPACK (optimized CPU kernels)
- `backends/apple/coreml/` - CoreML (Apple Neural Engine)
- `backends/apple/mps/` - MPS (Metal Performance Shaders)
- `backends/qualcomm/` - Qualcomm (Hexagon DSP/NPU)
- `backends/arm/` - ARM (Ethos-U NPU, VGF)
- `backends/vulkan/` - Vulkan (cross-platform GPU)
- `backends/mediatek/`, `backends/samsung/`, `backends/cadence/`, etc.

### Extensions (`extension/`)

Higher-level libraries built on runtime:

- `extension/module/` - **Module wrapper** - Simplified C++ API (`Module::load()`, `Module::forward()`)
- `extension/llm/` - LLM-specific runtime, quantization, and optimization
  - `runner/` - Text/multimodal runner API for LLMs
  - `tokenizers/` - Tokenizer implementations
  - `custom_ops/` - Custom ops for LLM optimization
- `extension/data_loader/` - FileDataLoader, BufferDataLoader implementations
- `extension/tensor/` - TensorPtr and tensor creation utilities
- `extension/pybindings/` - Python runtime API
- `extension/runner_util/` - Helpers for C++ execution tools

### Developer Tools (`devtools/`)

- `devtools/etdump/` - **ETDump**: Runtime profiling/debugging data format
- `devtools/etrecord/` - **ETRecord**: AOT debug artifact
- `devtools/inspector/` - Python API to inspect ETDump and ETRecord
- `devtools/bundled_program/` - Model validation tool with test I/O

### Schema (`schema/`)

FlatBuffer schema defining `.pte` file format (`program.fbs`).

### Code Generation (`codegen/`)

Tools to autogenerate kernel bindings between operators and runtime.

## Key Workflows

### Model Export to .pte

```python
import torch
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

# 1. Export PyTorch model
model = MyModel().eval()
example_inputs = (torch.randn(1, 3, 224, 224),)
exported_program = torch.export.export(model, example_inputs)

# 2. Lower to backend (switch backends by changing partitioner)
program = to_edge_transform_and_lower(
    exported_program,
    partitioner=[XnnpackPartitioner()]  # or CoreMLPartitioner(), QnnPartitioner()
).to_executorch()

# 3. Save .pte file
with open("model.pte", "wb") as f:
    f.write(program.buffer)
```

### C++ Runtime Execution

```cpp
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

// Using Module API (simplified)
Module module("model.pte");
auto input = make_tensor_ptr({1, 3, 224, 224}, input_data);
auto outputs = module.forward(input);

// Or using lower-level Program/Method API
// See runtime/executor for Program and Method classes
```

### LLM Export and Execution

```bash
# Export LLM
python -m executorch.extension.llm.export.export_llm \
  --model llama3_2 \
  --output llama.pte

# C++ execution
#include <executorch/extension/llm/runner/text_llm_runner.h>
auto runner = create_llama_runner("llama.pte", "tiktoken.bin");
runner->generate("Hello", config);
```

## Code Style

### C++ Guidelines

- **C++17 standard**
- **Function names**: `lower_snake_case()` (PyTorch convention, not Google style)
- **File names**: `lower_snake_case.cpp` (not `.cc` or `PascalCase.cpp`)
- **Headers**: Use `#pragma once` (not include guards)
- **Includes**: Use `<angle brackets>` always (not `"quotes"`)
- **Documentation**: Doxygen syntax (`/** ... */` or `///`) with `@param`, `@retval`
- **TODOs**: Reference issue numbers `TODO(#123): description` (not usernames)

### Portability Restrictions

ExecuTorch targets bare-metal systems. **Do not use**:
- **Exceptions** - Not supported on many microcontrollers
- **RTTI/dynamic_cast** - Adds binary bloat
- **Dynamic allocation** - `std::vector`, `std::string`, `new`, `malloc()`
- **Threading** - `std::thread`, `thread_local`, mutexes (except in optional libraries)
- **Iostreams** - File I/O via iostream

**Allowed stdlib**:
- Static concepts: `std::move`, `std::forward`
- Metaprogramming: `std::is_floating_point<>`, `std::enable_if<>`
- Pure functions: `<cmath>`, `<cstring>` (avoid allocation functions like `strdup()`)
- Placement `new` (with manual destruction)

### Python Style

Follow PyTorch core project conventions. Use `lintrunner`:

```bash
lintrunner init      # Setup
lintrunner           # Check
lintrunner -a        # Auto-fix
```

## Build Presets

See `CMakePresets.json` for workflow presets:
- `llm-release` - LLM build with optimized kernels
- `llm-release-cuda` - LLM with CUDA backend
- `llm-release-metal` - LLM with Metal backend

Model-specific presets in `examples/models/<model>/CMakePresets.json`.

## File Layout Constraints

**IMPORTANT**: The repo directory must be named exactly `executorch` (not `executorch-1`, `et`, etc.) due to include path requirements (`#include <executorch/...>`). See CMakeLists.txt:290-298 for enforcement.

## Backend Development

When adding a backend delegate:
1. Implement partitioner (split graph into subgraphs)
2. Implement runtime delegate (execute lowered graph)
3. Optional: Custom quantizer for backend-specific quantization
4. Add CMake integration in `backends/<name>/CMakeLists.txt`
5. Update backend list in root `CMakeLists.txt`

See `docs/source/backend-delegates-integration.md` for detailed guide.

## Memory and Binary Size

- **Selective Build**: Specify operators via YAML to reduce binary size
- **Memory Planning**: AOT memory allocation strategies
- **Size Optimization**: Use `-DEXECUTORCH_OPTIMIZE_SIZE=ON` for `-Os`
- **Strip Logging**: `-DEXECUTORCH_ENABLE_LOGGING=OFF` removes log strings
- **Disable Verification**: `-DEXECUTORCH_ENABLE_PROGRAM_VERIFICATION=OFF` saves ~20KB

## CI and Testing Requirements

- CI runs automatically on PRs: https://hud.pytorch.org/hud/pytorch/executorch/main <!-- @lint-ignore -->
- All tests must pass before merge
- Add tests for new features/bug fixes
- C++ tests use GTest framework
- Python tests use pytest

## Additional Resources

- Full documentation: https://pytorch.org/executorch
- API reference: https://pytorch.org/executorch/main/api-section.html
- Backend integration guide: docs/source/backend-delegates-integration.md
- Portable C++ programming: docs/source/portable-cpp-programming.md
- Contributing guide: CONTRIBUTING.md
