(desktop-section)=

# Desktop & Laptop Platforms

ExecuTorch provides robust, high-performance deployment capabilities for desktop and laptop environments across macOS, Linux, and Windows. By leveraging native hardware acceleration and a cross-platform C++ runtime, developers can execute PyTorch models efficiently on CPUs, GPUs, and dedicated AI accelerators (NPUs/ANEs).

This section provides comprehensive, platform-specific guides for setting up, building, and optimizing ExecuTorch for native desktop execution.

## Platform-Specific Guides

Select your target operating system below for detailed setup instructions, prerequisites, and backend integration steps.

::::{grid} 3
:::{grid-item-card} macOS
:class-card: card-prerequisites
**→ {doc}`desktop-macos`**

Native execution on Apple Silicon and Intel Macs using Core ML, MPS, and XNNPACK.
:::
:::{grid-item-card} Linux
:class-card: card-prerequisites
**→ {doc}`desktop-linux`**

High-performance deployment on Linux distributions using XNNPACK and OpenVINO.
:::
:::{grid-item-card} Windows
:class-card: card-prerequisites
**→ {doc}`desktop-windows`**

Native Windows and WSL support using XNNPACK and OpenVINO with Visual Studio.
:::
::::

## Backend Hardware Support

ExecuTorch relies on specialized backends to map model execution to the underlying desktop hardware. The table below summarizes the available backends and their supported platforms.

| Backend | Primary Hardware Target | macOS | Linux | Windows | Key Features |
|---|---|:---:|:---:|:---:|---|
| **[XNNPACK](backends/xnnpack/xnnpack-overview)** | CPU (ARM64, x86-64) | ✅ | ✅ | ✅ | Highly optimized CPU execution; supports fp32, fp16, and 8-bit quantization. Included by default. |
| **[Core ML](backends/coreml/coreml-overview)** | Apple CPU, GPU, ANE | ✅ | ❌ | ❌ | Dynamic dispatch across Apple hardware; recommended for Apple Silicon. |
| **[MPS](backends/mps/mps-overview)** | Apple Silicon GPU | ✅ | ❌ | ❌ | Direct execution on Metal Performance Shaders for high-throughput GPU inference. |
| **[OpenVINO](build-run-openvino)** | Intel CPU, GPU, NPU | ❌ | ✅ | ✅ | Intel-optimized execution across integrated graphics, discrete GPUs, and NPUs. |
| **[Vulkan](backends/vulkan/vulkan-overview)** | Cross-platform GPU | ❌ | ✅ | ❌ | GPU execution via GLSL compute shaders; primarily focused on Android but supports Linux. |

## Core Runtime Integration

Regardless of the target desktop platform, integrating ExecuTorch into a native application follows a consistent pattern using the C++ `Module` API.

- **{doc}`using-executorch-cpp`** — Learn how to use the C++ `Module` API to load `.pte` files, configure memory allocation, and execute inferences natively.
- **{doc}`using-executorch-building-from-source`** — Comprehensive reference for the CMake build system, configuration options, and presets used across all desktop platforms.

## Tutorials

- **{doc}`raspberry_pi_llama_tutorial`** — Cross compiling ExecuTorch for the Raspberry Pi on a Linux Host.

```{toctree}
:hidden:
:maxdepth: 2
:caption: Desktop Platforms

desktop-macos
desktop-linux
desktop-windows
using-executorch-cpp
using-executorch-building-from-source
raspberry_pi_llama_tutorial
```
