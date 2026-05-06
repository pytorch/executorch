# macOS Desktop Deployment

ExecuTorch provides robust support for macOS deployment, offering hardware-accelerated execution across both Apple Silicon and Intel-based Macs. The runtime is optimized to take advantage of Apple's Core ML framework, Metal Performance Shaders (MPS), and the CPU-optimized XNNPACK backend.

This guide covers the platform-specific requirements, available backends, and steps to build and run ExecuTorch natively on macOS.

## Prerequisites

To build and run ExecuTorch on macOS, ensure your system meets the following minimum requirements:

- **Operating System**: macOS Big Sur (11.0) or higher. For Core ML and MPS support, macOS 13.0+ and 12.4+ are required, respectively [1].
- **Development Tools**: Xcode 14.1 or higher. The Xcode Command Line Tools must be installed (`xcode-select --install`) [2].
- **Python Environment**: Python 3.10–3.13, preferably managed via Conda or `venv` [3].

### Intel Mac Considerations

For Intel-based macOS systems, PyTorch does not provide pre-built binaries. When installing ExecuTorch Python dependencies, you must build PyTorch from source by passing specific flags to the installation script:

```bash
./install_executorch.sh --use-pt-pinned-commit --minimal
```

## Available Backends

ExecuTorch supports three primary backends for macOS, allowing you to target the CPU, GPU, or Apple Neural Engine (ANE) depending on your hardware and model requirements.

| Backend | Hardware Target | Minimum macOS | Key Features |
|---|---|---|---|
| **Core ML** | CPU, GPU, ANE | 13.0 | Dynamic dispatch across all Apple hardware; supports fp32 and fp16; recommended for Apple Silicon [2]. |
| **MPS** | Apple Silicon GPU | 12.4 | Direct execution on Metal Performance Shaders; supports fp32 and fp16 [4]. |
| **XNNPACK** | CPU (ARM64 & x86_64) | 11.0 | Highly optimized CPU execution; supports 8-bit quantization; works on both Apple Silicon and Intel Macs [5]. |

## Building for macOS

The ExecuTorch CMake build system includes a dedicated `macos` preset that configures the runtime with the features and backends common for Mac targets [3].

### 1. Enable Required Backends

By default, the ExecuTorch installation script builds the XNNPACK and Core ML backends. If you intend to use the MPS backend, you must enable it during the initial setup:

```bash
CMAKE_ARGS="-DEXECUTORCH_BUILD_MPS=ON" ./install_executorch.sh
```

### 2. Compile the Runtime

Once the Python environment is configured, use the `macos` CMake preset to build the C++ runtime:

```bash
mkdir cmake-out
cmake -B cmake-out --preset macos
cmake --build cmake-out -j10
```

This will compile the core ExecuTorch libraries and the registered backends (e.g., `libxnnpack_backend.a`, `libcoremldelegate.a`).

## Runtime Integration

To integrate ExecuTorch into your macOS C++ application, link against the compiled runtime and backend libraries. 

When linking the Core ML or XNNPACK backends, the use of static initializers requires linking with the whole-archive flag to ensure the backend registration code is not stripped by the linker [2] [5].

```cmake
# CMakeLists.txt
add_subdirectory("executorch")

target_link_libraries(
    my_macos_app
    PRIVATE 
    executorch
    extension_module_static
    extension_tensor
    optimized_native_cpu_ops_lib
    $<LINK_LIBRARY:WHOLE_ARCHIVE,coremldelegate>
    $<LINK_LIBRARY:WHOLE_ARCHIVE,xnnpack_backend>
)
```

No additional code is required to initialize the backends; any `.pte` file exported for Core ML, MPS, or XNNPACK will automatically execute on the appropriate hardware when loaded by the `Module` API.

## Next Steps

- **{doc}`backends/coreml/coreml-overview`** — Deep dive into Core ML export and execution.
- **{doc}`backends/mps/mps-overview`** — Deep dive into MPS export and execution.
- **{doc}`using-executorch-cpp`** — Learn how to use the C++ `Module` API to load and run models.

---

## References

[1] ExecuTorch Documentation: [Building from Source](using-executorch-building-from-source.md)  
[2] ExecuTorch Documentation: [Core ML Backend](backends/coreml/coreml-overview.md)  
[3] ExecuTorch Documentation: [Building the C++ Runtime](using-executorch-building-from-source.md#building-the-c-runtime)  
[4] ExecuTorch Documentation: [MPS Backend](backends/mps/mps-overview.md)  
[5] ExecuTorch Documentation: [XNNPACK Backend](backends/xnnpack/xnnpack-overview.md)  
