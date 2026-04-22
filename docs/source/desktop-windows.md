# Windows Desktop Deployment

ExecuTorch provides support for native Windows deployment, enabling high-performance model execution across a wide range of hardware configurations. The runtime leverages optimized backends like XNNPACK for CPU execution and OpenVINO for Intel hardware acceleration.

This guide details the system requirements, available backends, and steps to build and run ExecuTorch natively on Windows operating systems.

## Prerequisites

ExecuTorch is actively tested and supported on Windows 10 and newer. Ensure your environment meets the following minimum requirements:

- **Operating System**: Windows 10 or higher [1].
- **Compiler Toolchain**: Visual Studio 2022+ with Clang-CL support. Clang/LLVM support must be explicitly enabled in the Visual Studio installer [1].
- **Python Environment**: Python 3.10–3.13, preferably managed via Conda or `venv` [1].
- **Build Tools**: CMake [1].

### Windows Subsystem for Linux (WSL)

As an alternative to native Windows deployment, ExecuTorch fully supports Windows Subsystem for Linux (WSL) with any compatible Linux distribution (e.g., Ubuntu). If using WSL, refer to the **{doc}`desktop-linux`** guide for setup instructions [1].

### Important: Enable Symlinks

ExecuTorch requires symbolic links to be enabled to build the Python components. You must configure Git to support symlinks *before* cloning the repository. Missing symlinks will manifest as an error related to `version.py` when running `pip install .` [1].

```bash
# Run this command in an Administrator prompt before cloning
git config --system core.symlinks true
```

## Available Backends

ExecuTorch supports multiple backends for Windows, allowing you to optimize execution for specific CPU architectures or dedicated AI accelerators.

| Backend | Hardware Target | Architecture | Key Features |
|---|---|---|---|
| **XNNPACK** | CPU | x86, x86-64, ARM64 | Highly optimized CPU execution; supports fp32, fp16, and 8-bit quantization; works up to AVX512 on x86-64 [2]. |
| **OpenVINO** | Intel Hardware | x86-64 | Accelerates inference on Intel CPUs, integrated GPUs, discrete GPUs, and NPUs [3]. |

## Building for Windows

The ExecuTorch CMake build system configures the runtime with the features and backends common for Windows targets.

### 1. Environment Setup

Begin by cloning the ExecuTorch repository and configuring your Python environment. Once the environment is active, install the Python dependencies and the default backends (which includes XNNPACK).

```bash
# Clone and setup the environment (ensure symlinks are enabled first)
git clone -b viable/strict https://github.com/pytorch/executorch.git
cd executorch
conda create -yn executorch python=3.10.0
conda activate executorch

# Install Python packages and dependencies
./install_executorch.sh
```

### 2. Compile the Runtime

With the environment configured, use CMake to build the C++ runtime. This process will compile the core ExecuTorch libraries and the registered backends.

```bash
mkdir cmake-out
cmake -B cmake-out
cmake --build cmake-out -j10
```

This will generate the `executorch.lib` static library and the associated backend libraries (e.g., `xnnpack_backend.lib`).

### 3. OpenVINO Integration (Optional)

If you intend to target Intel hardware, the OpenVINO backend requires additional setup. You must install the OpenVINO toolkit and build the backend separately using the provided scripts.

```bash
# From the executorch/backends/openvino/ directory
pip install -r requirements.txt
cd scripts/
./openvino_build.sh
```

This generates the OpenVINO backend library in the `cmake-out/backends/openvino/` directory [3].

## Runtime Integration

To integrate ExecuTorch into your Windows C++ application, link against the compiled runtime and backend libraries. 

When linking the XNNPACK backend, the use of static initializers requires linking with the whole-archive flag to ensure the backend registration code is not stripped by the linker [2].

```cmake
# CMakeLists.txt
add_subdirectory("executorch")

target_link_libraries(
    my_windows_app
    PRIVATE 
    executorch
    extension_module_static
    extension_tensor
    optimized_native_cpu_ops_lib
    $<LINK_LIBRARY:WHOLE_ARCHIVE,xnnpack_backend>
)
```

No additional code is required to initialize the backends; any `.pte` file exported for XNNPACK or OpenVINO will automatically execute on the appropriate hardware when loaded by the `Module` API.

## Next Steps

- **{doc}`backends/xnnpack/xnnpack-overview`** — Deep dive into XNNPACK export and execution.
- **{doc}`build-run-openvino`** — Deep dive into OpenVINO setup and hardware acceleration.
- **{doc}`using-executorch-cpp`** — Learn how to use the C++ `Module` API to load and run models.

---

## References

[1] ExecuTorch Documentation: [System Requirements](using-executorch-building-from-source.md#system-requirements)  
[2] ExecuTorch Documentation: [XNNPACK Backend](backends/xnnpack/xnnpack-overview.md)  
[3] ExecuTorch Documentation: [OpenVINO Backend](build-run-openvino.md)  
