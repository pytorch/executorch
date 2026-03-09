---
name: building
description: Build ExecuTorch from source — Python package, C++ runtime, runners, cross-compilation, and backend-specific builds. Use when compiling anything in the ExecuTorch repo, diagnosing build failures, or setting up platform-specific builds.
---

# Building

## Prerequisites

Before building, ensure the environment is set up (see `/setup` skill):
```bash
conda activate executorch
```

Required toolchain:
- **Python** 3.10–3.13
- **CMake** >= 3.24, < 4.0
- **C++17** compiler: `g++` >= 7, `clang++` >= 5, or MSVC 2022+ with Clang-CL
- **Git submodules** must be initialized (handled by `install_executorch.sh`, or manually: `git submodule sync && git submodule update --init --recursive`)

Optional but recommended:
- **ccache** — automatically detected and used if installed (`sudo apt install ccache` / `brew install ccache`)
- **Ninja** — faster than Make (`sudo apt install ninja-build` / `brew install ninja`); use with `-G Ninja`

## 1. Building the Python Package

This installs the ExecuTorch Python package (exir, runtime bindings, etc.) into the active environment.

```bash
# First time (installs deps + builds + installs)
./install_executorch.sh

# Editable mode (Python changes reflected without rebuild)
./install_executorch.sh --editable

# Minimal (skip example dependencies)
./install_executorch.sh --minimal

# Subsequent installs (deps already present)
pip install -e . --no-build-isolation
```

**Enable additional backends** during Python install:
```bash
CMAKE_ARGS="-DEXECUTORCH_BUILD_MPS=ON" ./install_executorch.sh
CMAKE_ARGS="-DEXECUTORCH_BUILD_COREML=ON -DEXECUTORCH_BUILD_MPS=ON" ./install_executorch.sh
```

**Verify Python install:**
```bash
python -m executorch.examples.xnnpack.aot_compiler --model_name="mv2" --delegate
```

## 2. Building the C++ Runtime (Standalone)

### Using Presets (Recommended)

```bash
cmake -B cmake-out --preset <preset> -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-out -j$(nproc)
```

| Preset | Platform | What it builds |
|--------|----------|----------------|
| `linux` | Linux x86_64 | Runtime + XNNPACK + LLM + executor_runner |
| `macos` | macOS | Runtime + XNNPACK + CoreML + MPS + executor_runner |
| `windows` | Windows | Runtime + XNNPACK + executor_runner |
| `llm-release` | Host | LLM extension (CPU, Release) |
| `llm-release-cuda` | Linux/Windows | LLM extension (CUDA, Release) |
| `llm-release-metal` | macOS | LLM extension (Metal, Release) |
| `llm-debug` | Host | LLM extension (CPU, Debug) |
| `llm-debug-cuda` | Linux/Windows | LLM extension (CUDA, Debug) |
| `llm-debug-metal` | macOS | LLM extension (Metal, Debug) |
| `profiling` | Host | Runtime with profiling/event tracing |
| `android-arm64-v8a` | Android | JNI bindings + runtime for arm64 |
| `android-x86_64` | Android | JNI bindings + runtime for x86_64 |
| `ios` | iOS | Frameworks for device |
| `ios-simulator` | iOS Sim | Frameworks for simulator |
| `arm-baremetal` | Embedded | Cortex-M / Ethos-U bare-metal |
| `zephyr` | RTOS | Zephyr RTOS build |

### Using CMake Workflow Presets

Workflow presets combine configure + build + install in one command:
```bash
cmake --workflow --preset llm-release        # CPU
cmake --workflow --preset llm-release-cuda   # CUDA
cmake --workflow --preset llm-release-metal  # Metal
```

### Manual CMake (No Preset)

```bash
mkdir -p cmake-out
cmake -B cmake-out \
  -DCMAKE_BUILD_TYPE=Release \
  -DEXECUTORCH_BUILD_XNNPACK=ON \
  -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON
cmake --build cmake-out -j$(nproc)
```

### Verify C++ Build

```bash
# Enable executor_runner if not already
cmake -B cmake-out --preset linux -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=ON
cmake --build cmake-out -j$(nproc)
cmake-out/executor_runner --model_path=mv2_xnnpack_fp32.pte
```

## 3. Building Runners (Makefile)

Model-specific runners use the top-level `Makefile`:
```bash
make help              # list all targets
make llama-cpu         # Llama on CPU
make llama-cuda        # Llama on CUDA
make llama-cuda-debug  # Llama on CUDA (debug)
make llava-cpu         # Llava on CPU
make gemma3-cpu        # Gemma3 on CPU
make gemma3-cuda       # Gemma3 on CUDA
make whisper-cpu       # Whisper on CPU
make whisper-metal     # Whisper on Metal
make parakeet-cpu      # Parakeet on CPU
make parakeet-metal    # Parakeet on Metal
make clean             # remove cmake-out/
```

Output binaries: `cmake-out/examples/models/<model>/<runner>`

Each `make` target internally runs `cmake --workflow --preset` for the core libraries, then builds the runner on top.

## 4. Cross-Compilation

### Android

```bash
# AAR (Java bindings)
export ANDROID_ABIS=arm64-v8a
export BUILD_AAR_DIR=aar-out
mkdir -p $BUILD_AAR_DIR
sh scripts/build_android_library.sh

# Native C++ (direct cross-compile)
cmake -B cmake-out \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  --preset android-arm64-v8a
cmake --build cmake-out -j$(nproc)
```

### iOS / macOS Frameworks

```bash
# Build all frameworks
./scripts/build_apple_frameworks.sh

# With specific backends
./scripts/build_apple_frameworks.sh --coreml --mps --xnnpack
```

Link frameworks in Xcode with `-all_load` linker flag.

### Windows

Requires Visual Studio 2022+ with Clang-CL:
```bash
cmake -B cmake-out --preset windows -T ClangCL
cmake --build cmake-out --config Release
```

**Windows-specific notes:**
- Enable symlinks before cloning: `git config --system core.symlinks true`
- Missing symlinks cause `version.py` errors during `pip install`
- LLM custom kernels and quantized kernels do not compile with MSVC; use `-T ClangCL` or build with CUDA

## 5. Key Build Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `CMAKE_BUILD_TYPE` | STRING | Debug | `Debug` or `Release`. Release disables logging/verification, adds optimizations |
| `EXECUTORCH_BUILD_XNNPACK` | BOOL | OFF | XNNPACK CPU backend (requires CPUINFO + PTHREADPOOL) |
| `EXECUTORCH_BUILD_COREML` | BOOL | OFF | Core ML backend (macOS/iOS only) |
| `EXECUTORCH_BUILD_MPS` | BOOL | OFF | MPS GPU backend (macOS/iOS only) |
| `EXECUTORCH_BUILD_CUDA` | BOOL | OFF | CUDA GPU backend (requires EXTENSION_TENSOR) |
| `EXECUTORCH_BUILD_METAL` | BOOL | OFF | Metal backend (requires EXTENSION_TENSOR) |
| `EXECUTORCH_BUILD_VULKAN` | BOOL | OFF | Vulkan GPU backend (Android) |
| `EXECUTORCH_BUILD_QNN` | BOOL | OFF | Qualcomm QNN backend |
| `EXECUTORCH_BUILD_KERNELS_OPTIMIZED` | BOOL | OFF | Optimized kernel implementations |
| `EXECUTORCH_BUILD_KERNELS_QUANTIZED` | BOOL | OFF | Quantized kernel implementations |
| `EXECUTORCH_BUILD_KERNELS_LLM` | BOOL | OFF | LLM custom kernels (requires KERNELS_OPTIMIZED) |
| `EXECUTORCH_BUILD_EXTENSION_MODULE` | BOOL | OFF | Module extension (requires DATA_LOADER + FLAT_TENSOR + NAMED_DATA_MAP) |
| `EXECUTORCH_BUILD_EXTENSION_TENSOR` | BOOL | OFF | Tensor extension |
| `EXECUTORCH_BUILD_EXTENSION_LLM` | BOOL | OFF | LLM extension |
| `EXECUTORCH_BUILD_EXTENSION_LLM_RUNNER` | BOOL | OFF | LLM runner extension (requires EXTENSION_LLM) |
| `EXECUTORCH_BUILD_PYBIND` | BOOL | OFF | Python bindings (requires EXTENSION_MODULE) |
| `EXECUTORCH_BUILD_TESTS` | BOOL | OFF | CMake-based unit tests |
| `EXECUTORCH_BUILD_DEVTOOLS` | BOOL | OFF | Developer tools (Inspector, ETDump) |
| `EXECUTORCH_ENABLE_EVENT_TRACER` | BOOL | OFF | Event tracing (requires DEVTOOLS) |
| `EXECUTORCH_OPTIMIZE_SIZE` | BOOL | OFF | Optimize for binary size (`-Os`, no exceptions/RTTI) |
| `EXECUTORCH_ENABLE_LOGGING` | BOOL | (Debug=ON) | Runtime logging |
| `EXECUTORCH_LOG_LEVEL` | STRING | Info | Log level: Debug, Info, Error, Fatal |
| `EXECUTORCH_USE_SANITIZER` | BOOL | OFF | ASAN + UBSAN (not supported on MSVC) |
| `EXECUTORCH_PAL_DEFAULT` | STRING | posix | Platform abstraction: `posix`, `minimal`, `android` |

**Dependency chains** — enabling some options requires others:
- `XNNPACK` requires `CPUINFO` + `PTHREADPOOL`
- `KERNELS_LLM` requires `KERNELS_OPTIMIZED`
- `EXTENSION_MODULE` requires `EXTENSION_DATA_LOADER` + `EXTENSION_FLAT_TENSOR` + `EXTENSION_NAMED_DATA_MAP`
- `BUILD_PYBIND` requires `EXTENSION_MODULE`
- `EXTENSION_LLM_RUNNER` requires `EXTENSION_LLM`
- `EVENT_TRACER` requires `DEVTOOLS`
- `CUDA` and `METAL` require `EXTENSION_TENSOR`

CMake will error with a clear message if a required option is missing.

## 6. Common Build Patterns

### Build core runtime only (minimal)
```bash
cmake -B cmake-out -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-out -j$(nproc)
```

### Build with XNNPACK backend
```bash
cmake -B cmake-out -DCMAKE_BUILD_TYPE=Release \
  -DEXECUTORCH_BUILD_XNNPACK=ON
cmake --build cmake-out -j$(nproc)
```

### Build with profiling
```bash
cmake -B cmake-out --preset profiling
cmake --build cmake-out -j$(nproc)
```

### Build tests
```bash
cmake -B cmake-out -DEXECUTORCH_BUILD_TESTS=ON \
  -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON
cmake --build cmake-out -j$(nproc)
ctest --test-dir cmake-out --output-on-failure
```

### Using ExecuTorch as a CMake subdirectory
```cmake
add_subdirectory(executorch)
# Set options before add_subdirectory:
set(EXECUTORCH_BUILD_XNNPACK ON)
set(EXECUTORCH_BUILD_EXTENSION_MODULE ON)
```

## 7. Troubleshooting

### Submodule issues
**Symptom:** Build fails with missing headers or `CMakeLists.txt not found` in third-party dirs.
```bash
git submodule sync --recursive
git submodule update --init --recursive
```

### Stale build artifacts
**Symptom:** Mysterious failures after pulling new changes or switching branches.
```bash
./install_executorch.sh --clean
# Or manually:
rm -rf cmake-out/ pip-out/ buck-out/
git submodule sync && git submodule update --init --recursive
```

### CMake version conflicts
**Symptom:** `cmake` errors about policy versions or unsupported features.
- ExecuTorch requires CMake >= 3.24, < 4.0
- Check: `cmake --version`
- If conda and system cmake conflict, ensure conda env cmake is used: `which cmake` should point to conda env

### Python version mismatch
**Symptom:** `install_executorch.sh` fails early with compatibility errors.
- Supported: Python 3.10–3.13
- Check: `python --version`

### Dependency version conflicts
**Symptom:** pip fails with conflicting torch/torchvision/torchaudio versions.
- Use a fresh conda environment
- If pinning to a specific PyTorch version: `./install_executorch.sh --use-pt-pinned-commit`

### Missing `python-dev` headers
**Symptom:** Build fails looking for `Python.h`.
```bash
sudo apt install python$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')-dev
```

### Linking errors with `--whole-archive`
**Symptom:** Missing operator registrations at runtime despite building kernels.
- Kernel binding libraries (e.g., `libportable_kernels_bindings.a`) use load-time registration
- Must link with: `-Wl,--whole-archive <lib> -Wl,--no-whole-archive` (Linux) or `-Wl,-force_load,<lib>` (macOS)

### XNNPACK build fails
**Symptom:** Errors about missing `cpuinfo` or `pthreadpool`.
- `EXECUTORCH_BUILD_XNNPACK=ON` requires `EXECUTORCH_BUILD_CPUINFO=ON` and `EXECUTORCH_BUILD_PTHREADPOOL=ON` (both ON by default unless `ARM_BAREMETAL` is set)

### Windows symlink errors
**Symptom:** `version.py` not found or import errors on Windows.
```bash
git config --system core.symlinks true
# Re-clone the repo after enabling
```

### MSVC kernel compilation failures
**Symptom:** LLM/quantized kernels fail to compile on Windows with MSVC.
- Use Clang-CL: `cmake -B cmake-out -T ClangCL`
- Or build with CUDA (which uses nvcc, not MSVC for kernels)

### Intel macOS
**Symptom:** `install_executorch.sh` fails — no prebuilt PyTorch wheels for Intel Mac.
- Must build PyTorch from source, or use `--use-pt-pinned-commit --minimal`

### Build directory not at repo root
**Symptom:** Include path errors when ExecuTorch checkout is not the top-level directory.
- ExecuTorch adds `..` to include directories; the build directory must be directly under the repo root or use `add_subdirectory` correctly

### Duplicate kernel registration
**Symptom:** Abort at runtime with duplicate kernel registration.
- Only link one `gen_operators_lib` per target
- Check for multiple kernel binding libraries being linked

## 8. Build Output

| Artifact | Location | Description |
|----------|----------|-------------|
| `executor_runner` | `cmake-out/executor_runner` | Standalone model runner |
| Core runtime | `cmake-out/libexecutorch.a` | Core ExecuTorch runtime |
| Portable ops | `cmake-out/kernels/portable/libportable_ops_lib.a` | Portable operator implementations |
| XNNPACK backend | `cmake-out/backends/xnnpack/libxnnpack_backend.a` | XNNPACK delegate |
| LLM runner | `cmake-out/examples/models/<model>/<runner>` | Model-specific runners |
| Python package | site-packages | `executorch` Python module |
| iOS frameworks | `cmake-out/*.xcframework` | iOS/macOS frameworks |
| Android AAR | `aar-out/` | Android Java bindings |

## 9. Tips

- Always use `Release` for performance measurement; `Debug` is 5–10x slower and significantly larger
- Use `ccache` to speed up rebuilds — ExecuTorch auto-detects it
- Use `Ninja` generator (`-G Ninja`) for faster parallel builds
- Use `cmake --list-presets` to see all available presets
- After `git pull`, always clean and re-init submodules before rebuilding
- For LLM workflows, `make <model>-<backend>` is the simplest path
- Set `EXECUTORCH_OPTIMIZE_SIZE=ON` for size-constrained deployments
- Check `cmake-out/compile_commands.json` for IDE integration (auto-generated)
