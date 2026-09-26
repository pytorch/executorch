---
title: "Build Failures Troubleshooting"
category: CONFIGURATION
backends: []
last_validated: 2026-04-05
source_issues: [10014, 10066, 10151, 1004, 1020, 11050, 11221, 1006, 10152, 10063, 10166, 3696, 3524, 2910]
---

# Build Failures Troubleshooting

## Common Build Errors and Fixes

### Missing Submodules (flatbuffers, etc.)

**Error:**
```
executorch/third-party/flatbuffers does not appear to contain CMakeLists.txt
```

**Fix:** Git submodules weren't initialized. Run:
```bash
git submodule sync
git submodule update --init
```

If the Arm/Ethos-U submodule fails (SSL certificate error for `git.mlplatform.org`), other submodules also fail. Remove the problematic submodule first:
```bash
git submodule deinit backends/arm/third-party/ethos-u-core-driver/
git rm backends/arm/third-party/ethos-u-core-driver/
rm -rf .git/modules/backends/arm/third-party/
git submodule update --init
```
Note: The `serialization_lib` submodule has been removed from the repo.
[Source: #1004]

### Missing zstd Module

**Error:**
```
ModuleNotFoundError: No module named 'zstd'
```
When running `./scripts/build_apple_frameworks.sh`.

**Fix:** Run `install_executorch.sh` first -- it installs pip dependencies including `zstd`. If it persists, `pip install zstd` manually. [Source: #10014]

### CMAKE_C_COMPILER Not Set

**Error:**
```
CMake Error: CMAKE_C_COMPILER not set, after EnableLanguage
CMake Error: CMAKE_CXX_COMPILER not set, after EnableLanguage
```

**Fix:** Set compiler explicitly:
```bash
export CC=gcc
export CXX=g++
```
Or ensure your Android NDK path is correct when cross-compiling. [Source: #10014]

## Platform-Specific Build Issues

### Android Cross-Compilation

**Error:**
```
Could not find toolchain file: /path/to/ndk/build/cmake/android.toolchain.cmake
```

**Fix:** Use the `$ANDROID_NDK` environment variable instead of hardcoded paths:
```bash
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a ..
```
[Source: #10014]

### iOS Build (build_apple_frameworks.sh)

1. Ensure `install_executorch.sh` has been run first
2. MPS backend requires Xcode 15+ and macOS Sonoma (Apple Silicon only) [Source: #1020]
3. MPS does NOT work on Intel Macs (`_mtl_device != nil` assertion failure) [Source: #1020]

**For Intel Mac users:** MPS was historically Apple Silicon only. Support for x86 Macs with AMD GPUs was added later (PR #1655), but Intel GPU Macs require commenting out a check in `MPSDevice.mm`. [Source: #1020]

### macOS: Dependency Version Conflicts

**Error (v0.5):**
```
The conflict is caused by:
    torchvision 0.21.0 depends on torch==2.6.0
    torchaudio 2.6.0 depends on torch==2.7.0
```

**Fix:** Upgrade to v0.6 which resolves torch dependency conflicts. For v0.6:
```bash
pip install executorch torch torchvision torchaudio
```
[Source: #10151]

### CoreML Backend on v0.5

CoreML was NOT supported out-of-the-box in v0.5. Required manual dependency installation:
```bash
./backends/apple/coreml/scripts/install_requirements.sh
```
Fixed in v0.6 where `pip install executorch` includes coremltools automatically. [Source: #10151]

### Cadence Backend: Missing lcadence_kernels

**Error:**
```
/usr/bin/ld: cannot find -lcadence_kernels: No such file or directory
```

**Fix:** Run the Cadence-specific install requirements:
```bash
cd executorch
rm -rf pip-out
git submodule sync
git submodule update --init --recursive
./install_requirements.sh
./install_executorch.sh
./backends/cadence/install_requirements.sh
```

Note: Cadence backend support is described as "brittle" by maintainers and the tutorial may not fully succeed. [Source: #11050]

### CMake 4.0 Incompatibility

**Error:**
```
Compatibility with CMake < 3.5 has been removed from CMake.
Update the VERSION argument <min> value.
```

CMake 4.0 breaks ExecuTorch builds due to third-party dependencies (gflags, googletest, cpuinfo) using `cmake_minimum_required` < 3.5. [Source: #10152, #10063]

**Fix:** Pin CMake to version 3.x:
```bash
pip install 'cmake<4.0'
```
This was fixed in PR #9732 which pins cmake < 4.0 in requirements. [Source: #10152]

### Missing abseil-cpp / re2 Submodules

**Error:**
```
The source directory .../extension/llm/tokenizers/third-party/abseil-cpp does not contain a CMakeLists.txt file.
```

**Fix:** These are nested submodules that require recursive initialization:
```bash
git submodule update --init --recursive
```
[Source: #10063]

### Low-Bit Kernels WHOLE_ARCHIVE Conflict

**Error:**
```
Impossible to link target 'llama_main' because the link item 'custom_ops',
specified without any feature, has already occurred with the feature 'WHOLE_ARCHIVE'
```

Building llama with `EXECUTORCH_BUILD_TORCHAO=ON` hits a CMake linking conflict where `custom_ops` is linked with conflicting features. Known issue with the torchao + custom_ops build configuration. [Source: #10166]

### XNNPACK + libtorch Linking Conflict

Linking both ExecuTorch's XNNPACK backend and libtorch in the same application causes XNNPACK initialization conflicts (duplicate global state). Use ExecuTorch's XNNPACK only -- do not link libtorch alongside it. [Source: #3696]

### buck2: "Error creating cell resolver"

buck2 builds fail if the working directory path contains a dot character (e.g., `/home/n.bansal1/project`). [Source: #3524]

**Workaround:** Use a directory path without dots.

### PYTHONPATH Shadows pip-installed Package

**Error:** `FileNotFoundError: .../exir/_serialize/program.fbs`

Having the ExecuTorch repo directory in `PYTHONPATH` causes the pip-installed package to look for flatbuffer schemas in the source tree. [Source: #2910]

**Fix:** `unset PYTHONPATH` when using pip-installed executorch.

## CMake Configuration Issues

### Release Build for Performance

Always use release builds for performance testing:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release
```
Debug builds are significantly slower and will give misleading benchmark results. [Source: #10297]

### Optimized Kernels

Enable optimized kernels for better CPU performance on non-delegated ops:
```cmake
option(EXECUTORCH_BUILD_KERNELS_OPTIMIZED "" ON)
```
Then link `optimized_native_cpu_ops_lib` to your target. [Source: #10297]

### iOS: Kernel Registration with Xcode

On iOS, kernel libraries use static initialization which requires `--force_load` linker flag. This is a known UX issue being addressed with manual registration APIs:

```xcconfig
// Current workaround: add force_load for each kernel library
OTHER_LDFLAGS = -force_load $(BUILT_PRODUCTS_DIR)/libkernels_optimized.a
```

A `register_<lib_name>_kernels()` API is being developed to replace this. [Source: #11221]

## Installation Flow Summary

### v0.6+ Quick Start (No Source Build)

```bash
# Create conda environment
conda create -n executorch python=3.10
conda activate executorch

# Install via pip (includes CoreML + XNNPACK export support)
pip install executorch torch torchvision torchaudio
```
[Source: #10066]

### Full Source Build

```bash
git clone https://github.com/pytorch/executorch.git
cd executorch
git submodule sync
git submodule update --init
./install_executorch.sh
```
[Source: #10014]
