# ExecuTorch Samsung Exynos Delegate

The subtree contains Exynos delegation implementation for ExecuTorch. The target of delegation
is deploying torch model run with exynos NPU/DSP.

This backend is implemented on the top of [EXYNOS_LITECORE](https://soc-developer.semiconductor.samsung.com/global/development/light-core)
Please prepare the SDK before you start, it is important to code compilation and runtime.

## Delegate Options

### Supported Chipset
- Exynos 2500 (E9955)

### Supported Inference Type
- Quantized (i8/u8/i16/u16)
- FP16

## Directory Structure

```
backends/samsung
├── aot        # Codes for generating binary buffer for ENN runtime.
├── builders   # Codes for lowering each operators.
├── partition  # ENN Partitioner.
├── passes     # Various passes helping lower models to ENN backend.
├── python     # Places to put pybind artifacts for accessing samsung libraries.
├── runtime    # ENN runtime for executing lowered models.
├── scripts    # Misc supporting scripts, not related to core functionality.
└── serialization # Codes for building Graph IR for Exynos and serializing.

examples
└── samsung # Examples to run ENN backends.
```

## How to build
Please download Exynos AI LiteCore, and set the root path of SDK directory to `EXYNOS_AI_LITECORE_PATH`.</br>
Please navigate to [Android NDK](https://developer.android.com/ndk) and download a version of NDK.
`ANDROID_NDK` refers the root path of NDK directory.</br>

### Set up environment variables
```bash
export LD_LIBRARY_PATH=${EXYNOS_AI_LITECORE_PATH}/lib64
```

### Build AOT Targets
Generated python artifacts allow user call `Compile` interface to lower a model to ENN backend in python script.
```bash
./backends/samsung/build.sh -b x86_64
```

### Build ENN Runtime
```bash
./backends/samsung/build.sh -b android --ndk ${ANDROID_NDK}
```
ANDROID_ABI=arm64-v8a is default, necessary runtime executable generate in `build_exynos_android` directory.

### Build Anroid Extension
This is later exposed Java app. Please turn on CMake option `EXECUTORCH_BUILD_ENN`, and ENN runtime will be added.
```bash
cmake extension/android \
  -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI="${ANDROID_ABI}" \
  -DCMAKE_INSTALL_PREFIX=cmake-android-out \
  -Bcmake-android-out/extension/android

cmake --build cmake-android-out/extension/android -j8
```

## Examples

Please see this [README.md](../../examples/samsung/README.md).
