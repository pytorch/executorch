# Building from Source

ExecuTorch uses [CMake](https://cmake.org/) as the primary build system.
Even if you don't use CMake directly, CMake can emit scripts for other format
like Make, Ninja or Xcode. For information, see [cmake-generators(7)](https://cmake.org/cmake/help/latest/manual/cmake-generators.7.html).

## System Requirements

### Operating System

ExecuTorch is tested on the following systems, although it should also work in similar environments.

 * Linux (x86_64)
    * CentOS 8+
    * Ubuntu 20.04.6 LTS+
    * RHEL 8+
 * macOS (x86_64/ARM64)
    * Big Sur (11.0)+
 * Windows (x86_64)
    * Windows 10+ with Visual Studio 2022+ and [Clang-CL](https://learn.microsoft.com/en-us/cpp/build/clang-support-msbuild?view=msvc-170)
    * Windows Subsystem for Linux (WSL) with any of the Linux options

### Software Requirements

* `conda` or another virtual environment manager
  - `conda` is recommended as it provides cross-language
    support and integrates smoothly with `pip` (Python's built-in package manager)
  - Otherwise, Python's built-in virtual environment manager `python venv` is a good alternative.
* `g++` version 7 or higher, `clang++` version 5 or higher, or another
  C++17-compatible toolchain.
* `python` version 3.10-3.12
* `ccache` (optional) - A compiler cache that speeds up recompilation
* **macOS**
  - `Xcode Command Line Tools`
* **Windows**
  - `Visual Studio Clang Tools` - See [Clang/LLVM support in Visual Studio](https://learn.microsoft.com/en-us/cpp/build/clang-support-msbuild?view=msvc-170).

Additional dependencies will be automatically installed when running the [Python installation](#building-the-python-package).
Note that the cross-compilable core runtime code supports a wider range of
toolchains, down to C++17. See [Runtime Overview](runtime-overview.md) for
portability details.

## Environment Setup
 Clone the ExecuTorch repository from GitHub and create a conda environment. Venv can be used in place of conda.
   ```bash
   git clone -b viable/strict https://github.com/pytorch/executorch.git
   cd executorch
   conda create -yn executorch python=3.10.0
   conda activate executorch
   ```

> **_NOTE:_** Addition Windows Setup
>
> ExecuTorch requires symlinks to be enabled to build the Python components. To enable symlinks, run the following command before cloning the repository. Missing symlinks will manifest as an error related to `version.py` when running `pip install .`. See [src/README.md](https://github.com/pytorch/executorch/blob/main/src/README.md) for more information.
> ```bash
>   git config --system core.symlinks true
>  ```

<hr/>

## Building the Python package
  To build and install the ExecuTorch Python components, used for PTE creation and Python runtime bindings, run the following command.
  This will install the ExecuTorch python package and its dependencies into the active Python environment.

   ```bash
   # Install ExecuTorch pip package and its dependencies.
   ./install_executorch.sh
   ```

   The `install_executorch.sh` script supports the following flags:

  * `--clean`: Removes build artifacts.
  * `--editable`: Install the ExecuTorch python package in editable mode (see [Editable Install](#editable-install)).
  * `--minimal`: Install only the minimal set of dependencies required to run ExecuTorch. Do not install dependencies for examples.
  * `--use-pt-pinned-commit`: Install the pinned PyTorch commit or release version. When not specified, the latest PyTorch nightly build is installed.

  For Intel-based macOS systems, use `--use-pt-pinned-commit --minimal`. As PyTorch does not provide pre-built binaries for Intel Mac, installation requires building PyTorch from source. Instructions can be found in [PyTorch Installation](https://github.com/pytorch/pytorch#installation).

  Note that only the XNNPACK and CoreML backends are built by default. Additional backends can be enabled or disabled by setting the corresponding CMake flags:

  ```bash
  # Enable the MPS backend
  CMAKE_ARGS="-DEXECUTORCH_BUILD_MPS=ON" ./install_executorch.sh
  ```

  ### Verify the Build

To verify that the Python components are installed correctly, run the following command. This will create a file named mv2_xnnpack_fp32.pte in the current directory for the MobileNet V2 model with the XNNPACK backend. If it completes without error, the ExecuTorch Python components are installed successfully.
```bash
python -m executorch.examples.xnnpack.aot_compiler --model_name="mv2" --delegate
```

  ### Editable Install
   For development, include the `--editable` flag, which allows for local changes to ExecuTorch Python code to be reflected without a re-install. Note that when C++ files are modified, you will need to re-run the full installation to reflect the changes.
   ```bash
   ./install_executorch.sh --editable

   # Or you can directly do the following if dependencies are already installed
   # either via a previous invocation of `./install_executorch.sh` or by explicitly installing requirements via `./install_requirements.sh` first.
   pip install -e . --no-build-isolation
   ```

> **_WARNING:_**
> Some modules can't be imported directly in editable mode. This is a known [issue](https://github.com/pytorch/executorch/issues/9558) and we are actively working on a fix for this. To work around this:
> ```bash
> # This will fail
> python -c "from executorch.exir import CaptureConfig"
> # But this will succeed
> python -c "from executorch.exir.capture import CaptureConfig"
> ```

> **_NOTE:_**  Cleaning the build system
>
> When fetching a new version of the upstream repo (via `git fetch` or `git
> pull`) it is a good idea to clean the old build artifacts. The build system
> does not currently adapt well to changes in build dependencies.
>
> You should also update and pull the submodules again, in case their versions
> have changed.
>
> ```bash
> # From the root of the executorch repo:
> ./install_executorch.sh --clean
> git submodule sync
> git submodule update --init --recursive
> ```
>
> The `--clean` command removes build artifacts, pip outputs, and also clears the ccache if it's installed, ensuring a completely fresh build environment.

<hr/>

## Building the C++ Runtime

The ExecuTorch runtime uses CMake as the build system. When using ExecuTorch from C++ user code with CMake, adding ExecuTorch as a submodule and referencing via CMake `add_subdirectory` will build the runtime as part of the user build.

When user code is not using CMake, the runtime can be built standalone and linked. The CMake options described below apply in both cases. Scripts are also provided for [Android AAR](#cross-compiling-for-android) and [iOS framework](#cross-compiling-for-ios) builds.

| Use Case                   | How to Build                                                                       |
| :------------------------- | :--------------------------------------------------------------------------------- |
| C++ with user CMake        | Use CMake `add_subdirectory`.                                                      |
| C++ without user CMake     | Bulild ExecuTorch standalone with CMake. Link libraries with user build.           |
| Android with Java/Kotlin   | Use [scripts/build_android_libraries.sh](#cross-compiling-for-android).            |
| Android with C++           | Follow C++ build steps, [cross-compile for Android](#cross-compiling-for-android). |
| iOS                        | Use [scripts/build_ios_frameworks.sh](#cross-compiling-for-ios).                   |

### Configuring

Configuration should be done after cloning, pulling the upstream repo, or changing build options. Once this is done, you won't need to do it again until you pull from the upstream repo or modify any CMake-related files.

When building as a submodule as part of a user CMake build, ExecuTorch CMake options can be specified either as part of the user CMake configuration or in user CMake code.

CMake configuration for standalone runtime build:
```bash
mkdir cmake-out
cmake -B cmake-out --preset [preset] [options]
cmake --build cmake-out -j10
```

#### Build Presets

ExecuTorch provides fine-grained control over what is built, as described in [Build Options](#build-options). These options are grouped into CMake presets to cover common scenarios while preserving the ability to override individual options. Presets can be specified when configuring CMake by specifying `--preset [name]` when configuring.

Preset values for common scenarios are listed below. Using a platform preset is recommended to avoid needing to specify many fine-grained build options.

 * `android-arm64-v8a` - Build features and backends common for arm64-v8a Android targets.
 * `android-x86_64` - Build features and backends common for x86_64 Android targets.
 * `arm-baremetal` - Build for bare-metal ARM targets.
 * `ios` - Build features and backends common for iOS targets.
 * `macos` - Build features and backends common for Mac targets.
 * `linux` - Build features and backends for Linux targets.
 * `llm` - Build Large Language Model-specific features.
 * `profiling` - Build the ExecuTorch runtime with profiling enabled.
 * `zephyr` - Build for Zephyr RTOS.

User CMake:
```cmake
set(EXECUTORCH_BUILD_PRESET_FILE ${CMAKE_SOURCE_DIR}/executorch/tools/cmake/preset/llm.cmake)
```

Standalone build:
```bash
# Configure the build with the ios preset.
cmake .. --preset ios
```

#### Build Options

CMake options can be used to for fine-grained control of build type, control which features are built, and configure functionality, such as logging. Options are typically specified during CMake configuration. Default values of each option are set by the active preset, but can be overridden by specifying the option when configuring.

Note that many build options require other options to be enabled. This may require enabling multiple options to enable a given feature. The CMake build output will provide an error message when a required option is not enabled.

User CMake:
```cmake
set(EXECUTORCH_BUILD_XNNPACK ON)
```

Standalone build:
```bash
cmake -DEXECUTORCH_BUILD_XNNPACK=ON
```

##### Build Type

The CMake build is typically set to `Debug` or `Release`. For production use or profiling, release mode should be used to improve performance and reduce binary size. It disables program verification and executorch logging and adds optimizations flags. The `EXECUTORCH_OPTIMIZE_SIZE` flag can be used to further optimize for size with a small performance tradeoff.

```bash
# Specify build type during CMake configuration
cmake .. -DCMAKE_BUILD_TYPE=Release
```

##### Backends

Typically, each hardware backend exposes a CMake option to control whether the backend is built. See backend-specific documentation for more details.

 * `EXECUTORCH_BUILD_CADENCE` - Build the Cadence DSP backend.
 * `EXECUTORCH_BUILD_COREML` - Build the Apple CoreML backend.
 * `EXECUTORCH_BUILD_CORTEX_M` - Build the ARM Cortex-M backend.
 * `EXECUTORCH_BUILD_MPS` - Build the Apple Metal Performance Shader backend.
 * `EXECUTORCH_BUILD_NEURON` - Build the MediaTek Neuron backend.
 * `EXECUTORCH_BUILD_OPENVINO` - Build the Intel OpenVINO backend.
 * `EXECUTORCH_BUILD_QNN` - Build the Qualcomm AI Engine backend.
 * `EXECUTORCH_BUILD_VGF` - Build the ARM VGF backend.
 * `EXECUTORCH_BUILD_VULKAN` - Build the Vulkan GPU backend.
 * `EXECUTORCH_BUILD_XNNPACK` - Build the XNNPACK CPU backend.

```bash
# Build the XNNPACK and Vulkan backends.
cmake .. -DEXECUTORCH_BUILD_XNNPACK=ON -DEXECUTORCH_BUILD_VULKAN=ON
```

##### Extensions

ExecuTorch extensions provide optional functionality outside of the core runtime. As the core runtime is designed to run in constrained environments, these features are typically disabled by default. Extensions include higher-level APIs (Module and Tensor), multi-threading support (Threadpool), training, and more.

 * `EXECUTORCH_BUILD_EXTENSION_APPLE` - Build the Apple extension. This provides Swift and Objective-C bindings, log routing, and platform integration with Mac and iOS. See [Using ExecuTorch on iOS](using-executorch-ios.md).
 * `EXECUTORCH_BUILD_EXTENSION_DATA_LOADER` - Build the data loader extension. Provides classes to load PTEs from files or buffers.
 * `EXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR` - Build the flat tensor extension. Provides functionality to load and save tensor data in .ptd format.
 * `EXECUTORCH_BUILD_EXTENSION_LLM` - Build the Large Language Model extension. Provides LLM-specific functionality, such as tokenizer APIs. See [Working with LLMs](llm/getting-started.md).
 * `EXECUTORCH_BUILD_EXTENSION_LLM_APPLE` - Build the Large Language Model Apple extensions.
 * `EXECUTORCH_BUILD_EXTENSION_LLM_RUNNER` - Build the Large Language Model runner extension.
 * `EXECUTORCH_BUILD_EXTENSION_MODULE` - Build the Module API extension. See [High-Level APIs](using-executorch-cpp.md#high-level-apis).
 * `EXECUTORCH_BUILD_EXTENSION_TENSOR` - Build the Tensor API extension. Provides convenience APIs for creating and managing tensors. See [High-Level APIs](using-executorch-cpp.md#high-level-apis) and [extension/tensor](https://github.com/pytorch/executorch/tree/main/extension/tensor).
 * `EXECUTORCH_BUILD_EXTENSION_TRAINING` - Build the training extension. This is experimental.
 * `EXECUTORCH_BUILD_EXTENSION_EVALUE_UTIL` - Build the EValue utility extension. Provides a method to print EValue objects. See [print_evalue.h](https://github.com/pytorch/executorch/blob/main/extension/evalue_util/print_evalue.h).
 * `EXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL` - Build the runner utility extension. Provides utility methods for running models, such as allocating input and output tensor memory and generating inputs. See [executor_runner.cpp](https://github.com/pytorch/executorch/blob/main/examples/portable/executor_runner/executor_runner.cpp) for example usage.

 ```
# Enable the data loader extension.
cmake .. -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON
 ```

##### Logging

Logging is enabled by default in debug builds and disabled in release. When enabled, the default log level is Info. Both log enable and level can be overriden with options. See [Logging](using-executorch-runtime-integration.md#logging). Disabling logging and decreasing log verbosity will reduce binary size by stripping unused strings from the build.

* `EXECUTORCH_ENABLE_LOGGING` - Enable or disable framework log messages.
* `EXECUTORCH_LOG_LEVEL` - The minimum log level to emit. One of `debug`, `info`, `error`, or `fatal`.

 ```
# Enable logging at debug
cmake .. -DEXECUTORCH_ENABLE_LOGGING=ON -DEXECUTORCH_LOG_LEVEL=debug
 ```

### Building

Build all targets with `cmake --build`.

```bash
# cd to the root of the executorch repo
cd executorch

# Build using the configuration that you previously generated under the
# `cmake-out` directory.
#
# NOTE: The `-j` argument specifies how many jobs/processes to use when
# building, and tends to speed up the build significantly. It's typical to use
# "core count + 1" as the `-j` value.
cmake --build cmake-out -j9
```

> **_TIP:_** For faster rebuilds, consider installing ccache (see [Compiler Cache section](#compiler-cache-ccache) above). On first builds, ccache populates its cache. Subsequent builds with the same compiler flags can be significantly faster.

<hr/>


## CMake Targets and Output Libraries

To link against the ExecuTorch framework from CMake, the following top-level targets are exposed:

 * `executorch::backends`: Contains all configured backends.
 * `executorch::extensions`: Contains all configured extensions.
 * `executorch::kernels`: Contains all configured kernel libraries.

The backends, extensions, and kernels included in these targets are controlled by the various `EXECUTORCH_` CMake options specified by the build. Using these targets will automatically pull in the required dependencies to use the configured features.

### Linking Without CMake

To link against the runtime from outside of the CMake ecosystem, the runtime can be first built with CMake and then linked directly. A few of the relevant top-level targets are described below. Note that this is a more involved process than using CMake and is only recommended when using CMake is not viable.

- `libexecutorch.a`: The core of the ExecuTorch runtime. Does not contain any
  operator/kernel definitions or backend definitions.
- `libportable_kernels.a`: The implementations of ATen-compatible operators,
  following the signatures in `//kernels/portable/functions.yaml`.
- `libportable_kernels_bindings.a`: Generated code that registers the contents
  of `libportable_kernels.a` with the runtime.
  - NOTE: This must be linked into your application with a flag like
    `-Wl,-force_load` or `-Wl,--whole-archive`. It contains load-time functions
    that automatically register the kernels, but linkers will often prune those
    functions by default because there are no direct calls to them.
  `libportable_kernels.a`, so the program may use any of the operators it
  implements.

Backends typically introduce additional targets. See backend-specific documentation for more details.

### Verify the Build

To verify the build, ExecuTorch optionally compiles a simple, stand-alone model runner to run PTE files with all-one input tensors. It is not enabled by default in most presets, but can be enabled by configuring with `-DEXECUTORCH_BUILD_EXECUTOR_RUNNER=ON -DEXECUTORCH_BUILD_EXTENSION_EVALUE_UTIL=ON`.

Once compiled, invoke the runner with a sample PTE (such as the one generated by [verifying the Python build](#verify-the-build)).
```bash
cmake-out/executor_runner --model_path=mv2_xnnpack_fp32.pte
```

If the runner runs successfully, you should see output similar to the following:
```
I 00:00:00.043703 executorch:executor_runner.cpp:379] Model executed successfully 1 time(s) in 15.013292 ms.
I 00:00:00.043720 executorch:executor_runner.cpp:383] 1 outputs:
OutputX 0: tensor(sizes=[1, 1000], [
  -0.509859, 0.300644, 0.0953884, 0.147724, 0.231202, 0.338554, 0.206888, -0.0575762, -0.389273, -0.0606864,
  ...,
  0.421219, 0.100447, -0.506771, -0.115824, -0.693017, -0.183262, 0.154781, -0.410684, 0.0119296, 0.449713,
])
```

<hr/>

## Cross-Compiling for Android

### Pre-requisites
- Set up a Python environment and clone the ExecuTorch repository, as described in [Environment Setup](#environment-setup).
- Install the [Android SDK](https://developer.android.com/studio). Android Studio is recommended.
- Install the [Android NDK](https://developer.android.com/ndk).
  - Option 1: Install via [Android Studio](https://developer.android.com/studio/projects/install-ndk).
  - Option 2: Download from [NDK Downloads](https://developer.android.com/ndk/downloads).

### Building the AAR

With the NDK installed, the `build_android_library.sh` script will build the ExecuTorch Java AAR, which contains ExecuTorch Java bindings. See [Using the AAR File](using-executorch-android.md#using-aar-file) for usage.

```bash
export ANDROID_ABIS=arm64-v8a
export BUILD_AAR_DIR=aar-out
mkdir -p $BUILD_AAR_DIR
sh scripts/build_android_library.sh
```

### Android Native

To use the ExecuTorch runtime from native Android C++ code, the runtime can be cross-compiled for Android. The recommended approach is to add ExecuTorch as a submodule of the user project and use [CMake](https://developer.android.com/ndk/guides/cmake) for the native build. The above steps for C++ with CMake can be followed.

For direct cross-compilation, the ExecuTorch runtime can be configured to build with the NDK toolchain:
```bash
# point -DCMAKE_TOOLCHAIN_FILE to the location where ndk is installed
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a ..
```

<hr/>

## Cross-Compiling for iOS

iOS binaries are built as [frameworks](https://developer.apple.com/documentation/xcode/creating-a-multi-platform-binary-framework-bundle) instead of static libraries. The frameworks contain the compiled ExecuTorch runtime and public headers.

### Pre-requisites

* Install Xcode from the
[Mac App Store](https://apps.apple.com/app/xcode/id497799835) and install
the Command Line Tools using the terminal.

```bash
xcode-select --install
```

### Building

1. Build the frameworks:

```bash
./scripts/build_apple_frameworks.sh
```

Run the above command with `--help` flag to learn more on how to build additional backends
(like [Core ML](backends/coreml/coreml-overview.md), [MPS](backends/mps/mps-overview.md) or XNNPACK), etc.
Note that some backends may require additional dependencies and certain versions of Xcode and iOS.
See backend-specific documentation for more details.

2. Copy over the generated `.xcframework` bundles to your Xcode project, link them against
your targets and don't forget to add an extra linker flag `-all_load`.

See the [iOS Demo App](https://github.com/meta-pytorch/executorch-examples/tree/main/mv3/apple/ExecuTorchDemo) tutorial for example usage of the ExecuTorch frameworks.

## Compiler Cache (ccache)

ExecuTorch automatically detects and enables [ccache](https://ccache.dev/) if it's installed. This significantly speeds up recompilation by caching previously compiled objects:

- If ccache is detected, you'll see: `ccache found and enabled for faster builds`
- If ccache is not installed, you'll see: `ccache not found, builds will not be cached`

To install ccache:
```bash
# Ubuntu/Debian
sudo apt install ccache

# macOS
brew install ccache

# CentOS/RHEL
sudo yum install ccache
# or
sudo dnf install ccache
```

No additional configuration is needed - the build system will automatically use ccache when available.

See [CMakeLists.txt](https://github.com/pytorch/executorch/blob/main/CMakeLists.txt)
