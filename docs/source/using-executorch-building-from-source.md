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
    * Windows Subsystem for Linux (WSL) with any of the Linux options
    * Windows 10+ with Visual Studio 2022+ (experimental)

### Software Requirements
* `conda` or another virtual environment manager
  - `conda` is recommended as it provides cross-language
    support and integrates smoothly with `pip` (Python's built-in package manager)
  - Otherwise, Python's built-in virtual environment manager `python venv` is a good alternative.
* `g++` version 7 or higher, `clang++` version 5 or higher, or another
  C++17-compatible toolchain.
* `python` version 3.10-3.12
* `Xcode Command Line Tools` (macOS only)
* `ccache` (optional) - A compiler cache that speeds up recompilation

Additional dependencies will be installed automatically when running the [Python installation](#building-the-python-package).
Note that the cross-compilable core runtime code supports a wider range of
toolchains, down to C++17. See the [Runtime Overview](runtime-overview.md) for
portability details.

## Environment Setup
 Clone the ExecuTorch repository from GitHub and create a conda environment as follows. Venv can be used in place on conda.
   ```bash
   git clone -b viable/strict https://github.com/pytorch/executorch.git
   cd executorch
   conda create -yn executorch python=3.10.0
   conda activate executorch
   ```

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
  * `--use-pt-pinned-commit`: Install the pinned PyTorch commit. When not specified, the latest PyTorch nightly build is installed.

  For Intel-based macOS systems, use `--use-pt-pinned-commit --minimal`. As PyTorch does not provide pre-built binaries for Intel Mac, installation requires building PyTorch from source. Instructions can be found in [PyTorch Installation](https://github.com/pytorch/pytorch#installation).

  Note that only the XNNPACK and CoreML backends are built by default. Additional backends can be enabled or disabled by setting the corresponding CMake flags:

  ```bash
  # Enable the MPS backend
  CMAKE_ARGS="-DEXECUTORCH_BUILD_MPS=ON" ./install_executorch.sh
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

The ExecuTorch C++ runtime is built using CMake. It can be compiled standalone to run examples, added as a CMake dependency, or cross-compiled for Android, iOS, or embedded platforms.

### Configuring

Configuration should be done after cloning, pulling the upstream repo, or changing build options. Once this is done, you won't need to do it again until you pull from the upstream repo or modify any CMake-related files.

```bash
# cd to the root of the executorch repo
cd executorch

# Clean and configure the CMake build system. It's good practice to do this
# whenever cloning or pulling the upstream repo.
./install_executorch.sh --clean
(mkdir cmake-out && cd cmake-out && cmake ..)
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

### Build Presets

ExecuTorch provides fine-grained control over what is built, as described in [Build Options](#build-options). These options are grouped into CMake presets to cover common scenarios, while providing the ability to override individual options. Presets can be specified when configuring CMake by specifying `--preset [name]` when configuring.

Preset values for common scenarios are listed below. Using a platform preset is recommended to avoid needing to specify many fine-grained build options.

 * `arm-baremetal` - Build for bare-metal ARM targets.
 * `ios` - Build features and backends common for iOS targets.
 * `macos` - Build features and backends common for Mac targets.
 * `linux` - Build features and backends for Linux targets.
 * `llm` - Build Large Language Model-specific features.
 * `profiling` - Build the ExecuTorch runtime with profiling enabled.
 * `zephyr` - Build for Zephyr RTOS.

```bash
# Configure the build with the ios preset.
cmake .. --preset ios
```

### CMake Targets and Libraries

To link against the ExecuTorch framework from CMake, the following top-level targets are exposed:

 * `executorch::backends`: Contains all configured backends.
 * `executorch::extensions`: Contains all configured extensions.
 * `executorch::kernels`: Contains all configured kernel libraries.

The backends, extensions, and kernels included in these targets are controlled by the various `EXECUTORCH_` CMake options specified by the build. Using these targets will automatically pull in the required dependencies to use the configured features.

### Running an Example Model

The example `executor_runner` binary can be used to run a model and sanity-check the build. Run the following commands to generate and run a simple model.
You should see the message "Model executed successfully" followed by the output values.

``` bash
python -m examples.portable.scripts.export --model_name="add"
./cmake-out/executor_runner --model_path add.pte
```

```
I 00:00:00.000526 executorch:executor_runner.cpp:82] Model file add.pte is loaded.
I 00:00:00.000595 executorch:executor_runner.cpp:91] Using method forward
I 00:00:00.000612 executorch:executor_runner.cpp:138] Setting up planned buffer 0, size 48.
I 00:00:00.000669 executorch:executor_runner.cpp:161] Method loaded.
I 00:00:00.000685 executorch:executor_runner.cpp:171] Inputs prepared.
I 00:00:00.000764 executorch:executor_runner.cpp:180] Model executed successfully.
I 00:00:00.000770 executorch:executor_runner.cpp:184] 1 outputs:
Output 0: tensor(sizes=[1], [2.])
```


### Compiler Cache (ccache)

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

<hr/>

## Build Options

CMake options can be used to for fine-grained control of build type, control which features are built, and configure functionality, such as logging. Options are typically specified during CMake configuration. Default values of each option are set by the active preset, but can be overridden by specifying the option when configuring.

Note that many build options require other options to be enabled. This may require enabling multiple options to enable a given feature. The CMake build output will provide an error message when a required option is not enabled.

#### Build Type

The CMake build is typically set to `Debug` or `Release`. For production use or profiling, release mode should be used to improve performance and reduce binary size. It disables program verification and executorch logging and adds optimizations flags. The `EXECUTORCH_OPTIMIZE_SIZE` flag can be used to further optimize for size with a small performance tradeoff.

```bash
# Specify build type during CMake configuration
cmake .. -DCMAKE_BUILD_TYPE=Release
```

#### Backends

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

#### Extensions

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

#### Logging

Logging is enabled by default in debug builds and disabled in release. When enabled, the default log level is Info. Both log enable and level can be overriden with options. See [Logging](using-executorch-runtime-integration.md#logging). Disabling logging and decreasing log verbosity will reduce binary size by stripping unused strings from the build.

* `EXECUTORCH_ENABLE_LOGGING` - Enable or disable framework log messages.
* `EXECUTORCH_LOG_LEVEL` - The minimum log level to emit. One of `debug`, `info`, `error`, or `fatal`.

 ```
# Enable logging at debug
cmake .. -DEXECUTORCH_ENABLE_LOGGING=ON -DEXECUTORCH_LOG_LEVEL=debug
 ```

#### Output Libraries

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

<hr/>

## Cross-Compiling for Android

### Pre-requisites
- Set up a Python environment and clone the ExecuTorch repository, as described in [Environment Setup](#environment-setup).
- Install the [Android SDK](https://developer.android.com/studio). Android Studio is recommended.
- Install the [Android NDK](https://developer.android.com/ndk).
  - Option 1: Install via [Android Studio](https://developer.android.com/studio/projects/install-ndk).
  - Option 2: Download from [NDK Downloads](https://developer.android.com/ndk/downloads).

### Building the AAR

With the NDK installed, the `build_android_library.sh` script will build the ExecuTorch Java AAR. This file contains the ExecuTorch Java bindings
and native code. See [Using the AAR File](using-executorch-android.md#using-aar-file) for usage.

```bash
export ANDROID_ABIS=arm64-v8a
export BUILD_AAR_DIR=aar-out
mkdir -p $BUILD_AAR_DIR
sh scripts/build_android_library.sh
```

### Building the Example Runner

The native executor runner can be cross-compiled for android and deployed via ADB. This step is intended as
an example of CMake cross compilation and is not necessary for integration into an app.

```bash
# Run the following lines from the `executorch/` folder
./install_executorch.sh --clean
mkdir cmake-android-out && cd cmake-android-out

# point -DCMAKE_TOOLCHAIN_FILE to the location where ndk is installed
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake  -DANDROID_ABI=arm64-v8a ..

cd  ..
cmake --build  cmake-android-out  -j9

adb shell mkdir -p /data/local/tmp/executorch
# push the binary to an Android device
adb push  cmake-android-out/executor_runner  /data/local/tmp/executorch
# push the model file
adb push  add.pte  /data/local/tmp/executorch

adb shell  "/data/local/tmp/executorch/executor_runner --model_path /data/local/tmp/executorch/add.pte"
```

<hr/>

## Cross-Compiling for iOS

For iOS, we'll build [frameworks](https://developer.apple.com/documentation/xcode/creating-a-multi-platform-binary-framework-bundle) instead of static libraries. The frameworks contain the compiled ExecuTorch runtime and public headers.

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

Check out the [iOS Demo App](https://github.com/meta-pytorch/executorch-examples/tree/main/mv3/apple/ExecuTorchDemo) tutorial for more info.

<hr/>

## Building on Windows

ExecuTorch provides experimental support for native Windows builds.

> **_NOTE:_**  All commands should be executed on Windows powershell in administrator mode.

### Environment Setup

#### Pre-requisites

1. Install miniconda for Windows from the [official website](https://docs.conda.io/en/latest/miniconda.html).
2. Install Git for Windows from the [official website](https://git-scm.com/download/win).
3. Install ClangCL for Windows from the [official website](https://learn.microsoft.com/en-us/cpp/build/clang-support-msbuild?view=msvc-170) or through a [Visual Studio](https://learn.microsoft.com/en-us/cpp/build/clang-support-msbuild?view=msvc-170) or [Visual Studio Code](https://code.visualstudio.com/docs/cpp/config-clang-mac) installation.

#### Clone and Configure Environment

```bash
git config --global core.symlinks true
git clone --recurse -submodules https://github.com/pytorch/executorch.git
cd executorch
conda create -yn et python=3.12
conda activate et
```

If Conda is not available, run conda-hook.ps1, where `$miniconda_dir` is the directory where miniconda is installed.
This is `“C:\Users\<username>\AppData\Local”` by default.

```bash
$miniconda_dir\\shell\\condabin\\conda-hook.ps1
```

### Build the Python Package

Run `install_executorch.bat` to build and install the ExecuTorch Python package and runtime bindings.

```bash
cd executorch
./install_executorch.bat
```

> **_NOTE_** Many components are not currently buildable on Windows. These instructions install a very minimal ExecuTorch which can be used as a sanity check.

### Build the C++ Runtime

```bash
del -Recurse -Force cmake-out; `
cmake . `
  -DCMAKE_INSTALL_PREFIX=cmake-out `
  -DPYTHON_EXECUTABLE=$miniconda_dir\\envs\\et\\python.exe `
  -DCMAKE_PREFIX_PATH=$miniconda_dir\\envs\\et\\Lib\\site-packages `
  -DCMAKE_BUILD_TYPE=Release `
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON `
  -DEXECUTORCH_BUILD_FLATC=ON `
  -DEXECUTORCH_BUILD_PYBIND=OFF `
  -DEXECUTORCH_BUILD_XNNPACK=ON `
  -DEXECUTORCH_BUILD_KERNELS_LLM=ON `
  -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON `
  -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON `
  -DEXECUTORCH_ENABLE_LOGGING=ON `
  -T ClangCL `
  -Bcmake-out; `
cmake --build cmake-out -j64 --target install --config Release
```

> **_NOTE_** `$miniconda_dir` is the directory where you installed miniconda. This is `“C:\Users\<username>\AppData\Local”` by default.

### Running an Example Model

To validate the installation by running a model, create a file named export_mv2.py. Then, run the powershell commands to export and run the model.
The expected output is a tensor of size 1x1000, containing class scores.

```py
# export_mv2.py
import torch
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from torchvision.models import mobilenet_v2
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights

mv2 = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()
example_inputs = (torch.randn((1, 3, 224, 224)),)

program = to_edge_transform_and_lower(
  torch.export.export(model, example_inputs)
).to_executorch()

with open("mv2_xnnpack.pte", "wb") as file:
    executorch_program.write_to_file(file)
```

```bash
python .\\export_mv2.py
.\\cmake-out\\backends\\xnnpack\\Release\\xnn_executor_runner.exe --model_path=.\\mv2_xnnpack.pte
```

```bash
Output 0: tensor(sizes=[1, 1000], [
  -0.50986, 0.30064, 0.0953904, 0.147726, 0.231205, 0.338555, 0.206892, -0.0575775, … ])
```

## Next Steps

* [Selective Build](kernel-library-selective-build.md) to link only kernels used by the program. This can provide significant binary size savings.
* Tutorials on building [Android](https://github.com/meta-pytorch/executorch-examples/tree/main/dl3/android/DeepLabV3Demo#executorch-android-demo-app) and [iOS](https://github.com/meta-pytorch/executorch-examples/tree/main/mv3/apple/ExecuTorchDemo) demo apps.
* Tutorials on deploying applications to embedded devices such as [ARM Cortex-M/Ethos-U](backends-arm-ethos-u.md) and [XTensa HiFi DSP](backends-cadence.md).
