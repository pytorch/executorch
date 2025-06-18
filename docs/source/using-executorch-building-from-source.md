# Building from Source

ExecuTorch uses [CMake](https://cmake.org/) as the primary build system.
Even if you don't use CMake directly, CMake can emit scripts for other format
like Make, Ninja or Xcode. For information, see [cmake-generators(7)](https://cmake.org/cmake/help/latest/manual/cmake-generators.7.html).

## System Requirements
### Operating System

We've tested these instructions on the following systems, although they should
also work in similar environments.


Linux (x86_64)
- CentOS 8+
- Ubuntu 20.04.6 LTS+
- RHEL 8+

macOS (x86_64/M1/M2)
- Big Sur (11.0)+

Windows (x86_64)
- Windows Subsystem for Linux (WSL) with any of the Linux options

### Software
* `conda` or another virtual environment manager
  - We recommend `conda` as it provides cross-language
    support and integrates smoothly with `pip` (Python's built-in package manager)
  - Otherwise, Python's built-in virtual environment manager `python venv` is a good alternative.
* `g++` version 7 or higher, `clang++` version 5 or higher, or another
  C++17-compatible toolchain.
* `python` version 3.10-3.12

Note that the cross-compilable core runtime code supports a wider range of
toolchains, down to C++17. See the [Runtime Overview](runtime-overview.md) for
portability details.

## Environment Setup

### Clone ExecuTorch

   ```bash
   # Clone the ExecuTorch repo from GitHub
   git clone -b viable/strict https://github.com/pytorch/executorch.git && cd executorch
   ```

### Create a Virtual Environment

Create and activate a Python virtual environment:
   ```bash
   python3 -m venv .venv && source .venv/bin/activate && pip install --upgrade pip
   ```

Or alternatively, [install conda on your machine](https://conda.io/projects/conda/en/latest/user-guide/install/index.html). Then, create a Conda environment named "executorch".
   ```bash
   conda create -yn executorch python=3.10.0 && conda activate executorch
   ```

## Install ExecuTorch pip package from Source
   ```bash
   # Install ExecuTorch pip package and its dependencies, as well as
   # development tools like CMake.
   # If developing on a Mac, make sure to install the Xcode Command Line Tools first.
   ./install_executorch.sh
   ```

   Not all backends are built into the pip wheel by default. You can link these missing/experimental backends by turning on the corresponding cmake flag. For example, to include the MPS backend:

  ```bash
  CMAKE_ARGS="-DEXECUTORCH_BUILD_MPS=ON" ./install_executorch.sh
  ```

   For development mode, run the command with `--editable`, which allows us to modify Python source code and see changes reflected immediately.
   ```bash
   ./install_executorch.sh --editable

   # Or you can directly do the following if dependencies are already installed
   # either via a previous invocation of `./install_executorch.sh` or by explicitly installing requirements via `./install_requirements.sh` first.
   pip install -e .
   ```

   If C++ files are being modified, you will still have to reinstall ExecuTorch from source.

> **_WARNING:_**
> Some modules can't be imported directly in editable mode. This is a known [issue](https://github.com/pytorch/executorch/issues/9558) and we are actively working on a fix for this. To workaround this:
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

## Build ExecuTorch C++ runtime from source

ExecuTorch's CMake build system covers the pieces of the runtime that are
likely to be useful to embedded systems users.

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
- `executor_runner`: An example tool that runs a `.pte` program file using all
  `1` values as inputs, and prints the outputs to stdout. It is linked with
  `libportable_kernels.a`, so the program may use any of the operators it
  implements.


### Configure the CMake build

Follow these steps after cloning or pulling the upstream repo, since the build
dependencies may have changed.

```bash
# cd to the root of the executorch repo
cd executorch

# Clean and configure the CMake build system. It's good practice to do this
# whenever cloning or pulling the upstream repo.
./install_executorch.sh --clean
(mkdir cmake-out && cd cmake-out && cmake ..)
```

Once this is done, you don't need to do it again until you pull from the upstream repo again, or if you modify any CMake-related files.

### CMake build options

The release build offers optimizations intended to improve performance and reduce binary size. It disables program verification and executorch logging, and adds optimizations flags.
```bash
-DCMAKE_BUILD_TYPE=Release
```

To further optimize the release build for size, use both:
```bash
-DCMAKE_BUILD_TYPE=Release \
-DEXECUTORCH_OPTIMIZE_SIZE=ON
```

See [CMakeLists.txt](https://github.com/pytorch/executorch/blob/main/CMakeLists.txt)

### Build the runtime components

Build all targets with

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

## Use an example binary `executor_runner` to execute a .pte file

First, generate a .pte file, either by exporting an example model or following
the instructions in [Model Export and Lowering](using-executorch-export.md).

To generate a simple model file, run the following command from the ExecuTorch directory. It
will create a file named "add.pte" in the current directory.
```
python -m examples.portable.scripts.export --model_name="add"
```
Then, pass it to the command line tool:
```bash
./cmake-out/executor_runner --model_path add.pte
```

You should see the message "Model executed successfully" followed
by the output values.

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
## Build ExecuTorch for Windows

This document outlines the current known working build instructions for building and validating ExecuTorch on a Windows machine.

This demo uses the
[MobileNet v2](https://pytorch.org/vision/main/models/mobilenetv2.html) model to classify images using the [XNNPACK](https://github.com/google/XNNPACK) backend.

Note that all commands should be executed on Windows powershell in administrator mode.

### Pre-requisites

#### 1. Install Miniconda for Windows
Install miniconda for Windows from the [official website](https://docs.conda.io/en/latest/miniconda.html).

#### 2. Install Git for Windows
Install Git for Windows from the [official website](https://git-scm.com/download/win).

#### 3. Install ClangCL for Windows
Install ClangCL for Windows from the [official website](https://learn.microsoft.com/en-us/cpp/build/clang-support-msbuild?view=msvc-170).


### Create the Conda Environment
To check if conda is detected by the powershell prompt, try `conda list` or `conda --version`

If conda is not detected, you could run the powershell script for conda named `conda-hook.ps1`.
To verify that Conda is available in the in the powershell environment, run try `conda list` or `conda --version`. 
If Conda is not available, run conda-hook.ps1 as follows:
```bash
$miniconda_dir\\shell\\condabin\\conda-hook.ps1
```
where `$miniconda_dir` is the directory where you installed miniconda
This is `“C:\Users\<username>\AppData\Local”` by default.

#### Create and activate the conda environment:
```bash
conda create -yn et python=3.12
conda activate et
```

### Check Symlinks
Set the following environment variable to enable symlinks:
```bash
git config --global core.symlinks true
```

### Set up ExecuTorch
Clone ExecuTorch from the [official GitHub repository](https://github.com/pytorch/executorch).

```bash
git clone --recurse -submodules https://github.com/pytorch/executorch.git
```

### Run the Setup Script

Currently, there are a lot of components that are not buildable on Windows. The below instructions install a very minimal ExecuTorch which can be used as a sanity check.

#### Move into the `executorch` directory
```bash
cd executorch
```

#### (Optional) Run a --clean script prior to running the .bat file.
```bash
./install_executorch.bat --clean
```

#### Run the setup script.
You could run the .bat file or the python script.
```bash
./install_executorch.bat
# OR
# python install_executorch.py
```

### Export MobileNet V2

Create the following script named export_mv2.py

```bash
from torchvision.models import mobilenet_v2
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights

mv2 = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT) # This is torch.nn.Module

import torch
from executorch.exir import to_edge
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

model = mv2.eval() # turn into evaluation mode

example_inputs = (torch.randn((1, 3, 224, 224)),) # Necessary for exporting the model

exported_graph = torch.export.export(model, example_inputs) # Core Aten graph

edge = to_edge(exported_graph) # Edge Dialect

edge_delegated = edge.to_backend(XnnpackPartitioner()) # Parts of the graph are delegated to XNNPACK

executorch_program = edge_delegated.to_executorch() # ExecuTorch program

pte_path = "mv2_xnnpack.pte"

with open(pte_path, "wb") as file:
    executorch_program.write_to_file(file) # Serializing into .pte file
```

#### Run the export script to create a `mv2_xnnpack.pte` file.

```bash
python .\\export_mv2.py
```

### Build and Install C++ Libraries + Binaries
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
  -DEXECUTORCH_BUILD_KERNELS_CUSTOM=ON `
  -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON `
  -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON `
  -DEXECUTORCH_ENABLE_LOGGING=ON `
  -T ClangCL `
  -Bcmake-out; `
cmake --build cmake-out -j64 --target install --config Release
```
where `$miniconda_dir` is the directory where you installed miniconda
This is `“C:\Users\<username>\AppData\Local”` by default.

### Run Mobilenet V2 model with XNNPACK delegation

```bash
.\\cmake-out\\backends\\xnnpack\\Release\\xnn_executor_runner.exe --model_path=.\\mv2_xnnpack.pte
```

The expected output would print a tensor of size 1x1000, containing values of class scores.

```bash
Output 0: tensor(sizes=[1, 1000], [
  -0.50986, 0.30064, 0.0953904, 0.147726, 0.231205, 0.338555, 0.206892, -0.0575775, … ])
```

Congratulations! You've successfully set up ExecuTorch on your Windows device and ran a MobileNet V2 model.
Now, you can explore and enjoy the power of ExecuTorch on your own Windows device!

## Cross compilation

Following are instruction on how to perform cross compilation for Android and iOS.

### Android

#### Building executor_runner shell binary
- Prerequisite: [Android NDK](https://developer.android.com/ndk), choose one of the following:
  - Option 1: Download Android Studio by following the instructions to [install ndk](https://developer.android.com/studio/projects/install-ndk).
  - Option 2: Download Android NDK directly from [here](https://developer.android.com/ndk/downloads).

Assuming Android NDK is available, run:
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

#### Building AAR for app integration from source
- Prerequisite: Android NDK from the previous section, and Android SDK (Android Studio is recommended).

Assuming Android NDK and SDK is available, run:
```bash
export ANDROID_ABIS=arm64-v8a
export BUILD_AAR_DIR=aar-out
mkdir -p $BUILD_AAR_DIR
sh scripts/build_android_library.sh
```

This script will build the AAR, which contains the Java API and its corresponding JNI library. Please see
[this documentation](using-executorch-android.md#using-aar-file) for usage.

### iOS

For iOS we'll build [frameworks](https://developer.apple.com/documentation/xcode/creating-a-multi-platform-binary-framework-bundle) instead of static libraries, that will also contain the public headers inside.

1. Install Xcode from the
[Mac App Store](https://apps.apple.com/app/xcode/id497799835) and then install
the Command Line Tools using the terminal:

```bash
xcode-select --install
```

2. Build the frameworks:

```bash
./scripts/build_apple_frameworks.sh
```

Run the above command with `--help` flag to learn more on how to build additional backends
(like [Core ML](backends-coreml.md), [MPS](backends-mps.md) or XNNPACK), etc.
Note, some backends may require additional dependencies and certain versions of Xcode and iOS.

3. Copy over the generated `.xcframework` bundles to your Xcode project, link them against
your targets and don't forget to add an extra linker flag `-all_load`.

Check out the [iOS Demo App](https://github.com/pytorch-labs/executorch-examples/tree/main/mv3/apple/ExecuTorchDemo) tutorial for more info.


## Next steps

You have successfully cross-compiled `executor_runner` binary to iOS and Android platforms. You can start exploring advanced features and capabilities. Here is a list of sections you might want to read next:

* [Selective build](kernel-library-selective-build.md) to build the runtime that links to only kernels used by the program, which can provide significant binary size savings.
* Tutorials on building [Android](https://github.com/pytorch-labs/executorch-examples/tree/main/dl3/android/DeepLabV3Demo#executorch-android-demo-app) and [iOS](https://github.com/pytorch-labs/executorch-examples/tree/main/mv3/apple/ExecuTorchDemo) demo apps.
* Tutorials on deploying applications to embedded devices such as [ARM Cortex-M/Ethos-U](backends-arm-ethos-u.md) and [XTensa HiFi DSP](backends-cadence.md).
