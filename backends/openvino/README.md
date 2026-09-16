# OpenVINO Backend for ExecuTorch
The OpenVINO backend enables optimized execution of deep learning models on Intel hardware, leveraging Intel's [OpenVINO toolkit](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html) for inference acceleration.

## Supported Hardware

OpenVINO backend supports the following hardware:

- Intel CPUs
- Intel integrated GPUs
- Intel discrete GPUs
- Intel NPUs

For more information on the supported hardware, please refer to [OpenVINO System Requirements](https://docs.openvino.ai/2025/about-openvino/release-notes-openvino/system-requirements.html) page.

## Quick Start (pip wheel)

On Linux and Windows, the OpenVINO backend is included in the ExecuTorch pip wheel. Install the OpenVINO runtime to activate it:

```bash
pip install executorch[openvino]
```

For now the extra installs no runtime on Python 3.15, where OpenVINO publishes no build. On Linux aarch64, OpenVINO needs glibc 2.35 or newer, as on Ubuntu 22.04 or Raspberry Pi OS Bookworm and later. On an older aarch64 system, `pip install executorch[openvino]` does not fail: pip finds no OpenVINO 2025.4 or newer for that system, so it installs an older ExecuTorch release instead. On such a system, install `executorch` without the extra, then build OpenVINO from source as described below.

The backend automatically discovers the OpenVINO C library from the pip-installed package, so no library search path setup is needed.

If auto-discovery fails (e.g. non-standard install), set `OPENVINO_LIB_PATH` to the full path of the OpenVINO C library: a `libopenvino_c.so` file on Linux, or `openvino_c.dll` on Windows. For the pip package on Linux:

```bash
export OPENVINO_LIB_PATH="$(python3 -c 'import glob, openvino, os; print(sorted(glob.glob(os.path.join(os.path.dirname(openvino.__file__), "libs", "libopenvino_c.so*")))[0])')"
```

On Windows, in PowerShell:

```powershell
$env:OPENVINO_LIB_PATH = python -c "import openvino, os; print(os.path.join(os.path.dirname(openvino.__file__), 'libs', 'openvino_c.dll'))"
```

On Windows, `PATH` and the current folder are not searched for the library or for its dependencies, such as `tbb12.dll`. Those must sit beside `openvino_c.dll`, as they do in the pip package, or in a folder added with `os.add_dll_directory` (`AddDllDirectory` in C++). Importing the `openvino` Python package adds its folders this way, including the ones Intel's `setupvars.bat` lists.

Verify the backend is available:

```python
from executorch.extension.pybindings.portable_lib import (
    _get_registered_backend_names,
)
print(_get_registered_backend_names())
# Should include 'OpenvinoBackend'
```

## Precision

When the device is `CPU`, a program computes in float32 by default. Without this default,
OpenVINO would compute in bfloat16 on x86 CPUs with bfloat16 units, which keeps only two to
three significant digits, and in float16 on ARM CPUs, which keeps about three.
Both are faster. To trade accuracy for speed, ask for the faster precision when
lowering:

```python
from executorch.backends.openvino.partitioner import OpenvinoPartitioner
from executorch.exir.backend.compile_spec_schema import CompileSpec

OpenvinoPartitioner(
    [CompileSpec("device", b"CPU"), CompileSpec("inference_precision", b"bf16")]
)
```

On an ARM CPU, ask for `f16` instead: there float32 is about 1.8x slower than
float16, as measured on an Orin Nano.

The value is OpenVINO's `INFERENCE_PRECISION_HINT`: `f32`, `f16` or `bf16`. An
empty value means the default, and any other value fails at lowering. A CPU
without float16 support computes an `f16` program in bfloat16 if it has bfloat16
units, as some x86 CPUs do, and in float32 otherwise. OpenVINO has no
bfloat16 on ARM CPUs, so there `bf16` keeps the ARM default, float16 where the
CPU supports it. Which x86 CPUs compute in bfloat16 is up to OpenVINO; see its
[Precision Control](https://docs.openvino.ai/2026/openvino-workflow/running-inference/optimize-inference/precision-control.html)
page. Other devices, such as GPUs, keep OpenVINO's default unless the option is
set.

The default also applies to a program exported in float16: on a CPU it computes
in float32 unless you lower it with `inference_precision`.

On x86 CPUs, OpenVINO applies the precision when the runtime loads a program.
So a program lowered without `inference_precision` computes in float32, even
one exported with an ExecuTorch release that did not have this default. On an
x86 CPU with bfloat16 units, float32 is slower and can use more memory; export it
again with `inference_precision` set to `bf16` to get the earlier speed. Other
x86 CPUs already computed such a program in float32. On ARM CPUs, OpenVINO applies the
precision when the program is lowered, so a program exported with such an
earlier release keeps computing in float16 until it is exported again.

OpenVINO's CPU plugin computes 64-bit integers in 32 bits. An integer result
outside the int32 range wraps around without an error. This applies to every
int64 tensor, including the int64 result of `cumsum` and `prod` over smaller
integer types. In OpenVINO 2026.4.1, the CPU plugin also computes some int32 and
int64 ops in float32. On x86 CPUs these are `sum`, `amax`, `amin`, `max` and
`min`. On ARM CPUs they are element-wise ops such as `add`, `maximum` and
`clamp`, and the same reductions along any dimension but the last. Once a value
or a running total is above 16,777,216 (2^24) or below -16,777,216, such a
result can be off, even inside the int32 range. For example, on x86 `sum` over a
tensor that holds 16,777,216 and 1 gives 16,777,216, and so does `amax` over one
that holds 16,777,217.

In OpenVINO 2026.4.1, the CPU plugin's `min(dim)` and `max(dim)` can return a
later index on ties than PyTorch, which returns the first one. This happens with
float32 as well as integer dtypes such as int32 and uint8.

## Directory Structure

```
executorch
├── backends
│   └── openvino
│       ├── quantizer
│           ├── observers
│               └── nncf_observers.py
│           ├── __init__.py
│           └── quantizer.py
│       ├── runtime
│           ├── OpenvinoApi.h
│           ├── OpenvinoBackend.cpp
│           └── OpenvinoBackend.h
│       ├── scripts
│           └── openvino_build.sh
│       ├── tests
│       ├── CMakeLists.txt
│       ├── README.md
│       ├── __init__.py
│       ├── partitioner.py
│       ├── preprocess.py
│       └── requirements.txt
└── examples
    └── openvino
        ├── aot_optimize_and_infer.py
        └── README.md
```

## Build Instructions

Choose the build flow for your platform. The Linux path uses `openvino_build.sh`; the Windows path provides equivalent PowerShell commands.

### Common Setup (All Platforms)

Before you begin, ensure OpenVINO 2025.4 or newer is installed and configured on your system.

1. Clone ExecuTorch:

   ```bash
   git clone --recurse-submodules https://github.com/pytorch/executorch.git
   cd executorch
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   ```

   On Windows PowerShell:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   ```

3. Install OpenVINO backend Python requirements:

   ```bash
   python -m pip install -r backends/openvino/requirements.txt
   ```

For more information about ExecuTorch environment setup, refer to the [Environment Setup](https://pytorch.org/executorch/main/getting-started-setup#environment-setup) guide.

### Linux Build (openvino_build.sh)

#### OpenVINO Runtime Setup

Use one of the following options:

- OpenVINO release package:

  1. Download the release package from [OpenVINO install docs](https://docs.openvino.ai/2025/get-started/install-openvino.html) (OpenVINO Archives).
  2. Extract and initialize environment variables:

     ```bash
     tar -zxf openvino_toolkit_<your_release_configuration>.tgz
     cd openvino_toolkit_<your_release_configuration>
     source setupvars.sh
     ```

- Optional: build OpenVINO from source:

  ```bash
  git clone https://github.com/openvinotoolkit/openvino.git
  cd openvino
  git submodule update --init --recursive
  sudo ./install_build_dependencies.sh
  mkdir build && cd build
  cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_PYTHON=ON
  make -j$(nproc)

  cd ..
  cmake --install build --prefix <your_preferred_install_location>
  cd <your_preferred_install_location>
  source setupvars.sh
  ```

For more information about OpenVINO build, refer to the [OpenVINO Build Instructions](https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/build_linux.md).

#### Build Commands

From `executorch/backends/openvino/scripts`:

```bash
./openvino_build.sh
```

Optional build variants:

- Python package with pybindings:

  ```bash
  ./openvino_build.sh --enable_python
  ```

- C++ runtime libraries:

  ```bash
  ./openvino_build.sh --cpp_runtime
  ```

- C++ runtime libraries with LLM extension:

  ```bash
  ./openvino_build.sh --cpp_runtime_llm
  ```

### Windows Build (PowerShell)

This flow is equivalent to `openvino_build.sh` and does not require Bash shell scripts.

#### Windows Prerequisites

1. Windows 10 or 11 (x64)
2. Visual Studio 2022 with C++ build tools (Desktop development with C++)
3. CMake (3.20+) and Ninja in `PATH`
4. Python 3.10+ in `PATH`
5. Git in `PATH`

Enable symlink support for Git:

```powershell
git config --global core.symlinks true
```

If this is a fresh setup, re-clone the repository after enabling symlinks.

Use Developer PowerShell for VS 2022 (recommended), or a PowerShell session where MSVC tools are already initialized.

If script activation is blocked:

```powershell
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
```

To persist for future sessions:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### OpenVINO Runtime Setup (Windows)

Choose one setup path:

- Pip package:

  ```powershell
  python -m pip install openvino
  ```

- OpenVINO archive package (official 2026 flow):

  Follow the official guide: [Install OpenVINO Runtime on Windows from an Archive File](https://docs.openvino.ai/2026/get-started/install-openvino/install-openvino-archive-windows.html).

  Step 1: Download and install OpenVINO core components.

  1. Create the Intel installation folder (recommended location):

    ```powershell
    mkdir "C:\Program Files (x86)\Intel"
    ```

  2. Download the archive (example):

    ```powershell
    cd $HOME\Downloads
    curl -L https://storage.openvinotoolkit.org/repositories/openvino/packages/2026.2.1/windows/openvino_toolkit_windows_2026.2.1.21919.ede283a88e3_x86_64.zip --output openvino_2026.2.1.zip
    ```

  3. Extract, rename, and move it under `C:\Program Files (x86)\Intel`:

    ```powershell
    tar -xf openvino_2026.2.1.zip
    ren openvino_toolkit_windows_2026.2.1.21919.ede283a88e3_x86_64 openvino_2026.2.1
    move openvino_2026.2.1 "C:\Program Files (x86)\Intel"
    ```

  4. Optional (Python API): install Python requirements from the OpenVINO package:

    ```powershell
    cd "C:\Program Files (x86)\Intel\openvino_2026.2.1"
    python -m pip install -r .\python\requirements.txt
    ```

  5. Optional: create a stable symbolic link for easier upgrades:

    ```cmd
    cd C:\Program Files (x86)\Intel
    mklink /D openvino_2026 openvino_2026.2.1
    ```

  Step 2: Configure environment variables for the current shell.

  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
  . "C:\Program Files (x86)\Intel\openvino_2026\setupvars.ps1"
  ```

  If you do not create the symbolic link, call `setupvars.ps1` from the full versioned folder path.

- Optional: build OpenVINO from source and install it locally:

  ```powershell
  git clone https://github.com/openvinotoolkit/openvino.git
  cd openvino
  git submodule update --init --recursive

  cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_PYTHON=ON
  cmake --build build --config Release --parallel

  cmake --install build --prefix C:\openvino-install
  & "C:\openvino-install\setupvars.ps1"
  ```

  Update `C:\openvino-install` to your preferred install directory.

For additional source build details, refer to the [OpenVINO Build Instructions](https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/build_windows.md).

#### Build Commands (Windows)

Before running CMake commands, verify the active Python comes from `.venv`:

```powershell
.\.venv\Scripts\Activate.ps1
where python
python --version
```

Run all commands from the ExecuTorch repository root.

##### Option A: Build all

```powershell
.\install_executorch.bat --clean

$env:CMAKE_ARGS = "-DEXECUTORCH_BUILD_OPENVINO=ON -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON -DEXECUTORCH_BUILD_XNNPACK=OFF -DEXECUTORCH_BUILD_VULKAN=OFF"
$env:CMAKE_BUILD_ARGS = "--target openvino_backend"
.\install_executorch.bat --use-pt-pinned-commit

if (Test-Path cmake-out) { Remove-Item -Recurse -Force cmake-out }

cmake -DCMAKE_BUILD_TYPE=Release `
   -DEXECUTORCH_BUILD_OPENVINO=ON `
   -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON `
   -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON `
   -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON `
   -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON `
   -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON `
   -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON `
   -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=ON `
   -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON `
   -DCMAKE_INSTALL_PREFIX=cmake-out `
   -B cmake-out

cmake --build cmake-out --target install --config Release -j $env:NUMBER_OF_PROCESSORS
```

##### Option B: Build only C++ runtime

```powershell
if (Test-Path cmake-out) { Remove-Item -Recurse -Force cmake-out }

cmake -DCMAKE_BUILD_TYPE=Release `
   -DEXECUTORCH_BUILD_OPENVINO=ON `
   -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON `
   -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON `
   -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON `
   -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON `
   -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON `
   -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON `
   -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=ON `
   -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON `
   -DCMAKE_INSTALL_PREFIX=cmake-out `
   -B cmake-out

cmake --build cmake-out --target install --config Release -j $env:NUMBER_OF_PROCESSORS
```

##### Option C: Build only C++ runtime with LLM extension

```powershell
if (Test-Path cmake-out) { Remove-Item -Recurse -Force cmake-out }

cmake -DCMAKE_BUILD_TYPE=Release `
   -DEXECUTORCH_BUILD_OPENVINO=ON `
   -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON `
   -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON `
   -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON `
   -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON `
   -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON `
   -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON `
   -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=ON `
   -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON `
   -DEXECUTORCH_BUILD_EXTENSION_LLM=ON `
   -DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=ON `
   -DCMAKE_INSTALL_PREFIX=cmake-out `
   -B cmake-out

cmake --build cmake-out --target install --config Release -j $env:NUMBER_OF_PROCESSORS
```

##### Option D: Build only Python package with pybindings

```powershell
python -m pip install -r backends/openvino/requirements.txt
.\install_executorch.bat --clean
$env:CMAKE_ARGS = "-DEXECUTORCH_BUILD_OPENVINO=ON -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON -DEXECUTORCH_BUILD_XNNPACK=OFF -DEXECUTORCH_BUILD_VULKAN=OFF"
$env:CMAKE_BUILD_ARGS = "--target openvino_backend"
.\install_executorch.bat --use-pt-pinned-commit
```

### Verify Backend Registration (All Platforms)

```python
from executorch.extension.pybindings.portable_lib import _get_registered_backend_names
print(_get_registered_backend_names())
```

Expected output should include `OpenvinoBackend`.


### Run

Please refer to [README.md](../../examples/openvino/README.md) for instructions on running examples of models with OpenVINO backend.
