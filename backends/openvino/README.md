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

For now the extra installs no runtime on Python 3.15, or on Linux aarch64 with Python 3.14. OpenVINO publishes no build for Python 3.15, and its aarch64 builds for Python 3.14 need glibc 2.35 or newer.

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
without float16 support computes an `f16` program in float32. OpenVINO has no
bfloat16 on ARM CPUs, so there `bf16` keeps the ARM default, float16 where the
CPU supports it. Which x86 CPUs compute in bfloat16 is up to OpenVINO; see its
[Precision Control](https://docs.openvino.ai/2026/openvino-workflow/running-inference/optimize-inference/precision-control.html)
page. Other devices, such as GPUs, keep OpenVINO's default unless the option is
set.

The default also applies to a program exported in float16: on a CPU it computes
in float32 unless you lower it with `inference_precision`.

On x86 CPUs, OpenVINO applies the precision when the runtime loads a program.
So a program lowered without `inference_precision` computes in float32, even
one exported with an ExecuTorch release that did not have this default. That
is slower and can use more memory. Export it again with `inference_precision`
set to `bf16` to get the earlier speed. On ARM CPUs, OpenVINO applies the
precision when the program is lowered, so a program exported with such an
earlier release keeps computing in float16 until it is exported again.

OpenVINO's CPU plugin computes 64-bit integers in 32 bits. An integer result
outside the int32 range wraps around without an error. This applies to every
int64 tensor, including the int64 result of `cumsum` and `prod` over smaller
integer types.

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

### Prerequisites

Before you begin, ensure you have openvino installed and configured on your system.

### Use OpenVINO from Release Packages

1. Download the OpenVINO release package from [here](https://docs.openvino.ai/2025/get-started/install-openvino.html). Make sure to select your configuration and click on **OpenVINO Archives** under the distribution section to download the appropriate archive for your platform.

2. Extract the release package from the archive and set the environment variables.

   ```bash
   tar -zxf openvino_toolkit_<your_release_configuration>.tgz
   cd openvino_toolkit_<your_release_configuration>
   source setupvars.sh
   ```

### (Optional) Build OpenVINO from Source

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

### Setup

Follow the steps below to setup your build environment:


1. **Create a Virtual Environment**
- Create a virtual environment and activate it by executing the commands below.
   ```bash
   python -m venv env
   source env/bin/activate
   ```
2. **Clone ExecuTorch Repository from Github**
- Clone Executorch repository by executing the command below.
   ```bash
   git clone --recurse-submodules https://github.com/pytorch/executorch.git
   ```
3. **Build ExecuTorch with OpenVINO Backend**
- Ensure that you are inside `executorch/backends/openvino/scripts` directory. The following command builds and installs ExecuTorch with the OpenVINO backend, also compiles the C++ runtime libraries and binaries into `<executorch_root>/cmake-out` for quick inference testing.
   ```bash
   openvino_build.sh
   ```
- Optionally, `openvino_build.sh` script can be used to build python package or C++ libraries/binaries separately.

   **Build OpenVINO Backend Python Package with Pybindings**: To build and install the OpenVINO backend Python package with Python bindings, run the `openvino_build.sh` script with the `--enable_python` argument as shown in the below command. This will compile and install the ExecuTorch Python package with the OpenVINO backend into your Python environment. This option will also enable python bindings required to execute OpenVINO backend tests and `aot_optimize_and_infer.py` script inside `executorch/examples/openvino` folder.
     ```bash
   ./openvino_build.sh --enable_python
   ```
   **Build C++ Runtime Libraries for OpenVINO Backend**: Run the `openvino_build.sh` script with the `--cpp_runtime` flag to build the C++ runtime libraries as shown in the below command. The compiled libraries files and binaries can be found in the `<executorch_root>/cmake-out` directory. The binary located at `<executorch_root>/cmake-out/executor_runner` can be used to run inference with vision models.
     ```bash
   ./openvino_build.sh --cpp_runtime
   ```
   **Build C++ Runtime Libraries with LLM Extension**: Run the `openvino_build.sh` script with the `--cpp_runtime_llm` flag to build the C++ runtime libraries with LLM extension as shown in the below command. Use this option instead of `--cpp_runtime` for LLM extension support which is required by LLM examples.
     ```bash
   ./openvino_build.sh --cpp_runtime_llm
   ```

For more information about ExecuTorch environment setup, refer to the [Environment Setup](https://pytorch.org/executorch/main/getting-started-setup#environment-setup) guide.

### Run

Please refer to [README.md](../../examples/openvino/README.md) for instructions on running examples of various of models with openvino backend.
