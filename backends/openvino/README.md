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

The backend automatically discovers the OpenVINO C library from the pip-installed package — no `LD_LIBRARY_PATH` (Linux) or `PATH` (Windows) setup is needed.

If auto-discovery fails (e.g. non-standard install), you can point to the library explicitly:

**Linux:**
```bash
export OPENVINO_LIB_PATH=$(python3 -c "import openvino, os; print(os.path.join(os.path.dirname(openvino.__file__), 'libs', 'libopenvino_c.so'))")
```

**Windows (PowerShell):**
```powershell
$env:OPENVINO_LIB_PATH = python -c "import openvino, os; print(os.path.join(os.path.dirname(openvino.__file__), 'libs', 'openvino_c.dll'))"
```

Verify the backend is available:

```python
from executorch.extension.pybindings.portable_lib import (
    _get_registered_backend_names,
)
print(_get_registered_backend_names())
# Should include 'OpenvinoBackend'
```

## Directory Structure

```
executorch
├── backends
│   └── openvino
│       ├── _passes
│       │   ├── __init__.py
│       │   └── decompose_floor_divide_pass.py
│       ├── quantizer
│       │   ├── __init__.py
│       │   ├── llm_compression.py
│       │   ├── observers.py
│       │   └── quantizer.py
│       ├── runtime
│       │   ├── OpenvinoApi.h
│       │   ├── OpenvinoBackend.cpp
│       │   └── OpenvinoBackend.h
│       ├── scripts
│       │   └── openvino_build.sh
│       ├── test
│       │   └── tester
│       │       ├── __init__.py
│       │       └── tester.py
│       ├── tests
│       │   ├── models
│       │   │   └── test_classification.py
│       │   ├── ops
│       │   │   ├── base_openvino_op_test.py
│       │   │   └── test_*.py
│       │   ├── quantizer
│       │   │   ├── synthetic_test_models.py
│       │   │   └── test_llm_compression.py
│       │   ├── README.md
│       │   └── test_runner.py
│       ├── CMakeLists.txt
│       ├── README.md
│       ├── __init__.py
│       ├── partitioner.py
│       ├── preprocess.py
│       └── requirements.txt
└── examples
    └── openvino                          # See examples/openvino/README.md
```

## Build Instructions

### Setup

Follow the steps below to setup your build environment:


1. **Create a Virtual Environment**
- Create a virtual environment and activate it by executing the commands below.

   **Linux:**
   ```bash
   python -m venv env
   source env/bin/activate
   ```

   **Windows (PowerShell):**
   ```powershell
   python -m venv env
   env\Scripts\Activate.ps1
   ```
2. **Clone ExecuTorch Repository from GitHub**
- On Windows, enable symlinks before cloning. Refer to [Building from Source](https://docs.pytorch.org/executorch/main/using-executorch-building-from-source.html#environment-setup) for more details.
- Clone Executorch repository by executing the command below.
   ```bash
   git clone --recurse-submodules https://github.com/pytorch/executorch.git
   ```
3. **Build ExecuTorch with OpenVINO Backend**
- The following commands build and install ExecuTorch with the OpenVINO backend into `cmake-out`.

   **Linux:**
   ```bash
   cmake -DCMAKE_INSTALL_PREFIX=cmake-out \
         -DCMAKE_BUILD_TYPE=Release \
         -DEXECUTORCH_BUILD_OPENVINO=ON \
         -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
         -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
         -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
         -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON \
         -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
         -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
         -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=ON \
         -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
         -Bcmake-out
   cmake --build cmake-out --target install --config Release -j $(nproc)
   ```

   **Windows (PowerShell):**
   ```powershell
   cmake -DCMAKE_INSTALL_PREFIX=cmake-out `
         -DCMAKE_BUILD_TYPE=Release `
         -DEXECUTORCH_BUILD_OPENVINO=ON `
         -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON `
         -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON `
         -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON `
         -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON `
         -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON `
         -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON `
         -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=ON `
         -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON `
         -Bcmake-out
   cmake --build cmake-out --target install --config Release -j $env:NUMBER_OF_PROCESSORS
   ```

   To additionally build with LLM extension support, append `-DEXECUTORCH_BUILD_EXTENSION_LLM=ON -DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=ON` to the configure step.

#### Build Python Package with Pybindings

Compiles and installs the ExecuTorch Python package with the OpenVINO backend into your Python environment, enabling python bindings required to execute OpenVINO backend tests and `aot_optimize_and_infer.py` script inside `executorch/examples/openvino` folder.

**Linux:**
```bash
pip install -r backends/openvino/requirements.txt
CMAKE_ARGS="-DEXECUTORCH_BUILD_OPENVINO=ON -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON" \
CMAKE_BUILD_ARGS="--target openvino_backend" \
./install_executorch.sh --use-pt-pinned-commit
```
- On Linux, `backends/openvino/scripts/openvino_build.sh` can be used as a convenience wrapper with `--enable_python`, `--cpp_runtime`, and `--cpp_runtime_llm` options instead of running the above commands directly.

**Windows (PowerShell):**
```powershell
pip install -r backends/openvino/requirements.txt
$env:CMAKE_ARGS = "-DEXECUTORCH_BUILD_OPENVINO=ON -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON"
$env:CMAKE_BUILD_ARGS = "--target openvino_backend"
.\install_executorch.bat --use-pt-pinned-commit
```



For more information about ExecuTorch environment setup, refer to the [Environment Setup](https://pytorch.org/executorch/main/getting-started-setup#environment-setup) guide.

## Runtime Setup

OpenVINO is a runtime-only dependency — it is not required at build time. The backend discovers and loads the OpenVINO C library dynamically when first used. You can provide the library via pip (recommended) or a manual install.

### Install via pip (Recommended)

```bash
pip install openvino
```

### Use OpenVINO from Release Packages

1. Download the OpenVINO release package from [here](https://docs.openvino.ai/2025/get-started/install-openvino.html). Make sure to select your configuration and click on **OpenVINO Archives** under the distribution section to download the appropriate archive for your platform.

2. Extract the release package from the archive and set the environment variables.

   **Linux:**
   ```bash
   tar -zxf openvino_toolkit_<your_release_configuration>.tgz
   cd openvino_toolkit_<your_release_configuration>
   source setupvars.sh
   ```

   **Windows (PowerShell):**
   ```powershell
   Expand-Archive openvino_toolkit_<your_release_configuration>.zip
   cd openvino_toolkit_<your_release_configuration>
   .\setupvars.ps1
   ```

### (Optional) Build OpenVINO from Source

**Linux:**
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

**Windows (PowerShell):**
```powershell
git clone https://github.com/openvinotoolkit/openvino.git
cd openvino
git submodule update --init --recursive
mkdir build; cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_PYTHON=ON
cmake --build . --config Release -j $env:NUMBER_OF_PROCESSORS

cd ..
cmake --install build --prefix <your_preferred_install_location>
cd <your_preferred_install_location>
.\setupvars.ps1
```

For more information about OpenVINO build, refer to the [OpenVINO Build Instructions](https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/build_linux.md).

### Examples

Please refer to [README.md](../../examples/openvino/README.md) for instructions on running examples of various models with the OpenVINO backend.
