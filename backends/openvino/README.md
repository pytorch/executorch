# OpenVINO Backend for ExecuTorch
The OpenVINO backend enables optimized execution of deep learning models on Intel hardware, leveraging Intel's [OpenVINO toolkit](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html) for inference acceleration.

## Supported Hardware

OpenVINO backend supports the following hardware:

- Intel CPUs
- Intel integrated GPUs
- Intel discrete GPUs
- Intel NPUs

For more information on the supported hardware, please refer to [OpenVINO System Requirements](https://docs.openvino.ai/2025/about-openvino/release-notes-openvino/system-requirements.html) page.

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
- Ensure that you are inside `executorch/backends/openvino/scripts` directory. The following command builds and installs ExecuTorch with the OpenVINO backend, and also compiles the C++ runtime binaries into `<executorch_root>/cmake-out` for quick inference testing.
   ```bash
   openvino_build.sh
   ```
- Optionally, `openvino_build.sh` script can be used to build python package or C++ bineries seperately.

   **Build OpenVINO Backend Python Package with Pybindings**: To build and install the OpenVINO backend Python package with Python bindings, run the `openvino_build.sh` script with the `--enable_python` argument. This will compile and install the ExecuTorch Python package with the OpenVINO backend into your Python environment. This option will also enable python bindings required to execute OpenVINO backend tests and `aot_optimize_and_infer.py` script inside `executorch/examples/openvino` folder.
     ```bash
   ./openvino_build.sh --enable_python
   ```
   **Build C++ Runtime Libraries for OpenVINO Backend**: Run the `openvino_build.sh` script with the `--cpp_runtime` argument to build C++ runtime libraries into `<executorch_root>/cmake-out` folder. `<executorch_root>/cmake-out/backends/openvino/openvino_executor_runner` binary file can be used for quick inferencing with vision models.
     ```bash
   ./openvino_build.sh --cpp_runtime
   ```
   **Build C++ Llama Runner**: This step requires first building the C++ runtime libraries by following the previous instructions. Then, run `openvino_build.sh` script with the `--llama_runner` argument to compile the llama runner to execute inference with models exported using `export_llama`. The compiled binary file is located in `<executorch_root>/cmake-out/examples/models/llama/llama_main`.
     ```bash
   ./openvino_build.sh --llama_runner
   ```

For more information about ExecuTorch environment setup, refer to the [Environment Setup](https://pytorch.org/executorch/main/getting-started-setup#environment-setup) guide.

### Run

Please refer to [README.md](../../examples/openvino/README.md) for instructions on running examples of various of models with openvino backend.
