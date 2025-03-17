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

Before you begin, ensure you have openvino installed and configured on your system:

## TODO: Add instructions for support with OpenVINO release package

```bash
git clone https://github.com/openvinotoolkit/openvino.git
cd openvino && git checkout releases/2025/1
git submodule update --init --recursive
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_PYTHON=ON
make -j$(nproc)

cd ..
cmake --install build --prefix <your_preferred_install_location>
cd <your_preferred_install_location>
source setupvars.sh
```

### Setup

Follow the steps below to setup your build environment:

1. **Setup ExecuTorch Environment**: Refer to the [Environment Setup](https://pytorch.org/executorch/stable/getting-started-setup#environment-setup) guide for detailed instructions on setting up the ExecuTorch environment.

2. **Setup OpenVINO Backend Environment**
- Install the dependent libs. Ensure that you are inside `executorch/backends/openvino/` directory
   ```bash
   pip install -r requirements.txt
   ```
  Note: To achieve optimal performance with NNCF quantization, you should install the latest development version of NNCF (version 2.16.0.dev0+191b53d9 or higher).
3. Navigate to `scripts/` directory.

4. **Build OpenVINO Backend C++ Libraries and Executor Runner**: Once the prerequisites are in place, run the `openvino_build.sh` script to start the build process. By default, OpenVINO backend will be built under `cmake-out/backends/openvino/` as `libopenvino_backend.a`

   ```bash
   ./openvino_build.sh
   ```
   **Build OpenVINO Backend Python Package with Pybindings**: To build and install the OpenVINO backend Python package with Python bindings, run the `openvino_build.sh` script with the `--enable_python` argument. This will compile and install the ExecuTorch Python package with the OpenVINO backend into your Python environment. This option will also enable python bindings required to execute OpenVINO backend tests and `export_and_infer_openvino.py` script inside `executorch/examples/openvino` folder.

   ```bash
   ./openvino_build.sh --enable_python
   ```

### Run

Please refer to [README.md](../../examples/openvino/README.md) for instructions on running examples of various of models with openvino backend.
