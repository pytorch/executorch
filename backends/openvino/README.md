# OpenVINO Backend on ExecuTorch
The OpenVINO backend enables optimized execution of deep learning models on Intel hardware, leveraging Intel's [OpenVINO toolkit](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html) for inference acceleration.

## Supported Hardware

OpenVINO backend supports the following hardware:

- Intel CPUs
- Intel integrated GPUs
- Intel discrete GPUs
- Intel NPUs

## Build Instructions

### Prerequisites

Before you begin, ensure you have openvino installed and configured on your system:

```bash
git clone -b executorch_ov_backend https://github.com/ynimmaga/openvino
cd openvino
git submodule update --init --recursive
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_PYTHON=ON -DENABLE_WHEEL=ON
make -j<N>

cd ../..
cmake --install build --prefix <your_preferred_install_location>
cd <your_preferred_install_location>
source setupvars.sh
```

### Setup

Follow the steps below to setup your build environment:

1. **Setup ExecuTorch Environment**: Refer to the [Setting up ExecuTorch](https://pytorch.org/executorch/stable/getting-started-setup) guide for detailed instructions on setting up the ExecuTorch environment.

2. **Setup OpenVINO Backend Environment**
- Install the dependent libs. Ensure that you are inside backends/openvino/ directory
   ```bash
   pip install -r requirements.txt
   ```

3. Navigate to `scripts/` directory.

4. **Build OpenVINO Backend**: Once the prerequisites are in place, run the `openvino_build.sh` script to start the build process, OpenVINO backend will be built under `cmake-openvino-out/backends/openvino/` as `libneuron_backend.so`

   ```bash
   ./openvino_build.sh
   ```

### Run

Please refer to `executorch/examples/openvino/` for aot optimization and execution examples of various of models.
