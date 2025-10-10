# Building and Running ExecuTorch with OpenVINO Backend

In this tutorial we will walk you through the process of setting up the prerequisites, building OpenVINO backend library, exporting `.pte` models with OpenVINO optimizations, and executing the exported models on Intel hardware.

<!----This will show a grid card on the page----->
::::{grid} 2
:::{grid-item-card}  What you will learn in this tutorial:
:class-card: card-prerequisites
* In this tutorial you will learn how to lower and deploy a model with OpenVINO.
:::
:::{grid-item-card}  Tutorials we recommend you complete before this:
:class-card: card-prerequisites
* [Introduction to ExecuTorch](intro-how-it-works.md)
* [Setting up ExecuTorch](getting-started.md)
* [Building ExecuTorch with CMake](using-executorch-building-from-source.md)
:::
::::

## Introduction to OpenVINO

[OpenVINO](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html) is an open-source toolkit designed to enhance AI inference on Intel hardware by reducing latency and increasing throughput while preserving accuracy. It optimizes hardware utilization and simplifies AI development and deep learning integration across domains such as computer vision, large language models (LLMs), and generative AI.

OpenVINO is integrated as an Executorch delegate to accelerate AI applications deployed with Executorch APIs.

## Supported Hardware

OpenVINO backend supports the following hardware:

- Intel CPUs
- Intel integrated GPUs
- Intel discrete GPUs
- Intel NPUs

For more information on the supported hardware, please refer to [OpenVINO System Requirements](https://docs.openvino.ai/2025/about-openvino/release-notes-openvino/system-requirements.html) page.

## Instructions for Building OpenVINO Backend

### Prerequisites

Before you begin, ensure you have openvino installed and configured on your system:


```bash
git clone https://github.com/openvinotoolkit/openvino.git
cd openvino && git checkout releases/2025/1
git submodule update --init --recursive
sudo ./install_build_dependencies.sh
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_PYTHON=ON
make -j<N>

cd ..
cmake --install build --prefix <your_preferred_install_location>
cd <your_preferred_install_location>
source setupvars.sh
```
Note: The OpenVINO backend is not yet supported with the current OpenVINO release packages. It is recommended to build from source. The instructions for using OpenVINO release packages will be added soon.
For more information about OpenVINO build, refer to the [OpenVINO Build Instructions](https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/build_linux.md).

### Setup

Follow the steps below to setup your build environment:

1. **Setup ExecuTorch Environment**: Refer to the [Environment Setup](using-executorch-building-from-source.md#environment-setup) guide for detailed instructions on setting up the ExecuTorch environment.

2. **Setup OpenVINO Backend Environment**
- Install the dependent libs. Ensure that you are inside `executorch/backends/openvino/` directory
   ```bash
   pip install -r requirements.txt
   ```

3. Navigate to `scripts/` directory.

4. **Build OpenVINO Backend**: Once the prerequisites are in place, run the `openvino_build.sh` script to start the build process, OpenVINO backend will be built under `cmake-out/backends/openvino/` as `libopenvino_backend.a`

   ```bash
   ./openvino_build.sh
   ```

## Build Instructions for Examples

### AOT step:
Refer to the [README.md](../../examples/openvino/README.md) in the `executorch/examples/openvino` folder for detailed instructions on exporting deep learning models from various model suites (TIMM, Torchvision, Hugging Face) to openvino backend using Executorch. Users can dynamically specify the model, input shape, and target device.

Below is an example to export a ResNet50 model from Torchvision model suite for CPU device with an input shape of `[1, 3, 256, 256]`

```bash
cd executorch/examples/openvino
python aot_optimize_and_infer.py --export --suite torchvision --model resnet50 --input_shape "(1, 3, 256, 256)" --device CPU
```
The exported model will be saved as 'resnet50.pte' in the current directory.

### Build C++ OpenVINO Examples

After building the OpenVINO backend following the [instructions](#setup) above, the executable will be saved in `<executorch_root>/cmake-out/backends/openvino/`.

The executable requires a model file (`.pte` file generated in the aot step) and the number of inference executions.

#### Example Usage

Run inference with a given model for 10 executions:

```
./openvino_executor_runner \
    --model_path=model.pte \
    --num_executions=10
```



## Support

If you encounter any issues while reproducing the tutorial, please file a github
issue on ExecuTorch repo and tag use `#openvino` tag
