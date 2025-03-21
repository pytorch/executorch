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
* [Setting up ExecuTorch](getting-started-setup.md)
* [Building ExecuTorch with CMake](runtime-build-and-cross-compilation.md)
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

## Instructions for Building OpenVINO Backend

### Prerequisites

Before you begin, ensure you have openvino installed and configured on your system:


```bash
git clone https://github.com/openvinotoolkit/openvino.git
cd openvino && git checkout releases/2025/1
git submodule update --init --recursive
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_PYTHON=ON
make -j<N>

cd ..
cmake --install build --prefix <your_preferred_install_location>
cd <your_preferred_install_location>
source setupvars.sh
```
Note: The OpenVINO backend is not yet supported in the current OpenVINO release. Therefore, it is recommended to build from source.

### Setup

Follow the steps below to setup your build environment:

1. **Setup ExecuTorch Environment**: Refer to the [Environment Setup](https://pytorch.org/executorch/stable/getting-started-setup#environment-setup) guide for detailed instructions on setting up the ExecuTorch environment.

2. **Setup OpenVINO Backend Environment**
- Install the dependent libs. Ensure that you are inside `executorch/backends/openvino/` directory
   ```bash
   pip install -r requirements.txt
   ```

3. Navigate to `scripts/` directory.

4. **Build OpenVINO Backend**: Once the prerequisites are in place, run the `openvino_build.sh` script to start the build process, OpenVINO backend will be built under `cmake-openvino-out/backends/openvino/` as `libopenvino_backend.so`

   ```bash
   ./openvino_build.sh
   ```

## Build Instructions for Examples

### AOT step:
Refer to the [README.md](../../examples/openvino/aot/README.md) in the `executorch/examples/openvino/aot` folder for detailed instructions on exporting deep learning models from various model suites (TIMM, Torchvision, Hugging Face) to openvino backend using Executorch. Users can dynamically specify the model, input shape, and target device. 

Below is an example to export a ResNet50 model from Torchvision model suite for CPU device with an input shape of `[1, 3, 256, 256]`

```bash
cd executorch/examples/openvino/aot
python aot_openvino_compiler.py --suite torchvision --model resnet50 --input_shape "(1, 3, 256, 256)" --device CPU
```
The exported model will be saved as 'resnet50.pte' in the current directory.

### Build C++ OpenVINO Examples
Build the backend and the examples by executing the script:
```bash
./openvino_build_example.sh
```
The executable is saved in `<executorch_root>/cmake-openvino-out/examples/openvino/`

Now, run the example using the executable generated in the above step. The executable requires a model file (`.pte` file generated in the aot step), number of inference iterations, and optional input/output paths.

#### Command Syntax:

```
cd ../../cmake-openvino-out/examples/openvino

./openvino_executor_runner \
    --model_path=<path_to_model> \
    --num_iter=<iterations> \
    [--input_list_path=<path_to_input_list>] \
    [--output_folder_path=<path_to_output_folder>]
```
#### Command-Line Arguments

- `--model_path`: (Required) Path to the model serialized in `.pte` format.
- `--num_iter`: (Optional) Number of times to run inference (default: 1).
- `--input_list_path`: (Optional) Path to a file containing the list of raw input tensor files.
- `--output_folder_path`: (Optional) Path to a folder where output tensor files will be saved.

#### Example Usage

Run inference with a given model for 10 iterations and save outputs:

```
./openvino_executor_runner \
    --model_path=model.pte \
    --num_iter=10 \
    --output_folder_path=outputs/
```

Run inference with an input tensor file:

```
./openvino_executor_runner \
    --model_path=model.pte \
    --num_iter=5 \
    --input_list_path=input_list.txt \
    --output_folder_path=outputs/
```

## Supported model list

### TODO

## FAQ

If you encounter any issues while reproducing the tutorial, please file a github
issue on ExecuTorch repo and tag use `#openvino` tag
