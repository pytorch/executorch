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

## Directory Structure

```
executorch
├── backends
│   └── openvino
│       ├── runtime
│           ├── OpenvinoBackend.cpp
│           └── OpenvinoBackend.hpp
│       ├── scripts
│           └── openvino_build.sh
│       ├── tests
│       ├── CMakeLists.txt
│       ├── README.md
│       ├── __init__.py
│       ├── openvino_functions.yaml
│       ├── partitioner.py
│       ├── preprocess.py
│       └── requirements.txt
└── examples
│   └── openvino
│       ├── aot
│           ├── README.md
│           └── aot_openvino_compiler.py
│       └── executor_runner
│           └── openvino_executor_runner.cpp
│       ├── CMakeLists.txt
│       ├── README.md
└──     └── openvino_build_example.sh
```

## Instructions for Building OpenVINO Backend

### Prerequisites

Before you begin, ensure you have openvino installed and configured on your system:

#### TODO: Update with the openvino commit/Release tag once the changes in OpenVINO are merged
#### TODO: Add instructions for support with OpenVINO release package

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
Refer to the [README.md](aot/README.md) in the `aot` folder for detailed instructions on exporting deep learning models from various model suites (TIMM, Torchvision, Hugging Face) to openvino backend using Executorch. Users can dynamically specify the model, input shape, and target device. 

Below is an example to export a ResNet50 model from Torchvision model suite for CPU device with an input shape of `[1, 3, 256, 256]`

```bash
cd aot
python aot_openvino_compiler.py --suite torchvision --model resnet50 --input_shape "(1, 3, 256, 256)" --device CPU
```
The exported model will be saved as 'resnet50.pte' in the current directory.

#### **Arguments**
- **`--suite`** (required):  
  Specifies the model suite to use.  
  Supported values:
  - `timm` (e.g., VGG16, ResNet50)
  - `torchvision` (e.g., resnet18, mobilenet_v2)
  - `huggingface` (e.g., bert-base-uncased)

- **`--model`** (required):  
  Name of the model to export.  
  Examples:
  - For `timm`: `vgg16`, `resnet50`
  - For `torchvision`: `resnet18`, `mobilenet_v2`
  - For `huggingface`: `bert-base-uncased`, `distilbert-base-uncased`

- **`--input_shape`** (required):  
  Input shape for the model. Provide this as a **list** or **tuple**.  
  Examples:
  - `[1, 3, 224, 224]` (Zsh users: wrap in quotes)
  - `(1, 3, 224, 224)`

- **`--device`** (optional):  
  Target device for the compiled model. Default is `CPU`.  
  Examples: `CPU`, `GPU`

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
