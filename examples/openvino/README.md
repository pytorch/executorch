# OpenVINO Backend Examples

This guide provides detailed instructions on how to export models for Executorch and execute them on the OpenVINO backend. The examples demonstrate how to export a model, load a model, prepare input tensors, execute inference, and save the output results.

## Directory Structure

Below is the layout of the `examples/openvino` directory, which includes the necessary files for the example applications:

```
examples/openvino
├── aot                                 # Directory with scripts and instructions for AoT export
    ├── README.md                       # Instructions to export models to '.pte'
    └── aot_openvino_compiler.py        # Example script for AoT export
├── executor_runner                     # Directory with examples for C++ execution
    └── openvino_executor_runner.cpp    # Example C++ file for execution
├── CMakeLists.txt                      # CMake build configuration to build examples
├── README.md                           # Documentation for examples (this file)
└── openvino_build_example.sh           # Script to build examples for openvino backend
```

# Build Instructions for Examples

## Environment Setup
Follow the [instructions](../../backends/openvino/README.md) of **Prerequisites** and **Setup** in `backends/openvino/README.md` to set up the OpenVINO backend.

## AOT step:
Refer to the [README.md](aot/README.md) in the `aot` folder for detailed instructions on exporting deep learning models from various model suites (TIMM, Torchvision, Hugging Face) to openvino backend using Executorch. Users can dynamically specify the model, input shape, and target device. 

Below is an example to export a ResNet50 model from Torchvision model suite for CPU device with an input shape of `[1, 3, 256, 256]`

```bash
cd aot
python aot_openvino_compiler.py --suite torchvision --model resnet50 --input_shape "(1, 3, 256, 256)" --device CPU
```
The exported model will be saved as 'resnet50.pte' in the current directory.

## Build OpenVINO Examples
Build the backend and the examples by executing the script:
```bash
./openvino_build_example.sh
```
The executable is saved in `<executorch_root>/cmake-openvino-out/examples/openvino/`

### Run the example

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
