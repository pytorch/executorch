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

Within the `aot` folder, you'll find the model export script called `aot_openvino_compiler.py`. This script allows users to export deep learning models from various model suites (TIMM, Torchvision, Hugging Face) to a openvino backend using **Executorch**. Users can dynamically specify the model, input shape, and target device.

### **Usage**

First, navigate to the `aot` directory by running the command `cd aot`. Then, refer to the instructions provided below.

#### **Command Structure**
```bash
python aot_openvino_compiler.py --suite <MODEL_SUITE> --model <MODEL_NAME> --input_shape <INPUT_SHAPE> --device <DEVICE>
```

#### **Arguments**
- **`--suite`** (required):
  Specifies the model suite to use.
  Supported values:
  - `timm` (e.g., VGG16, ResNet50)
  - `torchvision` (e.g., resnet18, mobilenet_v2)
  - `huggingface` (e.g., bert-base-uncased). NB: Quantization and validation is not supported yet.

- **`--model`** (required):
  Name of the model to export.
  Examples:
  - For `timm`: `vgg16`, `resnet50`
  - For `torchvision`: `resnet18`, `mobilenet_v2`
  - For `huggingface`: `bert-base-uncased`, `distilbert-base-uncased`

- **`--input_shape`**(optional):
  Input shape for the model. Provide this as a **list** or **tuple**.
  Examples:
  - `[1, 3, 224, 224]` (Zsh users: wrap in quotes)
  - `(1, 3, 224, 224)`

- **`--batch_size`** :
  Batch size for the validation. Default batch_size == 1.
  The dataset length must be evenly divisible by the batch size.

- **`--quantize`** (optional):
  Enable model quantization. --dataset argument is requred for the quantization. `huggingface` suite  does not supported yet.

- **`--quantization_flow`** (optional):
  Specifies the way to quantize torch.fx.GraphModule.
  Supported values:
  - `nncf`: `nncf quantize_pt2e` API (default)
  - `pt2e`: torch ao quantization pipeline.

- **`--validate`** (optional):
  Enable model validation. --dataset argument is requred for the validation. `huggingface` suite does not supported yet.

- **`--dataset`** (optional):
  Path to the imagenet-like calibration dataset.

- **`--device`** (optional)
  Target device for the compiled model. Default is `CPU`.
  Examples: `CPU`, `GPU`


### **Examples**

#### Export a TIMM VGG16 model for the CPU
```bash
python aot_openvino_compiler.py --suite timm --model vgg16 --input_shape [1, 3, 224, 224] --device CPU
```

#### Export a Torchvision ResNet50 model for the GPU
```bash
python aot_openvino_compiler.py --suite torchvision --model resnet50 --input_shape "(1, 3, 256, 256)" --device GPU
```

#### Export a Hugging Face BERT model for the CPU
```bash
python aot_openvino_compiler.py --suite huggingface --model bert-base-uncased --input_shape "(1, 512)" --device CPU
```
#### Export and validate TIMM Resnet50d model for the CPU
```bash
python aot_openvino_compiler.py --suite timm --model vgg16 --input_shape [1, 3, 224, 224] --device CPU --validate --dataset /path/to/dataset
```

#### Export, quantize and validate TIMM Resnet50d model for the CPU
```bash
python aot_openvino_compiler.py --suite timm --model vgg16 --input_shape [1, 3, 224, 224] --device CPU --validate --dataset /path/to/dataset --quantize
```

### **Notes**
1. **Input Shape in Zsh**:
   If you are using Zsh, wrap `--input_shape` in quotes or use a tuple:
   ```bash
   --input_shape '[1, 3, 224, 224]'
   --input_shape "(1, 3, 224, 224)"
   ```

2. **Model Compatibility**:
   Ensure the specified `model_name` exists in the selected `suite`. Use the corresponding library's documentation to verify model availability.

3. **Output File**:
   The exported model will be saved as `<MODEL_NAME>.pte` in the current directory.

4. **Dependencies**:
   - Python 3.8+
   - PyTorch
   - Executorch
   - TIMM (`pip install timm`)
   - Torchvision
   - Transformers (`pip install transformers`)

### **Error Handling**
- **Model Not Found**:
  If the script raises an error such as:
  ```bash
  ValueError: Model <MODEL_NAME> not found
  ```
  Verify that the model name is correct for the chosen suite.

- **Unsupported Input Shape**:
  Ensure `--input_shape` is provided as a valid list or tuple.


## Build OpenVINO Examples
Build the backend libraries and executor runner by executing the script below in `<executorch_root>/backends/openvino/scripts` folder:
```bash
./openvino_build.sh
```
The executable is saved in `<executorch_root>/cmake-out/backends/openvino/`

### Run the Example with Executor Runner

Now, run the example using the executable generated in the above step. The executable requires a model file (`.pte` file generated in the aot step), and optional number of inference executions.

#### Command Syntax:

```
cd ../../cmake-out/backends/openvino

./openvino_executor_runner \
    --model_path=<path_to_model> \
    --num_executions=<iterations>
```
#### Command-Line Arguments

- `--model_path`: (Required) Path to the model serialized in `.pte` format.
- `--num_executions`: (Optional) Number of times to run inference (default: 1).

#### Example Usage

Run inference with a given model for 10 iterations:

```
./openvino_executor_runner \
    --model_path=model.pte \
    --num_executions=10
```

## Running Python Example with Pybinding:

You can use the `export_and_infer_openvino.py` script to run models with the OpenVINO backend through the Python bindings.

### **Usage**

#### **Command Structure**
```bash
python export_and_infer_openvino.py <ARGUMENTS>
```

#### **Arguments**
- **`--suite`** (required if `--model_path` argument is not used):
  Specifies the model suite to use. Needs to be used with `--model` argument.
  Supported values:
  - `timm` (e.g., VGG16, ResNet50)
  - `torchvision` (e.g., resnet18, mobilenet_v2)
  - `huggingface` (e.g., bert-base-uncased). NB: Quantization and validation is not supported yet.

- **`--model`** (required if `--model_path` argument is not used):
  Name of the model to export. Needs to be used with `--suite` argument.
  Examples:
  - For `timm`: `vgg16`, `resnet50`
  - For `torchvision`: `resnet18`, `mobilenet_v2`
  - For `huggingface`: `bert-base-uncased`, `distilbert-base-uncased`

- **`--model_path`** (required if `--suite` and `--model` arguments are not used):
  Path to the saved model file. This argument allows you to load the compiled model from a file, instead of downloading it from the model suites using the `--suite` and `--model` arguments.   
  Example: `<path to model foler>/resnet50_fp32.pte`

- **`--input_shape`**(required for random inputs):
  Input shape for the model. Provide this as a **list** or **tuple**.  
  Examples:
  - `[1, 3, 224, 224]` (Zsh users: wrap in quotes)
  - `(1, 3, 224, 224)`

 - **`--input_tensor_path`**(optional):
   Path to the raw input tensor file. If this argument is not provided, a random input tensor will be generated with the input shape provided with `--input_shape` argument.  
  Example: `<path to the input tensor foler>/input_tensor.pt`

 - **`--output_tensor_path`**(optional):
   Path to the file where the output raw tensor will be saved.  
  Example: `<path to the output tensor foler>/output_tensor.pt`

- **`--device`** (optional)
  Target device for the compiled model. Default is `CPU`.  
  Examples: `CPU`, `GPU`

- **`--num_iter`** (optional)
  Number of iterations to execute inference for evaluation. The default value is `1`.  
  Examples: `100`, `1000`

- **`--warmup_iter`** (optional)
  Number of warmup iterations to execute inference before evaluation. The default value is `0`.  
  Examples: `5`, `10`


### **Examples**

#### Execute Torchvision ResNet50 model for the GPU with Random Inputs
```bash
python export_and_infer_openvino.py --suite torchvision --model resnet50 --input_shape "(1, 3, 256, 256)" --device GPU
```

#### Run a Precompiled Model for the CPU Using an Existing Input Tensor File and Save the Output.
```bash
python export_and_infer_openvino.py --model_path /path/to/model/folder/resnet50_fp32.pte --input_tensor_file /path/to/input/folder/input.pt --output_tensor_file /path/to/output/folder/output.pt --device CPU
```
