# OpenVINO Backend Examples

This guide provides detailed instructions on how to export models for Executorch and execute them on the OpenVINO backend. The examples demonstrate how to export a model, load a model, prepare input tensors, execute inference, and save the output results.

## Directory Structure

Below is the layout of the `examples/openvino` directory, which includes the necessary files for the example applications:

```
examples/openvino
├── README.md                           # Documentation for examples (this file)
├── aot_optimize_and_infer.py           # Example script to export and execute models
└── llama
    ├── README.md                       # Documentation for Llama example
    └── llama3_2_ov_4wo.yaml            # Configuration file for exporting Llama3.2 with OpenVINO backend
```

# Build Instructions for Examples

## Environment Setup
Follow the [instructions](../../backends/openvino/README.md) of **Prerequisites** and **Setup** in `backends/openvino/README.md` to set up the OpenVINO backend.

## AOT step:

The python script called `aot_optimize_and_infer.py` allows users to export deep learning models from various model suites (TIMM, Torchvision, Hugging Face) to a openvino backend using **Executorch**. Users can dynamically specify the model, input shape, and target device.

### **Usage**


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

- **`--export`** (optional):
  Save the exported model as a `.pte` file.

- **`--model_file_name`** (optional):
  Specify a custom file name to save the exported model.

- **`--batch_size`** :
  Batch size for the validation. Default batch_size == 1.
  The dataset length must be evenly divisible by the batch size.

- **`--quantize`** (optional):
  Enable model quantization. --dataset argument is requred for the quantization. `huggingface` suite  does not supported yet.

- **`--validate`** (optional):
  Enable model validation. --dataset argument is requred for the validation. `huggingface` suite does not supported yet.

- **`--dataset`** (optional):
  Path to the imagenet-like calibration dataset.

- **`--infer`** (optional):
  Execute inference with the compiled model and report average inference timing.

- **`--num_iter`** (optional):
  Number of iterations to execute inference. Default value for the number of iterations is `1`.

- **`--warmup_iter`** (optional):
  Number of warmup iterations to execute inference before timing begins. Default value for the warmup iterations is `0`.

- **`--input_tensor_path`** (optional):
  Path to the raw tensor file to be used as input for inference. If this argument is not provided, a random input tensor will be generated.

- **`--output_tensor_path`** (optional):
  Path to the raw tensor file which the output of the inference to be saved.

- **`--device`** (optional)
  Target device for the compiled model. Default is `CPU`.
  Examples: `CPU`, `GPU`


#### **Examples**

##### Export a TIMM VGG16 model for the CPU
```bash
python aot_optimize_and_infer.py --export --suite timm --model vgg16 --input_shape "[1, 3, 224, 224]" --device CPU
```

##### Export a Torchvision ResNet50 model for the GPU
```bash
python aot_optimize_and_infer.py --export --suite torchvision --model resnet50 --input_shape "(1, 3, 256, 256)" --device GPU
```

##### Export a Hugging Face BERT model for the CPU
```bash
python aot_optimize_and_infer.py --export --suite huggingface --model bert-base-uncased --input_shape "(1, 512)" --device CPU
```
##### Export and validate TIMM Resnet50d model for the CPU
```bash
python aot_optimize_and_infer.py --export --suite timm --model vgg16 --input_shape [1, 3, 224, 224] --device CPU --validate --dataset /path/to/dataset
```

##### Export, quantize and validate TIMM Resnet50d model for the CPU
```bash
python aot_optimize_and_infer.py --export --suite timm --model vgg16 --input_shape [1, 3, 224, 224] --device CPU --validate --dataset /path/to/dataset --quantize
```

##### Execute Inference with Torchvision Inception V3 model for the CPU
```bash
python aot_optimize_and_infer.py --suite torchvision --model inception_v3 --infer --warmup_iter 10 --num_iter 100 --input_shape "(1, 3, 256, 256)" --device CPU
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
The executable is saved in `<executorch_root>/cmake-out/`

### Run the Example with Executor Runner

Now, run the example using the executable generated in the above step. The executable requires a model file (`.pte` file generated in the aot step), and optional number of inference executions.

#### Command Syntax:

```
cd ../../cmake-out

./executor_runner \
    --model_path=<path_to_model> \
    --num_executions=<iterations>
```
#### Command-Line Arguments

- `--model_path`: (Required) Path to the model serialized in `.pte` format.
- `--num_executions`: (Optional) Number of times to run inference (default: 1).

#### Example Usage

Run inference with a given model for 10 iterations:

```
./executor_runner \
    --model_path=model.pte \
    --num_executions=10
```
