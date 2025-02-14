# **Model Export Script for Executorch**

This script allows users to export deep learning models from various model suites (TIMM, Torchvision, Hugging Face) to a openvino backend using **Executorch**. Users can dynamically specify the model, input shape, and target device.


## **Usage**

### **Command Structure**
```bash
python aot_openvino_compiler.py --suite <MODEL_SUITE> --model <MODEL_NAME> --input_shape <INPUT_SHAPE> --device <DEVICE>
```

### **Arguments**
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


## **Examples**

### Export a TIMM VGG16 model for the CPU
```bash
python aot_openvino_compiler.py --suite timm --model vgg16 --input_shape [1, 3, 224, 224] --device CPU
```

### Export a Torchvision ResNet50 model for the GPU
```bash
python aot_openvino_compiler.py --suite torchvision --model resnet50 --input_shape "(1, 3, 256, 256)" --device GPU
```

### Export a Hugging Face BERT model for the CPU
```bash
python aot_openvino_compiler.py --suite huggingface --model bert-base-uncased --input_shape "(1, 512)" --device CPU
```
### Export and validate TIMM Resnet50d model for the CPU
```bash
python aot_openvino_compiler.py --suite timm --model vgg16 --input_shape [1, 3, 224, 224] --device CPU --validate --dataset /path/to/dataset
```

### Export, quantize and validate TIMM Resnet50d model for the CPU
```bash
python aot_openvino_compiler.py --suite timm --model vgg16 --input_shape [1, 3, 224, 224] --device CPU --validate --dataset /path/to/dataset --quantize
```

## **Notes**
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

## **Error Handling**
- **Model Not Found**:
  If the script raises an error such as:
  ```bash
  ValueError: Model <MODEL_NAME> not found
  ```
  Verify that the model name is correct for the chosen suite.

- **Unsupported Input Shape**:
  Ensure `--input_shape` is provided as a valid list or tuple.


