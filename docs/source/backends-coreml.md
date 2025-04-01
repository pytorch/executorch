# Core ML Backend

Core ML delegate is the ExecuTorch solution to take advantage of Apple's [CoreML framework](https://developer.apple.com/documentation/coreml) for on-device ML.  With CoreML, a model can run on CPU, GPU, and the Apple Neural Engine (ANE).

## Features

- Dynamic dispatch to the CPU, GPU, and ANE.
- Supports fp32 and fp16 computation.

## Target Requirements

Below are the minimum OS requirements on various hardware for running a CoreML-delegated ExecuTorch model:
- [macOS](https://developer.apple.com/macos) >= 13.0
- [iOS](https://developer.apple.com/ios/) >= 16.0
- [iPadOS](https://developer.apple.com/ipados/) >= 16.0
- [tvOS](https://developer.apple.com/tvos/) >= 16.0

## Development Requirements
To develop you need:

- [macOS](https://developer.apple.com/macos) >= 13.0.
- [Xcode](https://developer.apple.com/documentation/xcode) >= 14.1


Before starting, make sure you install the Xcode Command Line Tools:

```bash
xcode-select --install
```

----

## Using the CoreML Backend

To target the CoreML backend during the export and lowering process, pass an instance of the `CoreMLPartitioner` to `to_edge_transform_and_lower`. The example below demonstrates this process using the MobileNet V2 model from torchvision.

```python
import torch
import torchvision.models as models
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from executorch.backends.apple.coreml.partition import CoreMLPartitioner
from executorch.exir import to_edge_transform_and_lower

mobilenet_v2 = models.mobilenetv2.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()
sample_inputs = (torch.randn(1, 3, 224, 224), )

et_program = to_edge_transform_and_lower(
    torch.export.export(mobilenet_v2, sample_inputs),
    partitioner=[CoreMLPartitioner()],
).to_executorch()

with open("mv2_coreml.pte", "wb") as file:
    et_program.write_to_file(file)
```

### Partitioner API

The CoreML partitioner API allows for configuration of the model delegation to CoreML. Passing an `CoreMLPartitioner` instance with no additional parameters will run as much of the model as possible on the CoreML backend with default settings. This is the most common use-case. For advanced use cases, the partitioner exposes the following options via the [constructor](https://github.com/pytorch/executorch/blob/14ff52ff89a89c074fc6c14d3f01683677783dcd/backends/apple/coreml/partition/coreml_partitioner.py#L60):


 - `skip_ops_for_coreml_delegation`: Allows you to skip ops for delegation by CoreML.  By default, all ops that CoreML supports will be delegated.  See [here](https://github.com/pytorch/executorch/blob/14ff52ff89a89c074fc6c14d3f01683677783dcd/backends/apple/coreml/test/test_coreml_partitioner.py#L42) for an example of skipping an op for delegation.
- `compile_specs`: A list of CompileSpec for the CoreML backend.  These control low-level details of CoreML delegation, such as the compute unit (CPU, GPU, ANE), the iOS deployment target, and the compute precision (FP16, FP32).  These are discussed more below.
- `take_over_mutable_buffer`: A boolean that indicates whether PyTorch mutable buffers in stateful models should be converted to [CoreML MLState](https://developer.apple.com/documentation/coreml/mlstate).  If set to false, mutable buffers in the PyTorch graph are converted to graph inputs and outputs to the CoreML lowered module under the hood.  Generally setting take_over_mutable_buffer to true will result in better performance, but using MLState requires iOS >= 18.0, macOS >= 15.0, and XCode >= 16.0.

#### CoreML CompileSpec

A list of CompileSpec is constructed with [CoreMLBackend.generate_compile_specs](https://github.com/pytorch/executorch/blob/14ff52ff89a89c074fc6c14d3f01683677783dcd/backends/apple/coreml/compiler/coreml_preprocess.py#L210).  Below are the available options:
- `compute_unit`: this controls the compute units (CPU, GPU, ANE) that are used by CoreML.  The default value is coremltools.ComputeUnit.ALL.  The available options from coremltools are:
    - coremltools.ComputeUnit.ALL (uses the CPU, GPU, and ANE)
    - coremltools.ComputeUnit.CPU_ONLY (uses the CPU only)
    - coremltools.ComputeUnit.CPU_AND_GPU (uses both the CPU and GPU, but not the ANE)
    - coremltools.ComputeUnit.CPU_AND_NE (uses both the CPU and ANE, but not the GPU)
- `minimum_deployment_target`: The minimum iOS deployment target (e.g., coremltools.target.iOS18).  The default value is coremltools.target.iOS15.
- `compute_precision`: The compute precision used by CoreML (coremltools.precision.FLOAT16, coremltools.precision.FLOAT32).  The default value is coremltools.precision.FLOAT16.  Note that the compute precision is applied no matter what dtype is specified in the exported PyTorch model.  For example, an FP32 PyTorch model will be converted to FP16 when delegating to the CoreML backend by default.  Also note that the ANE only supports FP16 precision.
- `model_type`: Whether the model should be compiled to the CoreML [mlmodelc format](https://developer.apple.com/documentation/coreml/downloading-and-compiling-a-model-on-the-user-s-device) during .pte creation ([CoreMLBackend.MODEL_TYPE.COMPILED_MODEL](https://github.com/pytorch/executorch/blob/14ff52ff89a89c074fc6c14d3f01683677783dcd/backends/apple/coreml/compiler/coreml_preprocess.py#L71)), or whether it should be compiled to mlmodelc on device ([CoreMLBackend.MODEL_TYPE.MODEL](https://github.com/pytorch/executorch/blob/14ff52ff89a89c074fc6c14d3f01683677783dcd/backends/apple/coreml/compiler/coreml_preprocess.py#L70)).  Using CoreMLBackend.MODEL_TYPE.COMPILED_MODEL and doing compilation ahead of time should improve the first time on-device model load time.

### Testing the Model

After generating the CoreML-delegated .pte, the model can be tested from Python using the ExecuTorch runtime python bindings. This can be used to sanity check the model and evaluate numerical accuracy. See [Testing the Model](using-executorch-export.md#testing-the-model) for more information.

----

### Quantization

To quantize a PyTorch model for the CoreML backend, use the `CoreMLQuantizer`. `Quantizers` are backend specific, and the `CoreMLQuantizer` is configured to quantize models to leverage the available quantization for the CoreML backend.

### 8-bit Quantization using the PT2E Flow

To perform 8-bit quantization with the PT2E flow, perform the following steps:

1) Define [coremltools.optimize.torch.quantization.LinearQuantizerConfig](https://apple.github.io/coremltools/source/coremltools.optimize.torch.quantization.html#coremltools.optimize.torch.quantization.LinearQuantizerConfig) and use to to create an instance of a `CoreMLQuantizer`.
2) Use `torch.export.export_for_training` to export a graph module that will be prepared for quantization.
3) Call `prepare_pt2e` to prepare the model for quantization.
4) For static quantization, run the prepared model with representative samples to calibrate the quantizated tensor activation ranges.
5) Call `convert_pt2e` to quantize the model.
6) Export and lower the model using the standard flow.

The output of `convert_pt2e` is a PyTorch model which can be exported and lowered using the normal flow. As it is a regular PyTorch model, it can also be used to evaluate the accuracy of the quantized model using standard PyTorch techniques.

```python
import torch
import coremltools as ct
import torchvision.models as models
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from executorch.backends.apple.coreml.quantizer import CoreMLQuantizer
from executorch.backends.apple.coreml.partition import CoreMLPartitioner
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.apple.coreml.compiler import CoreMLBackend

mobilenet_v2 = models.mobilenetv2.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()
sample_inputs = (torch.randn(1, 3, 224, 224), )

# Step 1: Define a LinearQuantizerConfig and create an instance of a CoreMLQuantizer
quantization_config = ct.optimize.torch.quantization.LinearQuantizerConfig.from_dict(
    {
        "global_config": {
            "quantization_scheme": ct.optimize.torch.quantization.QuantizationScheme.symmetric,
            "milestones": [0, 0, 10, 10],
            "activation_dtype": torch.quint8,
            "weight_dtype": torch.qint8,
            "weight_per_channel": True,
        }
    }
)
quantizer = CoreMLQuantizer(quantization_config)

# Step 2: Export the model for training
training_gm = torch.export.export_for_training(mobilenet_v2, sample_inputs).module()

# Step 3: Prepare the model for quantization
prepared_model = prepare_pt2e(training_gm, quantizer)

# Step 4: Calibrate the model on representative data
# Replace with your own calibration data
for calibration_sample in [torch.randn(1, 3, 224, 224)]:
	prepared_model(calibration_sample)

# Step 5: Convert the calibrated model to a quantized model
quantized_model = convert_pt2e(prepared_model)

# Step 6: Export the quantized model to CoreML
et_program = to_edge_transform_and_lower(
    torch.export.export(quantized_model, sample_inputs),
    partitioner=[
        CoreMLPartitioner(
             # iOS17 is required for the quantized ops in this example
            compile_specs=CoreMLBackend.generate_compile_specs(
                minimum_deployment_target=ct.target.iOS17
            )
        )
    ],
).to_executorch()
```

See [PyTorch 2 Export Post Training Quantization](https://pytorch.org/tutorials/prototype/pt2e_quant_ptq.html) for more information.

----

## Runtime integration

To run the model on-device, use the standard ExecuTorch runtime APIs. See [Running on Device](getting-started.md#running-on-device) for more information, including building the iOS frameworks.

When building from source, pass `-DEXECUTORCH_BUILD_COREML=ON` when configuring the CMake build to compile the CoreML backend.

To link against the `coremldelegate` target. Due to the use of static registration, it may be necessary to link with whole-archive. This can typically be done by passing `"$<LINK_LIBRARY:WHOLE_ARCHIVE,coremldelegate>"` to `target_link_libraries`.

```
# CMakeLists.txt
add_subdirectory("executorch")
...
target_link_libraries(
    my_target
    PRIVATE executorch
    executorch_module_static
    executorch_tensor
    optimized_native_cpu_ops_lib
    coremldelegate)
```

No additional steps are necessary to use the backend beyond linking the target. A CoreML-delegated .pte file will automatically run on the registered backend.

---

## Advanced

### Extracting the mlpackage

[CoreML *.mlpackage files](https://apple.github.io/coremltools/docs-guides/source/convert-to-ml-program.html#save-ml-programs-as-model-packages) can be extracted from a CoreML-delegated *.pte file.  This can help with debugging and profiling for users who are more familiar with *.mlpackage files:
```bash
python examples/apple/coreml/scripts/extract_coreml_models.py -m /path/to/model.pte
```

Note that if the ExecuTorch model has graph breaks, there may be multiple extracted *.mlpackage files.
