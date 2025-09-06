# CoreML Backend

CoreML delegate is the ExecuTorch solution to take advantage of Apple's [CoreML framework](https://developer.apple.com/documentation/coreml) for on-device ML.  With CoreML, a model can run on CPU, GPU, and the Apple Neural Engine (ANE).

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

- [macOS](https://developer.apple.com/macos) >= 13.0
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

The CoreML partitioner API allows for configuration of the model delegation to CoreML. Passing a `CoreMLPartitioner` instance with no additional parameters will run as much of the model as possible on the CoreML backend with default settings. This is the most common use case. For advanced use cases, the partitioner exposes the following options via the [constructor](https://github.com/pytorch/executorch/blob/14ff52ff89a89c074fc6c14d3f01683677783dcd/backends/apple/coreml/partition/coreml_partitioner.py#L60):


 - `skip_ops_for_coreml_delegation`: Allows you to skip ops for delegation by CoreML.  By default, all ops that CoreML supports will be delegated.  See [here](https://github.com/pytorch/executorch/blob/14ff52ff89a89c074fc6c14d3f01683677783dcd/backends/apple/coreml/test/test_coreml_partitioner.py#L42) for an example of skipping an op for delegation.
- `compile_specs`: A list of `CompileSpec`s for the CoreML backend.  These control low-level details of CoreML delegation, such as the compute unit (CPU, GPU, ANE), the iOS deployment target, and the compute precision (FP16, FP32).  These are discussed more below.
- `take_over_mutable_buffer`: A boolean that indicates whether PyTorch mutable buffers in stateful models should be converted to [CoreML `MLState`](https://developer.apple.com/documentation/coreml/mlstate).  If set to `False`, mutable buffers in the PyTorch graph are converted to graph inputs and outputs to the CoreML lowered module under the hood.  Generally, setting `take_over_mutable_buffer` to true will result in better performance, but using `MLState` requires iOS >= 18.0, macOS >= 15.0, and Xcode >= 16.0.
- `take_over_constant_data`: A boolean that indicates whether PyTorch constant data like model weights should be consumed by the CoreML delegate.  If set to False, constant data is passed to the CoreML delegate as inputs.  By deafault, take_over_constant_data=True.
- `lower_full_graph`: A boolean that indicates whether the entire graph must be lowered to CoreML.  If set to True and CoreML does not support an op, an error is raised during lowering.  If set to False and CoreML does not support an op, the op is executed on the CPU by ExecuTorch.  Although setting `lower_full_graph`=False can allow a model to lower where it would otherwise fail, it can introduce performance overhead in the model when there are unsupported ops.  You will see warnings about unsupported ops during lowering if there are any.  By default, `lower_full_graph`=False.


#### CoreML CompileSpec

A list of `CompileSpec`s is constructed with [`CoreMLBackend.generate_compile_specs`](https://github.com/pytorch/executorch/blob/14ff52ff89a89c074fc6c14d3f01683677783dcd/backends/apple/coreml/compiler/coreml_preprocess.py#L210).  Below are the available options:
- `compute_unit`: this controls the compute units (CPU, GPU, ANE) that are used by CoreML.  The default value is `coremltools.ComputeUnit.ALL`.  The available options from coremltools are:
    - `coremltools.ComputeUnit.ALL` (uses the CPU, GPU, and ANE)
    - `coremltools.ComputeUnit.CPU_ONLY` (uses the CPU only)
    - `coremltools.ComputeUnit.CPU_AND_GPU` (uses both the CPU and GPU, but not the ANE)
    - `coremltools.ComputeUnit.CPU_AND_NE` (uses both the CPU and ANE, but not the GPU)
- `minimum_deployment_target`: The minimum iOS deployment target (e.g., `coremltools.target.iOS18`).  By default, the smallest deployment target needed to deploy the model is selected.  During export, you will see a warning about the "CoreML specification version" that was used for the model, which maps onto a deployment target as discussed [here](https://apple.github.io/coremltools/mlmodel/Format/Model.html#model).  If you need to control the deployment target, please specify it explicitly.
- `compute_precision`: The compute precision used by CoreML (`coremltools.precision.FLOAT16` or `coremltools.precision.FLOAT32`).  The default value is `coremltools.precision.FLOAT16`.  Note that the compute precision is applied no matter what dtype is specified in the exported PyTorch model.  For example, an FP32 PyTorch model will be converted to FP16 when delegating to the CoreML backend by default.  Also note that the ANE only supports FP16 precision.
- `model_type`: Whether the model should be compiled to the CoreML [mlmodelc format](https://developer.apple.com/documentation/coreml/downloading-and-compiling-a-model-on-the-user-s-device) during .pte creation ([`CoreMLBackend.MODEL_TYPE.COMPILED_MODEL`](https://github.com/pytorch/executorch/blob/14ff52ff89a89c074fc6c14d3f01683677783dcd/backends/apple/coreml/compiler/coreml_preprocess.py#L71)), or whether it should be compiled to mlmodelc on device ([`CoreMLBackend.MODEL_TYPE.MODEL`](https://github.com/pytorch/executorch/blob/14ff52ff89a89c074fc6c14d3f01683677783dcd/backends/apple/coreml/compiler/coreml_preprocess.py#L70)).  Using `CoreMLBackend.MODEL_TYPE.COMPILED_MODEL` and doing compilation ahead of time should improve the first time on-device model load time.

### Dynamic and Enumerated Shapes in CoreML Export

When exporting an `ExportedProgram` to CoreML, **dynamic shapes** are mapped to [`RangeDim`](https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html#set-the-range-for-each-dimension).
This enables CoreML `.pte` files to accept inputs with varying dimensions at runtime.

⚠️ **Note:** The Apple Neural Engine (ANE) does not support true dynamic shapes.    If a model relies on `RangeDim`, CoreML will fall back to scheduling the model on the CPU or GPU instead of the ANE.

---

#### Enumerated Shapes

To enable limited flexibility on the ANE—and often achieve better performance overall—you can export models using **[enumerated shapes](https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html#select-from-predetermined-shapes)**.

- Enumerated shapes are *not fully dynamic*.
- Instead, they define a **finite set of valid input shapes** that CoreML can select from at runtime.
- This approach allows some adaptability while still preserving ANE compatibility.

---

#### Specifying Enumerated Shapes

Unlike `RangeDim`, **enumerated shapes are not part of the `ExportedProgram` itself.**
They must be provided through a compile spec.

For reference on how to do this, see:
- The annotated code snippet below, and
- The [end-to-end test in ExecuTorch](https://github.com/pytorch/executorch/blob/main/backends/apple/coreml/test/test_enumerated_shapes.py), which demonstrates how to specify enumerated shapes during export.


```python
class Model(torch.nn.Module):
        def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(10, 5)
                self.linear2 = torch.nn.Linear(11, 5)

        def forward(self, x, y):
            return self.linear1(x).sum() + self.linear2(y)

model = Model()
example_inputs = (
    torch.randn((4, 6, 10)),
    torch.randn((5, 11)),
)

# Specify the enumerated shapes.  Below we specify that:
#
# * x can take shape [1, 5, 10] and y can take shape [3, 11], or
# * x can take shape [4, 6, 10] and y can take shape [5, 11]
#
# Any other input shapes will result in a runtime error.
#
# Note that we must export x and y with dynamic shapes in the ExportedProgram
# because some of their dimensions are dynamic
enumerated_shapes = {"x": [[1, 5, 10], [4, 6, 10]], "y": [[3, 11], [5, 11]]}
dynamic_shapes = [
    {
        0: torch.export.Dim.AUTO(min=1, max=4),
        1: torch.export.Dim.AUTO(min=5, max=6),
    },
    {0: torch.export.Dim.AUTO(min=3, max=5)},
]
ep = torch.export.export(
    model.eval(), example_inputs, dynamic_shapes=dynamic_shapes
)

# If enumerated shapes are specified for multiple inputs, we must export
# for iOS18+
compile_specs = CoreMLBackend.generate_compile_specs(
    minimum_deployment_target=ct.target.iOS18
)
compile_specs.append(
    CoreMLBackend.generate_enumerated_shapes_compile_spec(
        ep,
        enumerated_shapes,
    )
)

# When using an enumerated shape compile spec, you must specify lower_full_graph=True
# in the CoreMLPartitioner.  We do not support using enumerated shapes
# for partially exported models
partitioner = CoreMLPartitioner(
    compile_specs=compile_specs, lower_full_graph=True
)
delegated_program = executorch.exir.to_edge_transform_and_lower(
    ep,
    partitioner=[partitioner],
)
et_prog = delegated_program.to_executorch()
```

### Backward compatibility

CoreML supports backward compatibility via the `minimum_deployment_target` option.  A model exported with a specific deployment target is guaranteed to work on all deployment targets >= the specified deployment target.  For example, a model exported with `coremltools.target.iOS17` will work on iOS 17 or higher.

### Testing the Model

After generating the CoreML-delegated .pte, the model can be tested from Python using the ExecuTorch runtime Python bindings. This can be used to quickly check the model and evaluate numerical accuracy. See [Testing the Model](using-executorch-export.md#testing-the-model) for more information.

----

### Quantization

To quantize a PyTorch model for the CoreML backend, use the `CoreMLQuantizer`.

### 8-bit Quantization using the PT2E Flow

Quantization with the CoreML backend requires exporting the model for iOS 17 or later.
To perform 8-bit quantization with the PT2E flow, follow these steps:

1) Create a [`coremltools.optimize.torch.quantization.LinearQuantizerConfig`](https://apple.github.io/coremltools/source/coremltools.optimize.torch.quantization.html#coremltools.optimize.torch.quantization.LinearQuantizerConfig) and use to to create an instance of a `CoreMLQuantizer`.
2) Use `torch.export.export_for_training` to export a graph module that will be prepared for quantization.
3) Call `prepare_pt2e` to prepare the model for quantization.
4) Run the prepared model with representative samples to calibrate the quantizated tensor activation ranges.
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
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.apple.coreml.compiler import CoreMLBackend

mobilenet_v2 = models.mobilenetv2.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()
sample_inputs = (torch.randn(1, 3, 224, 224), )

# Step 1: Define a LinearQuantizerConfig and create an instance of a CoreMLQuantizer
# Note that "linear" here does not mean only linear layers are quantized, but that linear (aka affine) quantization
# is being performed
static_8bit_config = ct.optimize.torch.quantization.LinearQuantizerConfig(
    global_config=ct.optimize.torch.quantization.ModuleLinearQuantizerConfig(
        quantization_scheme="symmetric",
        activation_dtype=torch.quint8,
        weight_dtype=torch.qint8,
        weight_per_channel=True,
    )
)
quantizer = CoreMLQuantizer(static_8bit_config)

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

The above does static quantization (activations and weights are quantized).

You can see a full description of available quantization configs in the [coremltools documentation](https://apple.github.io/coremltools/source/coremltools.optimize.torch.quantization.html#coremltools.optimize.torch.quantization.LinearQuantizerConfig).  For example, the config below will perform weight-only quantization:

```
weight_only_8bit_config = ct.optimize.torch.quantization.LinearQuantizerConfig(
    global_config=ct.optimize.torch.quantization.ModuleLinearQuantizerConfig(
        quantization_scheme="symmetric",
        activation_dtype=torch.float32,
        weight_dtype=torch.qint8,
        weight_per_channel=True,
    )
)
quantizer = CoreMLQuantizer(weight_only_8bit_config)
```

Quantizing activations requires calibrating the model on representative data.  Also note that PT2E currently requires passing at least 1 calibration sample before calling `convert_pt2e`, even for data-free weight-only quantization.

See [PyTorch 2 Export Post Training Quantization](https://docs.pytorch.org/ao/main/tutorials_source/pt2e_quant_ptq.html) for more information.

### LLM quantization with quantize_

The CoreML backend also supports quantizing models with the [torchao](https://github.com/pytorch/ao) quantize_ API.  This is most commonly used for LLMs, requiring more advanced quantization.  Since quantize_ is not backend aware, it is important to use a config that is compatible with CoreML:

* Quantize embedding/linear layers with IntxWeightOnlyConfig (with weight_dtype torch.int4 or torch.int8, using PerGroup or PerAxis granularity).  Using 4-bit or PerGroup quantization requires exporting with minimum_deployment_target >= ct.target.iOS18.  Using 8-bit quantization with per-axis granularity is supported on ct.target.IOS16+.  See [CoreML `CompileSpec`](#coreml-compilespec) for more information on setting the deployment target.
* Quantize embedding/linear layers with CodebookWeightOnlyConfig (with dtype torch.uint1 through torch.uint8, using various block sizes).  Quantizing with CodebookWeightOnlyConfig requires exporting with minimum_deployment_target >= ct.target.iOS18, see [CoreML `CompileSpec`](#coreml-compilespec) for more information on setting the deployment target.

Below is an example that quantizes embeddings to 8-bits per-axis and linear layers to 4-bits using group_size=32 with affine quantization:

```python
from torchao.quantization.granularity import PerGroup, PerAxis
from torchao.quantization.quant_api import (
    IntxWeightOnlyConfig,
    quantize_,
)

# Quantize embeddings with 8-bits, per channel
embedding_config = IntxWeightOnlyConfig(
    weight_dtype=torch.int8,
    granularity=PerAxis(0),
)
quantize_(
    eager_model,
    embedding_config,
    lambda m, fqn: isinstance(m, torch.nn.Embedding),
)

# Quantize linear layers with 4-bits, per-group
linear_config = IntxWeightOnlyConfig(
    weight_dtype=torch.int4,
    granularity=PerGroup(32),
)
quantize_(
    eager_model,
    linear_config,
)
```

Below is another example that uses codebook quantization to quantize both embeddings and linear layers to 3-bits.
In the coremltools documentation, this is called [palettization](https://apple.github.io/coremltools/docs-guides/source/opt-palettization-overview.html):

```
from torchao.quantization.quant_api import (
    quantize_,
)
from torchao.prototype.quantization.codebook_coreml import CodebookWeightOnlyConfig

quant_config = CodebookWeightOnlyConfig(
    dtype=torch.uint3,
    # There is one LUT per 16 rows
    block_size=[16, -1],
)

quantize_(
    eager_model,
    quant_config,
    lambda m, fqn: isinstance(m, torch.nn.Embedding) or isinstance(m, torch.nn.Linear),
)
```

Both of the above examples will export and lower to CoreML with the to_edge_transform_and_lower API.

----

## Runtime integration

To run the model on device, use the standard ExecuTorch runtime APIs. See [Running on Device](getting-started.md#running-on-device) for more information, including building the iOS frameworks.

When building from source, pass `-DEXECUTORCH_BUILD_COREML=ON` when configuring the CMake build to compile the CoreML backend.

Due to the use of static initializers for registration, it may be necessary to use whole-archive to link against the `coremldelegate` target. This can typically be done by passing `"$<LINK_LIBRARY:WHOLE_ARCHIVE,coremldelegate>"` to `target_link_libraries`.

```
# CMakeLists.txt
add_subdirectory("executorch")
...
target_link_libraries(
    my_target
    PRIVATE executorch
    extension_module_static
    extension_tensor
    optimized_native_cpu_ops_lib
    $<LINK_LIBRARY:WHOLE_ARHIVE,coremldelegate>)
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

## Common issues and what to do

### During lowering
1. "ValueError: In op, of type [X], named [Y], the named input [Z] must have the same data type as the named input x. However, [Z] has dtype fp32 whereas x has dtype fp16."

This happens because the model is in FP16, but CoreML interprets some of the arguments as FP32, which leads to a type mismatch.  The solution is to keep the PyTorch model in FP32.  Note that the model will be still be converted to FP16 during lowering to CoreML unless specified otherwise in the compute_precision [CoreML `CompileSpec`](#coreml-compilespec).  Also see the [related issue in coremltools](https://github.com/apple/coremltools/issues/2480).

2. coremltools/converters/mil/backend/mil/load.py", line 499, in export
    raise RuntimeError("BlobWriter not loaded")

If you're using Python 3.13, try reducing your python version to Python 3.12.  coremltools does not support Python 3.13 per [coremltools issue #2487](https://github.com/apple/coremltools/issues/2487).

### At runtime
1. [ETCoreMLModelCompiler.mm:55] [CoreML]  Failed to compile model, error = Error Domain=com.apple.mlassetio Code=1 "Failed to parse the model specification. Error: Unable to parse ML Program: at unknown location: Unknown opset 'CoreML7'." UserInfo={NSLocalizedDescription=Failed to par$

This means the model requires the the CoreML opset 'CoreML7', which requires running the model on iOS >= 17 or macOS >= 14.
