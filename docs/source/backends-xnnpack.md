# XNNPACK Backend

The XNNPACK delegate is the ExecuTorch solution for CPU execution on mobile CPUs. [XNNPACK](https://github.com/google/XNNPACK/tree/master) is a library that provides optimized kernels for machine learning operators on Arm and x86 CPUs.

## Features

- Wide operator support on Arm and x86 CPUs, available on any modern mobile phone.
- Support for a wide variety of quantization schemes and quantized operators.
- Supports fp32 and fp16 activations.
- Supports 8-bit quantization.

## Target Requirements

- ARM64 on Android, iOS, macOS, Linux, and Windows.
- ARMv7 (with NEON) on Android.
- ARMv6 (with VFPv2) on Linux.
- x86 and x86-64 (up to AVX512) on Windows, Linux, Android.

## Development Requirements

The XNNPACK delegate does not introduce any development system requirements beyond those required by
the core ExecuTorch runtime.

----

## Using the XNNPACK Backend

To target the XNNPACK backend during the export and lowering process, pass an instance of the `XnnpackPartitioner` to `to_edge_transform_and_lower`. The example below demonstrates this process using the MobileNet V2 model from torchvision.

```python
import torch
import torchvision.models as models
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower

mobilenet_v2 = models.mobilenetv2.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()
sample_inputs = (torch.randn(1, 3, 224, 224), )

et_program = to_edge_transform_and_lower(
    torch.export.export(mobilenet_v2, sample_inputs),
    partitioner=[XnnpackPartitioner()],
).to_executorch()

with open("mv2_xnnpack.pte", "wb") as file:
    et_program.write_to_file(file)
```

### Partitioner API

The XNNPACK partitioner API allows for configuration of the model delegation to XNNPACK. Passing an `XnnpackPartitioner` instance with no additional parameters will run as much of the model as possible on the XNNPACK backend. This is the most common use-case. For advanced use cases, the partitioner exposes the following options via the [constructor](https://github.com/pytorch/executorch/blob/release/0.6/backends/xnnpack/partition/xnnpack_partitioner.py#L31):

 - `configs`: Control which operators are delegated to XNNPACK. By default, all available operators all delegated. See [../config/\_\_init\_\_.py](https://github.com/pytorch/executorch/blob/release/0.6/backends/xnnpack/partition/config/__init__.py#L66) for an exhaustive list of available operator configs.
 - `config_precisions`: Filter operators by data type. By default, delegate all precisions. One or more of `ConfigPrecisionType.FP32`, `ConfigPrecisionType.STATIC_QUANT`, or `ConfigPrecisionType.DYNAMIC_QUANT`. See [ConfigPrecisionType](https://github.com/pytorch/executorch/blob/release/0.6/backends/xnnpack/partition/config/xnnpack_config.py#L24).
 - `per_op_mode`: If true, emit individual delegate calls for every operator. This is an advanced option intended to reduce memory overhead in some contexts at the cost of a small amount of runtime overhead. Defaults to false.
 - `verbose`: If true, print additional information during lowering.

### Testing the Model

After generating the XNNPACK-delegated .pte, the model can be tested from Python using the ExecuTorch runtime python bindings. This can be used to sanity check the model and evaluate numerical accuracy. See [Testing the Model](using-executorch-export.md#testing-the-model) for more information.

----

## Quantization

The XNNPACK delegate can also be used as a backend to execute symmetrically quantized models. To quantize a PyTorch model for the XNNPACK backend, use the `XNNPACKQuantizer`. `Quantizers` are backend specific, which means the `XNNPACKQuantizer` is configured to quantize models to leverage the quantized operators offered by the XNNPACK Library.

### Supported Quantization Schemes
The XNNPACK delegate supports the following quantization schemes:

- 8-bit symmetric weights with 8-bit asymmetric activations (via the PT2E quantization flow).
  - Supports both static and dynamic activations.
  - Supports per-channel and per-tensor schemes.
  - Supports linear, convolution, add, mul, cat, and adaptive avg pool 2d operators.

Weight-only quantization is not currently supported on XNNPACK.

### 8-bit Quantization using the PT2E Flow

To perform 8-bit quantization with the PT2E flow, perform the following steps prior to exporting the model:

1) Create an instance of the `XnnpackQuantizer` class. Set quantization parameters.
2) Use `torch.export.export` to prepare for quantization.
3) Call `prepare_pt2e` to prepare the model for quantization.
4) For static quantization, run the prepared model with representative samples to calibrate the quantized tensor activation ranges.
5) Call `convert_pt2e` to quantize the model.
6) Export and lower the model using the standard flow.

The output of `convert_pt2e` is a PyTorch model which can be exported and lowered using the normal flow. As it is a regular PyTorch model, it can also be used to evaluate the accuracy of the quantized model using standard PyTorch techniques.

```python
import torch
import torchvision.models as models
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import XNNPACKQuantizer, get_symmetric_quantization_config
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

model = models.mobilenetv2.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()
sample_inputs = (torch.randn(1, 3, 224, 224), )

qparams = get_symmetric_quantization_config(is_per_channel=True) # (1)
quantizer = XNNPACKQuantizer()
quantizer.set_global(qparams)

training_ep = torch.export.export(model, sample_inputs).module() # (2)
prepared_model = prepare_pt2e(training_ep, quantizer) # (3)

for cal_sample in [torch.randn(1, 3, 224, 224)]: # Replace with representative model inputs
	prepared_model(cal_sample) # (4) Calibrate

quantized_model = convert_pt2e(prepared_model) # (5)

et_program = to_edge_transform_and_lower( # (6)
    torch.export.export(quantized_model, sample_inputs),
    partitioner=[XnnpackPartitioner()],
).to_executorch()
```

See [PyTorch 2 Export Post Training Quantization](https://docs.pytorch.org/ao/main/tutorials_source/pt2e_quant_ptq.html) for more information.

### LLM quantization with quantize_

The XNNPACK backend also supports quantizing models with the [torchao](https://github.com/pytorch/ao) quantize_ API.  This is most commonly used for LLMs, requiring more advanced quantization.  Since quantize_ is not backend aware, it is important to use a config that is compatible with CPU/XNNPACK:

* Quantize embeedings with IntxWeightOnlyConfig (with weight_dtype torch.int2, torch.int4, or torch.int8, using PerGroup or PerAxis granularity)
* Quantize linear layers with Int8DynamicActivationIntxWeightConfig (with weight_dtype=torch.int4, using PerGroup or PerAxis granularity)

Below is a simple example, but a more detailed tutorial including accuracy evaluation on popular LLM benchmarks can be found in the [torchao documentation](https://docs.pytorch.org/ao/main/serving.html#mobile-deployment-with-executorch).

```python
from torchao.quantization.granularity import PerGroup, PerAxis
from torchao.quantization.quant_api import (
    IntxWeightOnlyConfig,
    Int8DynamicActivationIntxWeightConfig,
    quantize_,
)

# Quantize embeddings with 8-bits, per channel
embedding_config = IntxWeightOnlyConfig(
    weight_dtype=torch.int8,
    granularity=PerAxis(0),
)
qunatize_(
    eager_model,
    lambda m, fqn: isinstance(m, torch.nn.Embedding),
)


# Quatize linear layers with 8-bit dynamic activations and 4-bit weights
linear_config = Int8DynamicActivationIntxWeightConfig(
    weight_dtype=torch.int4,
    weight_granularity=PerGroup(32),
)
quantize_(eager_model, linear_config)
```

----

## Runtime Integration

To run the model on-device, use the standard ExecuTorch runtime APIs. See [Running on Device](getting-started.md#running-on-device) for more information.

The XNNPACK delegate is included by default in the published Android, iOS, and pip packages. When building from source, pass `-DEXECUTORCH_BUILD_XNNPACK=ON` when configuring the CMake build to compile the XNNPACK backend.

To link against the backend, add the `xnnpack_backend` CMake target as a build dependency, or link directly against `libxnnpack_backend`. Due to the use of static registration, it may be necessary to link with whole-archive. This can typically be done by passing `"$<LINK_LIBRARY:WHOLE_ARCHIVE,xnnpack_backend>"` to `target_link_libraries`.

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
    xnnpack_backend)
```

No additional steps are necessary to use the backend beyond linking the target. Any XNNPACK-delegated .pte file will automatically run on the registered backend.
