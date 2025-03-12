# XNNPACK Backend

The XNNPACK delegate is the ExecuTorch solution for CPU execution on mobile CPUs. XNNPACK is a library that provides optimized kernels for machine learning operators on Arm and x86 CPUs.

## Features

- Wide operator support on Arm and x86 CPUs, available on any modern mobile phone.
- Support for a wide variety of quantization schemes and quantized operators.

## Target Requirements

- ARM64 on Android, iOS, macOS, Linux, and Windows.
- ARMv7 (with NEON) on Android.
- ARMv6 (with VFPv2) on Linux.
- x86 and x86-64 (up to AVX512) on Windows, Linux, macOS, Android, and iOS simulator.

## Development Requirements

The XNNPACK delegate does not introduce any development system requirements beyond those required by the core ExecuTorch runtime.

## Lowering a Model to XNNPACK

To target the XNNPACK backend during the export and lowering process, pass an instance of the `XnnpackPartitioner` to `to_edge_transform_and_lower`. The example below demonstrates this process using the MobileNet V2 model from torchvision.

```python
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

The XNNPACK partitioner API allows for configuration of the model delegation to XNNPACK. Passing an `XnnpackPartitioner` instance with no additional parameters will run as much of the model as possible on the XNNPACK backend. This is the most common use-case. For advanced use cases, the partitioner exposes the following options via the [constructor](https://github.com/pytorch/executorch/blob/14ff52ff89a89c074fc6c14d3f01683677783dcd/backends/xnnpack/partition/xnnpack_partitioner.py#L31):

 - `configs`: Control which operators are delegated to XNNPACK. By default, all available operators all delegated. See [../config/\_\_init\_\_.py](https://github.com/pytorch/executorch/blob/14ff52ff89a89c074fc6c14d3f01683677783dcd/backends/xnnpack/partition/config/__init__.py#L66) for an exhaustive list of available operator configs.
 - `config_precisions`: Filter operators by data type. By default, delegate all precisions. One or more of `ConfigPrecisionType.FP32`, `ConfigPrecisionType.STATIC_QUANT`, or `ConfigPrecisionType.DYNAMIC_QUANT`. See [ConfigPrecisionType](https://github.com/pytorch/executorch/blob/14ff52ff89a89c074fc6c14d3f01683677783dcd/backends/xnnpack/partition/config/xnnpack_config.py#L24).
 - `per_op_mode`: If true, emit individual delegate calls for every operator. This is an advanced option intended to reduce memory overhead in some contexts at the cost of a small amount of runtime overhead. Defaults to false.
 - `verbose`: If true, print additional information during lowering.

### Quantization

The XNNPACK delegate can also be used as a backend to execute symmetrically quantized models. To quantize a PyTorch model for the XNNPACK backend, use the `XNNPACKQuantizer`. `Quantizers` are backend specific, which means the `XNNPACKQuantizer` is configured to quantize models to leverage the quantized operators offered by the XNNPACK Library.

### Configuring the XNNPACKQuantizer

```python
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
  XNNPACKQuantizer,
  get_symmetric_quantization_config,
)
quantizer = XNNPACKQuantizer()
quantizer.set_global(get_symmetric_quantization_config())
```
Here, the `XNNPACKQuantizer` is configured for symmetric quantization, indicating that the quantized zero point is set to zero with `qmin = -127` and `qmax = 127`. `get_symmetric_quantization_config()` can be configured with the following arguments:
* `is_per_channel`
    * Weights are quantized across channels
* `is_qat`
    * Quantize aware training
* `is_dynamic`
    * Dynamic quantization

```python
quantizer.set_global(quantization_config)
    .set_object_type(torch.nn.Conv2d, quantization_config) # can configure by module type
    .set_object_type(torch.nn.functional.linear, quantization_config) # or torch functional op typea
    .set_module_name("foo.bar", quantization_config)  # or by module fully qualified name
```

#### Quantizing a model with the XNNPACKQuantizer
After configuring the quantizer, the model can be quantized by via the `prepare_pt2e` and `convert_pt2e` APIs.
```python
from torch.ao.quantization.quantize_pt2e import (
  prepare_pt2e,
  convert_pt2e,
)
from torch.export import export_for_training

exported_model = export_for_training(model_to_quantize, example_inputs).module()
prepared_model = prepare_pt2e(exported_model, quantizer)

for cal_sample in cal_samples: # Replace with representative model inputs
	prepared_model(cal_sample) # Calibrate

quantized_model = convert_pt2e(prepared_model)
```
For static, post-training quantization (PTQ), the post-prepare\_pt2e model should be run with a representative set of samples, which are used to determine the quantization parameters.

After `convert_pt2e`, the model can be exported and lowered using the normal ExecuTorch XNNPACK flow. For more information on PyTorch 2 quantization [here](https://pytorch.org/tutorials/prototype/pt2e_quant_ptq.html).

### Testing the Model

After generating the XNNPACK-delegated .pte, the model can be tested from Python using the ExecuTorch runtime python bindings. This can be used to sanity check the model and evaluate numerical accuracy. See [Testing the Model](using-executorch-export.md#testing-the-model) for more information.

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
    executorch_module_static
    executorch_tensor
    optimized_native_cpu_ops_lib
    xnnpack_backend)
```

No additional steps are necessary to use the backend beyond linking the target. Any XNNPACK-delegated .pte file will automatically run on the registered backend.
