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

Placeholder - document available quantization flows (PT2E + ao), schemes, and operators.

## Runtime Integration

The XNNPACK delegate is included by default in the published Android, iOS, and pip packages. When building from source, pass `-DEXECUTORCH_BUILD_XNNPACK=ON` when configuring the CMake build to compile the XNNPACK backend.

To link against the backend, add the `xnnpack_backend` CMake target as a build dependency, or link directly against `libxnnpack_backend`. Due to the use of static registration, it may be necessary to link with whole-archive. This can typically be done by passing the following flags: `-Wl,--whole-archive libxnnpack_backend.a -Wl,--no-whole-archive`.

No additional steps are necessary to use the backend beyond linking the target. Any XNNPACK-delegated .pte file will automatically run on the registered backend.

### Runner

To test XNNPACK models on a development machine, the repository includes a runner binary, which can run XNNPACK delegated models. It is built by default when building the XNNPACK backend. The runner can be invoked with the following command, assuming that the CMake build directory is named cmake-out. Note that the XNNPACK delegate is also available by default from the Python runtime bindings (see [Testing the Model](using-executorch-export.md#testing-the-model) for more information).
```
./cmake-out/backends/xnnpack/xnn_executor_runner --model_path=./mv2_xnnpack.pte
```