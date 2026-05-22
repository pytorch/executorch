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

See [Partitioner API](/backends/xnnpack/xnnpack-partitioner) for a reference on available partitioner options. <!-- @lint-ignore -->

----

## Quantization

The XNNPACK delegate can also be used as a backend to execute symmetrically quantized models. See [XNNPACK Quantization](/backends/xnnpack/xnnpack-quantization) for more information on available quantization schemes and APIs. <!-- @lint-ignore -->

----

## Runtime Integration

To run the model on-device, use the standard ExecuTorch runtime APIs.

The XNNPACK delegate is included by default in the published Android, iOS, and pip packages. When building from source, pass `-DEXECUTORCH_BUILD_XNNPACK=ON` when configuring the CMake build to compile the XNNPACK backend. See [Running on Device](/getting-started.md#running-on-device) for more information.

To link against the backend, add the `executorch_backends` CMake target as a build dependency, or link directly against `libxnnpack_backend`. Due to the use of static registration, it may be necessary to link with whole-archive. This can typically be done by passing `"$<LINK_LIBRARY:WHOLE_ARCHIVE,xnnpack_backend>"` to `target_link_libraries`.

```
# CMakeLists.txt
add_subdirectory("executorch")
...
target_link_libraries(
    my_target
    PRIVATE executorch
    executorch_backends
    ...
)
```

No additional steps are necessary to use the backend beyond linking the target. Any XNNPACK-delegated .pte file will automatically run on the registered backend.

## Reference

**→{doc}`/backends/xnnpack/xnnpack-troubleshooting` — Debug common issues.**

**→{doc}`/backends/xnnpack/xnnpack-partitioner` — Partitioner options and supported operators.**

**→{doc}`/backends/xnnpack/xnnpack-quantization` — Supported quantization schemes.**

**→{doc}`/backends/xnnpack/xnnpack-arch-internals` — XNNPACK backend internals.**

```{toctree}
:maxdepth: 2
:hidden:
:caption: XNNPACK Backend

xnnpack-partitioner
xnnpack-quantization
xnnpack-troubleshooting
xnnpack-arch-internals
```
