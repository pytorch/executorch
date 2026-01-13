# Samsung Exynos Backend

ExecuTorch's Samsung Exynos backend enables the execution of ExecuTorch models on
Samsung SoCs via the NPU/DSP. The delegate is built on top of the
[Samsung Exynos AI Litecore SDK](https://soc-developer.semiconductor.samsung.com/global/development/ai-litecore).

## Features

- Wide range of operator support
- Supported inference precisions:
  - FP16
  - 8-bit statically quantized (int8/uint8)
  - 16-bit statically quantized (int16/uint16)

## Target Requirements

Currently, the Samsung Exynos backend is supported only for devices with the
following chipsets:

- Exynos 2500 (E9955)

## Development Requirements

The [Samsung Exynos AI Litecore SDK](https://soc-developer.semiconductor.samsung.com/global/development/ai-litecore)
is required to build the Exynos backend from source, and is also required to
export models to the Exynos delegate.

----

## Using the Samsung Exynos Backend

To target the Exynos backend during the export and lowering process, pass an instance of
the `EnnPartitioner` to `to_edge_transform_and_lower`. The example below
demonstrates this process using the MobileNet V2 model from torchvision.

```python
import torch
import torchvision.models as models
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from executorch.backends.samsung.partition.enn_partitioner import EnnPartitioner
from executorch.backends.samsung.serialization.compile_options import (
    gen_samsung_backend_compile_spec,
)
from executorch.exir import to_edge_transform_and_lower

mobilenet_v2 = models.mobilenetv2.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()
sample_inputs = (torch.randn(1, 3, 224, 224), )

chipset = "E9955"
compile_specs = [gen_samsung_backend_compile_spec(chipset)]

et_program = to_edge_transform_and_lower(
    torch.export.export(mobilenet_v2, sample_inputs),
    partitioner=[EnnPartitioner(compile_specs)],
).to_executorch()

with open("mv2_xnnpack.pte", "wb") as file:
    et_program.write_to_file(file)
```

See [Partitioner API](samsung-partitioner.md) for a reference on available partitioner options.

----

## Quantization

The Samsung Exynos backend support statically quantized models with 8-bit and 16-bit
integral types.

See [Samsung Exynos Quantization](samsung-quantization.md) for more
information on available quantization schemes and APIs.

----

## Runtime Integration

To run the model on-device, use the standard ExecuTorch runtime APIs.

The Exynos backend is currently not available in any of ExecuTorch's published packages.
To access it, build ExecuTorch from source. When building from source, pass
`-DEXECUTORCH_BUILD_EXYNOS=ON` when configuring the CMake build. See [Running on Device](/getting-started.md#running-on-device)
for more information.

Then, to link against the backend, add the `executorch_backends` CMake target as a build
dependency.

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

No additional steps are necessary to use the backend beyond linking the target. Any
Exynos delegated .pte file will automatically run on the registered backend.

## Reference

**→{doc}`samsung-partitioner` — Partitioner options.**

**→{doc}`samsung-quantization` — Supported quantization schemes.**

**→{doc}`samsung-op-support` — Supported operators.**

```{toctree}
:maxdepth: 2
:hidden:
:caption: Exynos Backend

samsung-partitioner
samsung-quantization
samsung-op-support
