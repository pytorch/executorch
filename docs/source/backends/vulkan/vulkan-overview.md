# Vulkan Backend

The ExecuTorch Vulkan (ET-VK) backend enables ExecuTorch models to execute on
GPUs via the cross-platform [Vulkan API](https://www.vulkan.org/). Although the
Vulkan API support is almost ubiquitous among modern GPUs, the ExecuTorch Vulkan
backend is developed with a focus on **Android GPUs**, and support for desktop
platforms is experimental.

## Features

- Wide operator support via an in-tree [GLSL compute shader library](https://github.com/pytorch/executorch/tree/main/backends/vulkan/runtime/graph/ops/glsl)
- Support for models that require dynamic shapes
- Support for FP32 and FP16 inference modes
- Support for quantized linear layers with 8-bit/4-bit weights and 8-bit dynamically quantized activations
- Support for quantized linear layers with 8-bit/4-bit weights and FP32/FP16 activations

Note that the Vulkan backend is under active development, and its GLSL compute
shader library is being consistently expanded over time. Additional support for
quantized operators (i.e. quantized convolution) and additional quantization
modes is on the way.

## Target Requirements

- Supports Vulkan 1.1

## Development Requirements

To build the Vulkan delegate, install the
[Vulkan SDK](https://vulkan.lunarg.com/sdk/home) 1.4.341.1 or newer.
After installation, the `glslc` binary must be found in your `PATH` in order
to compile Vulkan shaders. This can be checked by running

```sh
glslc --version
```

If this is not the case after completing the Vulkan SDK installation, you may have to
go into `~/VulkanSDK/<version>/` and run

```sh
source setup-env.sh
```

or alternatively,

```sh
python install_vulkan.py
```

To target Android, also install a current
[Android NDK](https://developer.android.com/ndk/downloads). ExecuTorch CI uses
NDK r28c.

----

## Using the Vulkan Backend

To lower a model to the Vulkan backend during the export and lowering process,
pass an instance of `VulkanPartitioner` to `to_edge_transform_and_lower`. The
example below demonstrates this process using the MobileNet V2 model from
torchvision.

```python
import torch
import torchvision.models as models

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower

from torchvision.models.mobilenetv2 import MobileNet_V2_Weights

mobilenet_v2 = models.mobilenetv2.mobilenet_v2(
    weights=MobileNet_V2_Weights.DEFAULT
).eval()

sample_inputs = (torch.randn(1, 3, 224, 224),)

exported_program = torch.export.export(mobilenet_v2, sample_inputs)

etvk_program = to_edge_transform_and_lower(
    exported_program,
    partitioner=[VulkanPartitioner()],
).to_executorch()

with open("mv2_vulkan.pte", "wb") as file:
    etvk_program.write_to_file(file)
```

See {doc}`/backends/vulkan/vulkan-partitioner`
for a reference on available partitioner options.

----

## Quantization

The Vulkan delegate currently supports execution of quantized linear layers.
See {doc}`/backends/vulkan/vulkan-quantization`
for more information on available quantization schemes and APIs.

----

## Runtime Integration

To run the model on-device, use the standard ExecuTorch runtime APIs.

For integration in Android applications, the Vulkan backend is included in the
[executorch-android-vulkan](https://mvnrepository.com/artifact/org.pytorch/executorch-android-vulkan)
package.

When building from source, pass `-DEXECUTORCH_BUILD_VULKAN=ON` when configuring
the CMake build to compile the Vulkan backend. See [Running on Device](/getting-started.md#running-on-device)
for more information.

To link against the backend, use the `vulkan_backend` CMake target. The target
propagates the platform-specific linker options needed to retain the static
initializers that register Vulkan compute shaders and operators.

```cmake
# CMakeLists.txt
find_package(executorch CONFIG REQUIRED COMPONENTS vulkan_backend)

target_link_libraries(
    my_target
    PRIVATE
    executorch
    vulkan_backend
)
```

No additional steps are necessary to use the backend beyond linking the target.
Any Vulkan-delegated .pte file will automatically run on the registered backend.

## Additional Resources

**→{doc}`/backends/vulkan/vulkan-partitioner`**

**→{doc}`/backends/vulkan/vulkan-quantization`**

**→{doc}`/backends/vulkan/vulkan-troubleshooting`**

```{toctree}
:maxdepth: 2
:hidden:
:caption: Vulkan Backend

/backends/vulkan/vulkan-partitioner
/backends/vulkan/vulkan-quantization
/backends/vulkan/vulkan-op-support
/backends/vulkan/vulkan-troubleshooting

/backends/vulkan/tutorials/vulkan-tutorials
