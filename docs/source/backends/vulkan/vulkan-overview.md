# Vulkan Backend

The ExecuTorch Vulkan (ET-VK) backend enables ExecuTorch models to execute on
GPUs via the cross-platform [Vulkan API](https://www.vulkan.org/). Although the
Vulkan API support is almost ubiquitous among modern GPUs, the ExecuTorch Vulkan
backend is currently developed with a specific focus for **Android GPUs**.

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

To contribute to the Vulkan delegate, the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home#android)
must be installed on the development system. After installation, the `glslc` binary must
be found in your `PATH` in order to compile Vulkan shaders. This can be checked by
running

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

The [Android NDK](https://developer.android.com/ndk/downloads) must also be installed.
Any NDK version past NDK r17c should suffice.

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

See [Partitioner API](vulkan-partitioner.md)
for a reference on available partitioner options.

----

## Quantization

The Vulkan delegate currently supports execution of quantized linear layers.
See [Vulkan Quantization](vulkan-quantization.md)
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

To link against the backend, add the `executorch_backends` CMake target as a
build dependency, or link directly against `libvulkan_backend`. Due to the use
of static initialization to register available compute shaders and operators,
it is required to ensure that the library is linked with `--whole-archive`.

```cmake
# CMakeLists.txt
find_package(executorch CONFIG REQUIRED COMPONENTS vulkan_backend executorch_backends)

...
target_link_libraries(
    my_target
    PRIVATE
    executorch
    executorch_backends
    ...
)

# Ensure that unused code is not discarded. The required linker options may be
# different depending on the target platform. Typically, the
# executorch_target_link_options_shared_lib function from
# executorch/tools/cmake/Utils.cmake can be used to set the required linker
# options.
target_link_options(
    executorch_backends INTERFACE "SHELL:LINKER:--whole-archive \
    $<TARGET_FILE:${target_name}> \
    LINKER:--no-whole-archive"
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

vulkan-partitioner
vulkan-quantization
vulkan-op-support
vulkan-troubleshooting

tutorials/vulkan-tutorials
