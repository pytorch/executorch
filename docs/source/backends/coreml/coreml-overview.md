# Core ML Backend

Core ML delegate is the ExecuTorch solution to take advantage of Apple's [Core ML framework](https://developer.apple.com/documentation/coreml) for on-device ML.  With Core ML, a model can run on CPU, GPU, and the Apple Neural Engine (ANE).

## Features

- Dynamic dispatch to the CPU, GPU, and ANE.
- Supports fp32 and fp16 computation.

## Target Requirements

Below are the minimum OS requirements on various hardware for running a Core ML-delegated ExecuTorch model:
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

## Using the Core ML Backend

To target the Core ML backend during the export and lowering process, pass an instance of the `CoreMLPartitioner` to `to_edge_transform_and_lower`. The example below demonstrates this process using the MobileNet V2 model from torchvision.

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

See [Partitioner API](coreml-partitioner.md) for a reference on available partitioner options.

----

## Quantization

The Core ML delegate can also be used as a backend to execute quantized models. See [Core ML Quantization](coreml-quantization.md) for more information on available quantization schemes and APIs.


## Backward compatibility

Core ML supports backward compatibility via the [`minimum_deployment_target`](coreml-partitioner.md#coreml-compilespec) option.  A model exported with a specific deployment target is guaranteed to work on all deployment targets >= the specified deployment target.  For example, a model exported with `coremltools.target.iOS17` will work on iOS 17 or higher.

----

## Runtime integration

To run the model on device, use the standard ExecuTorch runtime APIs. See [Running on Device](getting-started.md#running-on-device) for more information, including building the iOS frameworks.

When building from source, pass `-DEXECUTORCH_BUILD_COREML=ON` when configuring the CMake build to compile the Core ML backend.

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

No additional steps are necessary to use the backend beyond linking the target. A Core ML-delegated .pte file will automatically run on the registered backend.


## Reference

**→{doc}`coreml-troubleshooting` — Debug common issues.**

**→{doc}`coreml-partitioner` — Partitioner options.**

**→{doc}`coreml-quantization` — Supported quantization schemes.**

**→{doc}`coreml-op-support` — Supported operators.**

```{toctree}
:maxdepth: 2
:hidden:
:caption: Core ML Backend
coreml-troubleshooting
coreml-partitioner
coreml-quantization
coreml-op-support
