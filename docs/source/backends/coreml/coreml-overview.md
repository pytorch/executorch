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

To run the model on device, use the standard ExecuTorch runtime APIs. See [Running on Device](../../getting-started.md#running-on-device) for more information, including building the iOS frameworks.

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

### IOSurface-backed inputs and outputs (experimental)

A Core ML model can read its inputs from and write its outputs to IOSurface-backed pixel buffers in place, instead of Core ML copying them into and out of its own buffers. That saves a copy per tensor per inference, which matters for large fp16 tensors such as video frames. This API is experimental and may change.

Requirements:

- iOS 16, macOS 12, tvOS 16, watchOS 9 or later.
- The method must be nothing but one Core ML delegate call: a single delegate, a single instruction, and only tensor inputs and outputs that aren't memory planned. Otherwise loading with IOSurface options fails with `Error::NotSupported`. Disable memory planning for the inputs and outputs when exporting:

  ```python
  et_program = to_edge_transform_and_lower(
      exported_program, partitioner=[CoreMLPartitioner()]
  ).to_executorch(
      ExecutorchBackendConfig(
          memory_planning_pass=MemoryPlanningPass(
              alloc_graph_input=False, alloc_graph_output=False
          )
      )
  )
  ```

- Bound inputs and outputs must be fp16 tensors, each in an IOSurface-backed `kCVPixelFormatType_OneComponent16Half` pixel buffer whose width is the tensor's last dimension and whose height is the product of its other dimensions.

The example binds the input and output of a model that maps a `[1, 1, 180, 320]` fp16 input to a `[1, 1, 360, 640]` output:

```objc
#include <executorch/backends/apple/coreml/runtime/include/coreml_backend/coreml_backend_options.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

#import <CoreVideo/CoreVideo.h>
#import <IOSurface/IOSurfaceRef.h>

using executorch::aten::ScalarType;
using executorch::backends::coreml::LoadOptionsBuilder;
using executorch::extension::from_blob;
using executorch::extension::Module;
using executorch::runtime::BackendOptions;
using executorch::runtime::Error;
using executorch::runtime::LoadBackendOptionsMap;

CVPixelBufferRef makeSurfaceBuffer(size_t width, size_t height) {
  NSDictionary *attributes = @{(id)kCVPixelBufferIOSurfacePropertiesKey: @{}};
  CVPixelBufferRef buffer = NULL;
  CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_OneComponent16Half,
                      (__bridge CFDictionaryRef)attributes, &buffer);
  return buffer;
}

void *baseAddress(CVPixelBufferRef buffer) {
  CVPixelBufferLockBaseAddress(buffer, kCVPixelBufferLock_ReadOnly);
  void *address = CVPixelBufferGetBaseAddress(buffer);
  CVPixelBufferUnlockBaseAddress(buffer, kCVPixelBufferLock_ReadOnly);
  return address;
}

CVPixelBufferRef input = makeSurfaceBuffer(320, 180);
CVPixelBufferRef output = makeSurfaceBuffer(640, 360);

// Name each surface by its IOSurface ID. The key only has to start with
// kIOSurfaceSpecPrefix ("iosurface_") and be unique.
BackendOptions<2> options;
options.set_option("iosurface_0", static_cast<int>(IOSurfaceGetID(CVPixelBufferGetIOSurface(input))));
options.set_option("iosurface_1", static_cast<int>(IOSurfaceGetID(CVPixelBufferGetIOSurface(output))));
LoadBackendOptionsMap backendOptions;
backendOptions.set_options(LoadOptionsBuilder::backend_id(), options.view());

Module module("/path/to/model.pte");
// The surfaces are bound when the method loads. If loading fails, run the
// model with ordinary buffers instead.
if (module.load(backendOptions) != Error::Ok || module.load_method("forward") != Error::Ok) {
  // ...
}

// Tensors whose data is exactly a surface's base address use that surface.
auto inputTensor = from_blob(baseAddress(input), {1, 1, 180, 320}, ScalarType::Half);
auto outputTensor = from_blob(baseAddress(output), {1, 1, 360, 640}, ScalarType::Half);
module.set_output("forward", outputTensor, 0);

// Write the input row by row: rows are padded, so step by bytes-per-row.
CVPixelBufferLockBaseAddress(input, 0);
auto *inputBytes = static_cast<uint8_t *>(CVPixelBufferGetBaseAddress(input));
for (size_t y = 0; y < 180; ++y) {
  auto *row = reinterpret_cast<_Float16 *>(inputBytes + y * CVPixelBufferGetBytesPerRow(input));
  // Fill row[0] ... row[319].
}
CVPixelBufferUnlockBaseAddress(input, 0);

module.forward(inputTensor);

// Read the output the same way.
CVPixelBufferLockBaseAddress(output, kCVPixelBufferLock_ReadOnly);
auto *outputBytes = static_cast<const uint8_t *>(CVPixelBufferGetBaseAddress(output));
for (size_t y = 0; y < 360; ++y) {
  auto *row = reinterpret_cast<const _Float16 *>(outputBytes + y * CVPixelBufferGetBytesPerRow(output));
  // Read row[0] ... row[639].
}
CVPixelBufferUnlockBaseAddress(output, kCVPixelBufferLock_ReadOnly);
```

How the binding works:

- Each int option whose key starts with `executorch::backends::coreml::kIOSurfaceSpecPrefix` (`"iosurface_"`) names one IOSurface by its ID. The rest of the key only makes the key unique; it doesn't say which input or output the surface is for.
- When the method loads, the backend looks up each named surface and binds it to that model instance. The load fails if a surface can't be found or isn't an fp16 one-component buffer. The options come with each load, so each model instance has its own bindings, and the model keeps the surfaces it bound alive.
- When the method runs, an input or output whose data pointer is exactly a bound surface's base address uses that surface in place. A tensor whose shape doesn't fit its surface fails the run.

Things to keep in mind:

- Surface rows are padded, for example to 64 bytes. Read and write them through `CVPixelBufferGetBytesPerRow`. Only the bound path knows about the padding: anything else that gets the surface's memory, such as another backend, a CPU kernel or a plain copy, treats it as a dense tensor and reads or writes the wrong elements.
- `Module::load` doesn't load methods. A method loads on `load_method`, or on its first `forward` or `execute`, and that's when the surfaces are looked up, so they have to exist until then.
- Core ML reads the inputs and writes the outputs while the method runs. Write the inputs and read the outputs only between runs, lock a pixel buffer for CPU access, and don't run the same model from two threads at once. When models share a surface, for example one model's output bound as another's input, finish running the model that writes it before running the one that reads it.

## Reference

**→{doc}`/backends/coreml/coreml-troubleshooting` — Debug common issues.**

**→{doc}`/backends/coreml/coreml-partitioner` — Partitioner options.**

**→{doc}`/backends/coreml/coreml-quantization` — Supported quantization schemes.**

**→{doc}`/backends/coreml/coreml-op-support` — Supported operators.**

```{toctree}
:maxdepth: 2
:hidden:
:caption: Core ML Backend
coreml-troubleshooting
coreml-partitioner
coreml-quantization
coreml-op-support
