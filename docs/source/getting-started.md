# Getting Started with ExecuTorch
This section is intended to describe the necessary steps to take PyTorch model and run it using ExecuTorch. To use the framework, you will typically need to take the following steps:
- Install the ExecuTorch python package and runtime libraries.
- Export the PyTorch model for the target hardware configuration.
- Run the model using the ExecuTorch runtime APIs on your development platform.
- Deploy the model to the target platform using the ExecuTorch runtime.

## System Requirements
The following are required to install the ExecuTorch host libraries, needed to export models and run from Python. Requirements for target end-user devices are backend dependent. See the appropriate backend documentation for more information.

- Python 3.10 - 3.12
- g++ version 7 or higher, clang++ version 5 or higher, or another C++17-compatible toolchain.
- Linux or MacOS operating system (Arm or x86).
  - Windows is supported via WSL.

## Installation
To use ExecuTorch, you will need to install both the Python package and the appropriate platform-specific runtime libraries. Pip is the recommended way to install the ExecuTorch python package.

This package includes the dependencies needed to export a PyTorch model, as well as Python runtime bindings for model testing and evaluation. Consider installing ExecuTorch within a virtual environment, such as one provided by [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html#creating-environments) or [venv](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#create-and-use-virtual-environments).

```
pip install executorch
```

To build the framework from source, see [Building From Source](using-executorch-building-from-source.md). Backend delegates may require additional dependencies. See the appropriate backend documentation for more information.


<hr/>

## Preparing the Model
Exporting is the process of taking a PyTorch model and converting it to the .pte file format used by the ExecuTorch runtime. This is done using Python APIs. PTE files for common models, such as Llama 3.2, can be found on HuggingFace under [ExecuTorch Community](https://huggingface.co/executorch-community). These models have been exported and lowered for ExecuTorch, and can be directly deployed without needing to go through the lowering process.

A complete example of exporting, lowering, and verifying MobileNet V2 is available as a [Colab notebook](https://colab.research.google.com/drive/1qpxrXC3YdJQzly3mRg-4ayYiOjC6rue3?usp=sharing).

### Requirements
- A PyTorch model.
- Example model inputs, typically as PyTorch tensors. You should be able to successfully run the PyTorch model with these inputs.
- One or more target hardware backends.

### Selecting a Backend
ExecuTorch provides hardware acceleration for a wide variety of hardware. The most commonly used backends are XNNPACK, for Arm and x86 CPU, Core ML (for iOS), Vulkan (for Android GPUs), and Qualcomm (for Qualcomm-powered Android phones).

For mobile use cases, consider using XNNPACK for Android and Core ML or XNNPACK for iOS as a first step. See [Hardware Backends](backends-overview.md) for more information.

### Exporting
Exporting is done using Python APIs. ExecuTorch provides a high degree of customization during the export process, but the typical flow is as follows. This example uses the MobileNet V2 image classification model implementation in torchvision, but the process supports any [export-compliant](https://pytorch.org/docs/stable/export.html) PyTorch model.

```python
import torch
import torchvision.models as models
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower

model = models.mobilenetv2.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()
sample_inputs = (torch.randn(1, 3, 224, 224), )

et_program = to_edge_transform_and_lower(
    torch.export.export(model, sample_inputs),
    partitioner=[XnnpackPartitioner()]
).to_executorch()

with open("model.pte", "wb") as f:
    f.write(et_program.buffer)
```

If the model requires varying input sizes, you will need to specify the varying dimensions and bounds as part of the `export` call. See [Model Export and Lowering](using-executorch-export.md) for more information.

The hardware backend to target is controlled by the partitioner parameter to to\_edge\_transform\_and\_lower. In this example, the XnnpackPartitioner is used to target mobile CPUs. See the [backend-specific documentation](backends-overview.md) for information on how to use each backend.

Quantization can also be done at this stage to reduce model size and runtime. Quantization is backend-specific. See the documentation for the target backend for a full description of supported quantization schemes.

### Testing the Model

After successfully generating a .pte file, it is common to use the Python runtime APIs to validate the model on the development platform. This can be used to evaluate model accuracy before running on-device.

For the MobileNet V2 model from torchvision used in this example, image inputs are expected as a normalized, float32 tensor with a dimensions of (batch, channels, height, width). The output See [torchvision.models.mobilenet_v2](https://pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v2.html) for more information on the input and output tensor format for this model.

```python
import torch
from executorch.runtime import Runtime
from typing import List

runtime = Runtime.get()

input_tensor: torch.Tensor = torch.randn(1, 3, 224, 224)
program = runtime.load_program("model.pte")
method = program.load_method("forward")
outputs: List[torch.Tensor] = method.execute([input_tensor])
```


<hr/>

## Running on Device
ExecuTorch provides runtime APIs in Java, Objective-C, and C++.

Quick Links:
- [Android](#android)
- [iOS](#ios)
- [C++](#c)

### Android

#### Installation
ExecuTorch provides Java bindings for Android usage, which can be consumed from both Java and Kotlin.
To add the library to your app, add the following dependency to gradle build rule.

```
# app/build.gradle.kts
dependencies {
  implementation("org.pytorch:executorch-android:0.5.1")
}
```

#### Runtime APIs
Models can be loaded and run using the `Module` class:
```java
import org.pytorch.executorch.EValue;
import org.pytorch.executorch.Module;
import org.pytorch.executorch.Tensor;

// …

Module model = Module.load("/path/to/model.pte");

Tensor input_tensor = Tensor.fromBlob(float_data, new long[] { 1, 3, height, width });
EValue input_evalue = EValue.from(input_tensor);
EValue[] output = model.forward(input_evalue);
float[] scores = output[0].toTensor().getDataAsFloatArray();
```

For a full example of running a model on Android, see the [DeepLabV3AndroidDemo](https://github.com/pytorch-labs/executorch-examples/tree/main/dl3/android/DeepLabV3Demo). For more information on Android development, including building from source, a full description of the Java APIs, and information on using ExecuTorch from Android native code, see [Using ExecuTorch on Android](using-executorch-android.md).

### iOS

#### Installation
ExecuTorch supports both iOS and MacOS via C++, as well as hardware backends for CoreML, MPS, and CPU. The iOS runtime library is provided as a collection of .xcframework targets and are made available as a Swift PM package.

To get started with Xcode, go to File > Add Package Dependencies. Paste the URL of the ExecuTorch repo into the search bar and select it. Make sure to change the branch name to the desired ExecuTorch version in format “swiftpm-”, (e.g. “swiftpm-0.5.0”).  The ExecuTorch dependency can also be added to the package file manually. See [Using ExecuTorch on iOS](using-executorch-ios.md) for more information.

#### Runtime APIs
Models can be loaded and run from Objective-C using the C++ APIs.

For more information on iOS integration, including an API reference, logging setup, and building from source, see [Using ExecuTorch on iOS](using-executorch-ios.md).

### C++
ExecuTorch provides C++ APIs, which can be used to target embedded or mobile devices. The C++ APIs provide a greater level of control compared to other language bindings, allowing for advanced memory management, data loading, and platform integration.

#### Installation
CMake is the preferred build system for the ExecuTorch C++ runtime. To use with CMake, clone the ExecuTorch repository as a subdirectory of your project, and use CMake's `add_subdirectory("executorch")` to include the dependency. The `executorch` target, as well as kernel and backend targets will be made available to link against. The runtime can also be built standalone to support diverse toolchains. See [Using ExecuTorch with C++](using-executorch-cpp.md) for a detailed description of build integration, targets, and cross compilation.

```
git clone -b release/0.5 https://github.com/pytorch/executorch.git
```
```python
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

#### Runtime APIs
Both high-level and low-level C++ APIs are provided. The low-level APIs are platform independent, do not dynamically allocate memory, and are most suitable for resource-constrained embedded systems. The high-level APIs are provided as a convenience wrapper around the lower-level APIs, and make use of dynamic memory allocation and standard library constructs to reduce verbosity.

ExecuTorch uses CMake for native builds. Integration is typically done by cloning the ExecuTorch repository and using CMake add_subdirectory to add the dependency.

Loading and running a model using the high-level API can be done as follows:
```cpp
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

using namespace ::executorch::extension;

// Load the model.
Module module("/path/to/model.pte");

// Create an input tensor.
float input[1 * 3 * 256 * 256];
auto tensor = from_blob(input, {1, 3, 256, 256});

// Perform an inference.
const auto result = module.forward(tensor);

if (result.ok()) {
  // Retrieve the output data.
  const auto output = result->at(0).toTensor().const_data_ptr<float>();
}
```

For more information on the C++ APIs, see [Running an ExecuTorch Model Using the Module Extension in C++](extension-module.md) and [Managing Tensor Memory in C++](extension-tensor.md).

<hr/>

## Next Steps
ExecuTorch provides a high-degree of customizability to support diverse hardware targets. Depending on your use cases, consider exploring one or more of the following pages:

- [Export and Lowering](using-executorch-export.md) for advanced model conversion options.
- [Backend Overview](backends-overview.md) for available backends and configuration options.
- [Using ExecuTorch on Android](using-executorch-android.md) and [Using ExecuTorch on iOS](using-executorch-ios.md) for mobile runtime integration.
- [Using ExecuTorch with C++](using-executorch-cpp.md) for embedded and mobile native development.
- [Profiling and Debugging](using-executorch-troubleshooting.md) for developer tooling and debugging.
- [API Reference](export-to-executorch-api-reference.md) for a full description of available APIs.
- [Examples](https://github.com/pytorch/executorch/tree/main/examples) for demo apps and example code.
