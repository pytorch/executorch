# Model Export and Lowering

The section describes the process of taking a PyTorch model and converting to the runtime format used by ExecuTorch. This process is commonly known as "exporting", as it uses the PyTorch export functionality to convert a PyTorch model into a format suitable for on-device execution. This process yields a .pte file which is optimized for on-device execution using a particular backend.

## Prerequisites

Exporting requires the ExecuTorch python libraries to be installed, typically by running `pip install executorch`. See [Installation](getting-started.md#Installation) for more information. This process assumes you have a PyTorch model, can instantiate it from Python, and can provide example input tensors to run the model.

## The Export and Lowering Process

The process to export and lower a model to the .pte format typically involves the following steps:

1) Select a backend to target.
2) Prepare the PyTorch model, including inputs and shape specification.
3) Export the model using torch.export.export.
4) Optimize the model for the target backend using to_edge_transform_and_lower.
5) Create the .pte file by calling to_executorch and serializing the output.

<br/>

Quantization - the process of using reduced precision to reduce inference time and memory footprint - is also commonly done at this stage. See [Quantization Overview](quantization-overview.md) for more information.

## Hardware Backends

ExecuTorch backends provide hardware acceleration for a specific hardware target. In order to achieve maximum performance on target hardware, ExecuTorch optimizes the model for a specific backend during the export and lowering process. This means that the resulting .pte file is specialized for the specific hardware. In order to deploy to multiple backends, such as Core ML on iOS and Arm CPU on Android, it is common to generate a dedicated .pte file for each.

The choice of hardware backend is informed by the hardware that the model is intended to be deployed on. Each backend has specific hardware requires and level of model support. See the documentation for each hardware backend for more details.

As part of the .pte file creation process, ExecuTorch identifies portions of the model (partitions) that are supported for the given backend. These sections are processed by the backend ahead of time to support efficient execution. Portions of the model that are not supported on the delegate, if any, are executed using the portable fallback implementation on CPU. This allows for partial model acceleration when not all model operators are supported on the backend, but may have negative performance implications. In addition, multiple partitioners can be specified in order of priority. This allows for operators not supported on GPU to run on CPU via XNNPACK, for example.

### Available Backends

Commonly used hardware backends are listed below. For mobile, consider using XNNPACK for Android and XNNPACK or Core ML for iOS. To create a .pte file for a specific backend, pass the appropriate partitioner class to `to_edge_transform_and_lower`. See the appropriate backend documentation and the [Export and Lowering](#export-and-lowering) section below for more information. 

- [XNNPACK (Mobile CPU)](backends-xnnpack.md)
- [Core ML (iOS)](backends-coreml.md)
- [Metal Performance Shaders (iOS GPU)](backends-mps.md)
- [Vulkan (Android GPU)](backends-vulkan.md)
- [Qualcomm NPU](backends-qualcomm.md)
- [MediaTek NPU](backends-mediatek.md)
- [Arm Ethos-U NPU](backends-arm-ethos-u.md)
- [Cadence DSP](backends-cadence.md)

## Model Preparation

The export process takes in a standard PyTorch model, typically a `torch.nn.Module`. This can be an custom model definition, or a model from an existing source, such as TorchVision or HuggingFace. See [Getting Started with ExecuTorch](getting-started.md) for an example of lowering a TorchVision model.

Model export is done from Python. This is commonly done through a Python script or from an interactive Python notebook, such as Jupyter or Colab. The example below shows instantiation and inputs for a simple PyTorch model. The inputs are prepared as a tuple of torch.Tensors, and the model can run with these inputs.

```python
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, 3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 16, 3),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1,1))
       )
        self.linear = torch.nn.Linear(16, 10)
    
    def forward(self, x):
        y = self.seq(x)
        y = torch.flatten(y, 1)
        y = self.linear(y)
        return y

model = Model().eval()
inputs = (torch.randn(1,1,16,16),)
outputs = model(*inputs)
print(f"Model output: {outputs}")
```

Note that the model is set to evaluation mode using `.eval()`. Models should always be exported in evaluation mode unless performing on-device training. This mode configures certain operations with training-specific behavior, such as batch norm or dropout, to use the inference-mode configuration.

## Export and Lowering

To actually export and lower the model, call `export`, `to_edge_transform_and_lower`, and `to_executorch` in sequence. This yields an ExecuTorch program which can be serialized to a file. Putting it all together, lowering the example model above using the XNNPACK delegate for mobile CPU performance can be done as follows:

```python
import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from torch.export import Dim, export

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, 3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 16, 3),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1,1))
        )
        self.linear = torch.nn.Linear(16, 10)
    
    def forward(self, x):
        y = self.seq(x)
        y = torch.flatten(y, 1)
        y = self.linear(y)
        return y

model = Model()
inputs = (torch.randn(1,1,16,16),)
dynamic_shapes = {
    "x": {
        2: Dim("h", min=16, max=1024),
        3: Dim("w", min=16, max=1024),
    }
}

exported_program = export(model, inputs, dynamic_shapes=dynamic_shapes)
executorch_program = to_edge_transform_and_lower(
    exported_program,
    partitioner = [XnnpackPartitioner()]
).to_executorch()

with open("model.pte", "wb") as file:
    file.write(executorch_program.buffer)
```

This yields a `model.pte` file which can be run on mobile devices.

### Supporting Varying Input Sizes (Dynamic Shapes)

The PyTorch export process uses the example inputs provided to trace through the model and reason about the size and type of tensors at each step. Unless told otherwise, export will assume a fixed input size equal to the example inputs and will use this information to optimize the model.

Many models require support for varying input sizes. To support this, export takes a `dynamic_shapes` parameter, which informs the compiler of which dimensions can vary and their bounds. This takes the form of a nested dictionary, where keys correspond to input names and values specify the bounds for each input.

In the example model, inputs are provided as 4-dimensions tensors following the standard convention of batch, channels, height, and width (NCHW). An input with the shape `[1, 3, 16, 16]` indicates 1 batch, 3 channels, and a height and width of 16.

Suppose your model supports images with sizes between 16x16 and 1024x1024. The shape bounds can be specified as follows:

```
dynamic_shapes = {
    "x": {
        2: Dim("h", min=16, max=1024),
        3: Dim("w", min=16, max=1024),
    }
}

ep = torch.export.export(model, inputs, dynamic_shapes=dynamic_shapes)
```

In the above example, `"x"` corresponds to the parameter name in `Model.forward`. The 2 and 3 keys correpond to dimensions 2 and 3, which are height and width. As there are no specifications for batch and channel dimensions, these values are fixed according to the example inputs.

ExecuTorch uses the shape bounds both to optimize the model and to plan memory for model execution. For this reason, it is advised to set the dimension upper bounds to no higher than needed, as higher bounds increase memory consumption.

For more complex use cases, dynamic shape specification allows for mathematical relationships between dimensions. For more information on dynamic shape specification, see [Expressing Dynamism](https://pytorch.org/docs/stable/export.html#expressing-dynamism).

## Testing the Model

Before integrating the runtime code, it is common to test the exported model from Python. This can be used to evaluate model accuracy and sanity check behavior before moving to the target device. Note that not all hardware backends are available from Python, as they may require specialized hardware to function. See the specific backend documentation for more information on hardware requirements and the availablilty of simulators. The XNNPACK delegate used in this example is always available on host machines.

```python
from executorch.runtime import Runtime

runtime = Runtime.get()

input_tensor = torch.randn(1, 3, 32, 32)
program = runtime.load_program("model.pte")
method = program.load_method("forward")
outputs = method.execute([input_tensor])
```

For more information, see [Runtime API Reference](executorch-runtime-api-reference.md).

## Next Steps

The PyTorch and ExecuTorch export and lowering APIs provide a high level of customizability to meet the needs of diverse hardware and models. See [torch.export](https://pytorch.org/docs/main/export.html) and [Export API Reference](export-to-executorch-api-reference.md) for more information.

For advanced use cases, see the following:
- [Quantization Overview](quantization-overview.md) for information on quantizing models to reduce inference time and memory footprint.
- [Memory Planning](compiler-memory-planning.md) for information on controlling memory placement and planning.
- [Custom Compiler Passes](compiler-custom-compiler-passes.md) for information on writing custom compiler passes.
- [Export IR Specification](ir-exir.md) for information on the intermediate representation generated by export.
