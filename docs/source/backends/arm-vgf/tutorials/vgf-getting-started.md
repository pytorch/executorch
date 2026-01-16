# Getting Started Tutorial

<!----This will show a grid card on the page----->
::::{grid} 2

:::{grid-item-card}  Tutorials we recommend you complete before this:
:class-card: card-prerequisites
* [Introduction to ExecuTorch](intro-how-it-works.md)
* [Getting Started](getting-started.md)
* [Building ExecuTorch with CMake](using-executorch-building-from-source.md)
:::

:::{grid-item-card}  What you will learn in this tutorial:
:class-card: card-prerequisites
In this tutorial you will learn how to export a simple PyTorch model for the ExecuTorch VGF backend.
:::

::::

```{warning}
This delegate is under active development, to get best results please use a recent version.
The VGF backend support is in early development and you may encounter issues.
You may encounter some rough edges and features which may be documented or planned but not implemented, please refer to the in-tree documentation for the latest status of features.
```

```{tip}
If you are already familiar with this delegate, you may want to jump directly to the examples:
* [Examples in the ExecuTorch repository](https://github.com/pytorch/executorch/tree/main/examples/arm)
* [A commandline compiler for example models](https://github.com/pytorch/executorch/blob/main/examples/arm/aot_arm_compiler.py)
```

This tutorial serves as an introduction to using ExecuTorch to deploy PyTorch models on VGF targets. The tutorial is based on `vgf_minimal_example.ipyb`, provided in Arm's example folder.

## Prerequisites

### Hardware

To successfully complete this tutorial, you will need a Linux machine with aarch64 or x86_64 processor architecture, or a macOS&trade; machine with Apple&reg; Silicon.

To enable development without a specific development board, we will be using the [ML SDK for Vulkan&reg;](https://github.com/arm/ai-ml-sdk-for-vulkan/) to emulate the program consumer.

### Software

First, you will need to install ExecuTorch. Please follow the recommended tutorials if you haven't already, to set up a working ExecuTorch development environment. For the VGF backend it's recommended you [install from source](https://docs.pytorch.org/executorch/stable/using-executorch-building-from-source.html), or from a [nightly](https://download.pytorch.org/whl/nightly/executorch/).

In addition to this, you need to install a number of SDK dependencies for generating VGF files. Scripts to automate this are available in the main [ExecuTorch repository](https://github.com/pytorch/executorch/tree/main/examples/arm/). To install VGF dependencies, run
```bash
./examples/arm/setup.sh --i-agree-to-the-contained-eula --disable-ethos-u-deps --enable-mlsdk-deps
```
This will install:
- [TOSA Serialization Library](https://www.mlplatform.org/tosa/software.html) for serializing the Exir IR graph into TOSA IR.
- [ML SDK Model Converter](https://github.com/arm/ai-ml-sdk-model-converter) for converting TOSA flatbuffers to VGF files.
- [Vulkan API](https://www.vulkan.org) should be set up locally for GPU execution support.
- [ML Emulation Layer for Vulkan](https://github.com/arm/ai-ml-emulation-layer-for-vulkan) for testing on Vulkan API.


## Set Up the Developer Environment

The `setup.sh` script has generated a `setup_path.sh` script that you need to source whenever you restart your shell. Do this by running

`source examples/arm/arm-scratch/setup_path.sh`

As a simple check that your environment is set up correctly, run

```bash
which model-converter
```
Make sure the executable is located where you expect, in the `examples/arm` tree. 

## Build

### Ahead-of-Time (AOT) components

The ExecuTorch Ahead-of-Time (AOT) pipeline takes a PyTorch Model (a `torch.nn.Module`) and produces a `.pte` binary file, which is then typically consumed by the ExecuTorch Runtime. This [document](https://github.com/pytorch/executorch/blob/main/docs/source/getting-started-architecture.md) goes in much more depth about the ExecuTorch software stack for both AoT as well as Runtime.

The example below shows how to quantize a model consisting of a single addition, and export it it through the AOT flow using the VGF backend. For more details, se `examples/arm/vgf_minimal_example.ipynb`.

```python
import torch

class AddSigmoid(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(x + y)

example_inputs = (torch.ones(1,1,1,1),torch.ones(1,1,1,1))

model = AddSigmoid()
model = model.eval()
exported_program = torch.export.export(model, example_inputs)
graph_module = exported_program.module(check_guards=False)


from executorch.backends.arm.quantizer import (
    VgfQuantizer,
    get_symmetric_quantization_config,
)
from executorch.backends.arm.vgf import VgfCompileSpec
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

# Create a compilation spec describing the target for configuring the quantizer
compile_spec = VgfCompileSpec()

# Create and configure quantizer to use a symmetric quantization config globally on all nodes
quantizer = VgfQuantizer(compile_spec)
operator_config = get_symmetric_quantization_config(is_per_channel=False)

# Set default quantization config for the layers in the models.
# Can also be set to `None` to let layers run in FP as default.
quantizer.set_global(operator_config)

# OPTIONAL: skip quantizing all sigmoid ops (only one for this model); let it run in FP
quantizer.set_module_type(torch.nn.Sigmoid, None)

# Post training quantization
quantized_graph_module = prepare_pt2e(graph_module, quantizer)
quantized_graph_module(*example_inputs) # Calibrate the graph module with the example input
quantized_graph_module = convert_pt2e(quantized_graph_module)


# Create a new exported program using the quantized_graph_module
quantized_exported_program = torch.export.export(quantized_graph_module, example_inputs)
import os
from executorch.backends.arm.vgf import VgfPartitioner
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.extension.export_util.utils import save_pte_program

# Create partitioner from compile spec
partitioner = VgfPartitioner(compile_spec)

# Lower the exported program to the VGF backend
edge_program_manager = to_edge_transform_and_lower(
            quantized_exported_program,
            partitioner=[partitioner],
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False,
            ),
)

# Convert edge program to executorch
executorch_program_manager = edge_program_manager.to_executorch(
            config=ExecutorchBackendConfig(extract_delegate_segments=False)
)


# Save pte file
cwd_dir = os.getcwd()
pte_base_name = "simple_example"
pte_name = pte_base_name + ".pte"
pte_path = os.path.join(cwd_dir, pte_name)
save_pte_program(executorch_program_manager, pte_name)
assert os.path.exists(pte_path), "Build failed; no .pte-file found"
```


```{tip}
For a quick start, you can use the script `examples/arm/aot_arm_compiler.py` to produce the pte.
To produce a pte file equivalent to the one above, run
`python -m examples.arm.aot_arm_compiler --model_name=add --delegate --quantize --output=simple_example.pte --target=vgf` 
```

### Runtime:

## Build executor runtime

After the AOT compilation flow is done, we can build the executor runner target. For this tutorial, the default runner can be used. Build it with the following configuration:

```bash
# In ExecuTorch top-level, with sourced setup_path.sh
cmake \
  -DCMAKE_INSTALL_PREFIX=cmake-out \
  -DCMAKE_BUILD_TYPE=Debug \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
  -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
  -DEXECUTORCH_BUILD_XNNPACK=OFF \
  -DEXECUTORCH_BUILD_VULKAN=ON \
  -DEXECUTORCH_BUILD_VGF=ON \
  -DEXECUTORCH_ENABLE_LOGGING=ON \
  -DPYTHON_EXECUTABLE=python \
  -Bcmake-out .

cmake --build cmake-out --target executor_runner`
```


The block diagram below demonstrates, at the high level, how the various build artifacts are generated and are linked together to generate the final bare-metal executable.

![](arm-delegate-runtime-build.svg)


## Deploying and running on device

Since we are using the Vulkan emulation layer, we can run the executor runner with the VGF delegate on the host machine:

```bash
./cmake-out/executor_runner -model_path simple_example.pte
```

The example application is by default built with an input of ones, so the expected result of the quantized addition should be close to 2.

## Takeaways

In this tutorial you have learned how to use ExecuTorch to export a PyTorch model to an executable that can run on an embedded target, and then run that executable on simulated hardware.


## FAQs

Issue: glslc is not found when configuring the executor runner.
Solution: The Vulkan sdk is likely not in your path, check whether setup_path.sh contains something like
`export PATH=$(pwd)/examples/arm/arm-scratch/vulkan_sdk/1.4.321.1/x86_64/bin:$PATH`.
If not, add it and source the file.

If you encountered any bugs or issues following this tutorial please file a bug/issue here on [Github](https://github.com/pytorch/executorch/issues/new).
