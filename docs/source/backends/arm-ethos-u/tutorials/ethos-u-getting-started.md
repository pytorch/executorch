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
In this tutorial you will learn how to export a simple PyTorch model for the ExecuTorch Ethos-U backend.
:::

::::

```{tip}
If you are already familiar with this delegate, you may want to jump directly to the examples:
* [Examples in the ExecuTorch repository](https://github.com/pytorch/executorch/tree/main/examples/arm)
* [A commandline compiler for example models](https://github.com/pytorch/executorch/blob/main/examples/arm/aot_arm_compiler.py)
```

This tutorial serves as an introduction to using ExecuTorch to deploy PyTorch models on Arm&reg; Ethos&trade;-U targets. It is based on `ethos_u_minimal_example.ipynb`, provided in Arm’s examples folder.

## Prerequisites

### Hardware

To successfully complete this tutorial, you will need a Linux machine with aarch64 or x86_64 processor architecture, or a macOS&trade; machine with Apple&reg; Silicon.

To enable development without a specific development board, we will be using a [Fixed Virtual Platform (FVP)](https://www.arm.com/products/development-tools/simulation/fixed-virtual-platforms), simulating [Arm&reg; Corstone&trade;-300](https://developer.arm.com/Processors/Corstone-300)(cs300) and [Arm&reg; Corstone&trade;-320](https://developer.arm.com/Processors/Corstone-320)(cs320)systems. Think of it as virtual hardware.

### Software

First, you will need to install ExecuTorch. Please follow the recommended tutorials to set up a working ExecuTorch development environment.

In addition to this, you need to install a number of SDK dependencies for generating Ethos-U command streams. Scripts to automate this are available in the main [ExecuTorch repository](https://github.com/pytorch/executorch/tree/main/examples/arm/).
To install Ethos-U dependencies, run
```bash
./examples/arm/setup.sh --i-agree-to-the-contained-eula
```
This will install:
- [TOSA Serialization Library](https://www.mlplatform.org/tosa/software.html) for serializing the Exir IR graph into TOSA IR.
- [Ethos-U Vela graph compiler](https://pypi.org/project/ethos-u-vela/) for compiling TOSA flatbuffers into a Ethos-U command stream.
- [Arm GNU Toolchain](https://developer.arm.com/Tools%20and%20Software/GNU%20Toolchain) for cross compilation.
- [Corstone SSE-300 FVP](https://developer.arm.com/documentation/100966/1128/Arm--Corstone-SSE-300-FVP) for testing on Ethos-U55 reference design.
- [Corstone SSE-320 FVP](https://developer.arm.com/documentation/109760/0000/SSE-320-FVP) for testing on Ethos-U85 reference design.

## Set Up the Developer Environment

The setup.sh script generates a setup_path.sh script that you need to source whenever you restart your shell. Run:

```{bash}
source  examples/arm/arm-scratch/setup_path.sh
```

As a simple check that your environment is set up correctly, run `which FVP_Corstone_SSE-320` and make sure that the executable is located where you expect, in the `examples/arm` tree.

## Build

### Ahead-of-Time (AOT) components

The ExecuTorch Ahead-of-Time (AOT) pipeline takes a PyTorch Model (a `torch.nn.Module`) and produces a `.pte` binary file, which is then consumed by the ExecuTorch Runtime. This [document](getting-started-architecture.md) goes in much more depth about the ExecuTorch software stack for both AoT as well as Runtime.

The example below shows how to quantize a model consisting of a single addition, and export it it through the AOT flow using the EthosU backend. For more details, see `examples/arm/ethos_u_minimal_example.ipynb`.
```python
import torch

class Add(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y

example_inputs = (torch.ones(1,1,1,1),torch.ones(1,1,1,1))

model = Add()
model = model.eval()
exported_program = torch.export.export(model, example_inputs)
graph_module = exported_program.module(check_guards=False)


from executorch.backends.arm.ethosu import EthosUCompileSpec
from executorch.backends.arm.quantizer import (
    EthosUQuantizer,
    get_symmetric_quantization_config,
)
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

# Create a compilation spec describing the target for configuring the quantizer
# Some args are used by the Arm Vela graph compiler later in the example. Refer to Arm Vela documentation for an
# explanation of its flags: https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-vela/-/blob/main/OPTIONS.md
compile_spec = EthosUCompileSpec(
            target="ethos-u55-128",
            system_config="Ethos_U55_High_End_Embedded",
            memory_mode="Shared_Sram",
            extra_flags=["--output-format=raw", "--debug-force-regor"]
        )

# Create and configure quantizer to use a symmetric quantization config globally on all nodes
quantizer = EthosUQuantizer(compile_spec)
operator_config = get_symmetric_quantization_config()
quantizer.set_global(operator_config)

# Post training quantization
quantized_graph_module = prepare_pt2e(graph_module, quantizer)
quantized_graph_module(*example_inputs) # Calibrate the graph module with the example input
quantized_graph_module = convert_pt2e(quantized_graph_module)


# Create a new exported program using the quantized_graph_module
quantized_exported_program = torch.export.export(quantized_graph_module, example_inputs)
from executorch.backends.arm.ethosu import EthosUPartitioner
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.extension.export_util.utils import save_pte_program

# Create partitioner from compile spec
partitioner = EthosUPartitioner(compile_spec)

# Lower the exported program to the Ethos-U backend
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
save_pte_program(executorch_program_manager, "ethos_u_minimal_example.pte")
```


```{tip}
For a quick start, you can use the script `examples/arm/aot_arm_compiler.py` to produce the pte.
To produce a pte file equivalent to the one above, run
`python -m examples.arm.aot_arm_compiler --model_name=add --delegate --quantize --output=ethos_u_minimal_example.pte`
```

### Runtime:

After the AOT compilation flow is done, the runtime can be cross compiled and linked to the produced `.pte`-file using the Arm cross-compilation toolchain. This is done in two steps:

First, build and install the ExecuTorch libraries and EthosUDelegate:
```
# In ExecuTorch top-level, with sourced setup_path.sh
cmake -DCMAKE_BUILD_TYPE=Release --preset arm-baremetal -B cmake-out-arm .
cmake --build cmake-out-arm --target install -j$(nproc)
```
Second, build and link the `arm_executor_runner` and generate kernel bindings for any non delegated ops. This is the actual program that will run on target.

```
# In ExecuTorch top-level, with sourced setup_path.sh
cmake -DCMAKE_TOOLCHAIN_FILE=`pwd`/examples/arm/ethos-u-setup/arm-none-eabi-gcc.cmake \
      -DCMAKE_BUILD_TYPE=Release \
      -DET_PTE_FILE_PATH=ethos_u_minimal_example.pte \
      -DTARGET_CPU=cortex-m55 \
      -DETHOSU_TARGET_NPU_CONFIG=ethos-u55-128 \
      -DMEMORY_MODE=Shared_Sram \
      -DSYSTEM_CONFIG=Ethos_U55_High_End_Embedded \
      -Bethos_u_minimal_example \
      examples/arm/executor_runner
cmake --build ethos_u_minimal_example -j$(nproc) -- arm_executor_runner
```

```{tip}
For a quick start, you can use the script `backends/arm/scripts/build_executor_runner.sh` to build the runner.
To build a runner equivalent to the one above, run
`./backends/arm/scripts/build_executor_runner.sh --pte=ethos_u_minimal_example.pte`
````

The block diagram below shows, at the high level, how the various build artifacts are generated and are linked together to generate the final bare-metal executable.

![](arm-delegate-runtime-build.svg)


## Running on Corstone FVP Platforms

Finally, use the `backends/arm/scripts/run_fvp.sh` utility script to run the .elf-file on simulated Arm hardware.
```
backends/arm/scripts/run_fvp.sh --elf=$(find ethos_u_minimal_example -name arm_executor_runner) --target=ethos-u55-128
```
The example application is by default built with an input of ones, so the expected result of the quantized addition should be close to 2.


## Takeaways

In this tutorial you have learned how to use ExecuTorch to export a PyTorch model to an executable that can run on an embedded target, and then run that executable on simulated hardware.
To learn more, check out these learning paths:

https://learn.arm.com/learning-paths/embedded-and-microcontrollers/rpi-llama3/
https://learn.arm.com/learning-paths/embedded-and-microcontrollers/visualizing-ethos-u-performance/

## FAQs

If you encountered any bugs or issues following this tutorial please file a bug/issue here on [Github](https://github.com/pytorch/executorch/issues/new).


```
Arm is a registered trademark of Arm Limited (or its subsidiaries or affiliates).
```
