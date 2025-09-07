# Arm&reg; Backend Tutorial

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
In this tutorial you will learn how to export a simple PyTorch model for ExecuTorch Arm backends.
:::

::::

```{warning}
This delegate is under active development, to get best results please use a recent version.
The TOSA and Ethos(tm) backend support is reasonably mature and used in production by some users.
The VGF backend support is in early development and you may encounter issues.
You may encounter some rough edges and features which may be documented or planned but not implemented, please refer to the in-tree documentation for the latest status of features.
```

```{tip}
If you are already familiar with this delegate, you may want to jump directly to the examples:
* [Examples in the ExecuTorch repository](https://github.com/pytorch/executorch/tree/main/examples/arm)
* [Compilation for Ethos-U](https://github.com/pytorch/executorch/blob/main/examples/arm/ethos_u_minimal_example.ipynb)
* [A commandline compiler for example models](https://github.com/pytorch/executorch/blob/main/examples/arm/aot_arm_compiler.py)
```

## Prerequisites

Let's make sure you have everything you need before you get started.

### Hardware

To successfully complete this tutorial, you will need a Linux or MacOS host machine with Arm aarch64 or x86_64 processor architecture.

The target device will be an emulated platform to enable development without a specific development board. This tutorial has guidance for both Ethos-U targets and VGF via the ML SDK for Vulkan®.

For Ethos-U and Cortex-M, We will be using a [Fixed Virtual Platform (FVP)](https://www.arm.com/products/development-tools/simulation/fixed-virtual-platforms), simulating [Corstone-300](https://developer.arm.com/Processors/Corstone-300)(cs300) and [Corstone-320](https://developer.arm.com/Processors/Corstone-320)(cs320)systems. Since we will be using the FVP (think of it as virtual hardware), we won't be requiring any real embedded hardware for this tutorial.

For VGF we will be using the [ML SDK for Vulkan(R)](https://github.com/arm/ai-ml-sdk-for-vulkan/)) to emulate the program consumer.

### Software

First, you will need to install ExecuTorch. Please follow the recommended tutorials if you haven't already, to set up a working ExecuTorch development environment. For the VGF backend it's recommended you [install from source](https://docs.pytorch.org/executorch/stable/using-executorch-building-from-source.html), or from a [nightly](https://download.pytorch.org/whl/nightly/executorch/).

In addition to this, you need to install a number of SDK dependencies for generating Ethos-U command streams or VGF files. There are scripts which automate this, which are found in the main [ExecuTorch repository](https://github.com/pytorch/executorch/tree/main/examples/arm/).

## Set Up the Developer Environment

In this section, we will do a one-time setup of the platform support files needed to run ExecuTorch programs in this tutorial. It is recommended to run the script in a conda or venv environment.

With a checkout of the ExecuTorch repository, we will use the `examples/arm/setup.sh` script to pull each item in an automated fashion. 

For Ethos-U run:
```bash
./examples/arm/setup.sh --i-agree-to-the-contained-eula
```

For VGF run:
```bash
./examples/arm/setup.sh --i-agree-to-the-contained-eula --disable-ethos-u-deps --enable-mlsdk-deps
```
It is possible to install both sets of dependencies if you omit the disable options.


### Notes:

```{warning}
The `setup.sh` script has generated a `setup_path.sh` script that you need to source whenever you restart your shell.
```

i.e. run
`source  executorch/examples/arm/ethos-u-scratch/setup_path.sh`


To confirm your environment is set up correctly and will enable you to generate .pte's for your target:

For Ethos-U run:
```bash
# Check for Vela, which converts TOSA to Ethos-U command streams.
which vela
```

For VGF run:
```bash
# Check for model-converter, which converts TOSA to ML-SDK VGF format.
which model-converter
```

To ensure there's no environment pollution you should confirm these binaries reside within your executorch checkout, under the examples/arm tree. Other versions may present compatibility issues, so this should be corrected by modifying your environment variables such as ${PATH} appropriately.


## Convert the PyTorch Model to the `.pte` File

`.pte` is a binary file produced by ExecuTorch Ahead-of-Time (AoT) pipeline by taking in a PyTorch Model (a torch.nn.Module), exporting it, running a variety of passes, and finally serializing it to a `.pte` file format. This binary file is typically consumed by the ExecuTorch Runtime. This [document](https://github.com/pytorch/executorch/blob/main/docs/source/getting-started-architecture.md) goes in much more depth about the ExecuTorch software stack for both AoT as well as Runtime.

In this section, we will primarily focus on the AoT flow with the end goal of producing a `.pte` file. There are a set of export configurations to target different backends at runtime. For each, the AoT flow will produce a unique `.pte` file. We will explore a couple of different configurations producing different `.pte` files, particularly interesting for our Corstone-300 system and available processing elements.

Before we get started, let's first talk about the PyTorch modules we will be using.

### PyTorch Example Modules
We will use a couple of simple PyTorch Modules to explore the end-to-end flow. These modules will be used in various different ways throughout the tutorial, referring to them by their `<class_name>`.

#### SoftmaxModule
This is a very simple PyTorch module with just one [Softmax](https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html#torch.nn.Softmax) operator.

```python
import torch

class SoftmaxModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        z = self.softmax(x)
        return z
```

Running it using the Python environment (on the same development Linux machine), you get the expected output.

```python
>>> m = SoftmaxModule()
>>> m(torch.ones(2,2))
tensor([[0.5000, 0.5000],
        [0.5000, 0.5000]])
```

#### AddModule
Let's write another simple PyTorch module with just one [Add](https://pytorch.org/docs/stable/generated/torch.add.html#torch.add) operator.

```python
class AddModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x + x
```

Running it in python shows that 1 + 1 produces 2 as exepected:

```python
>>> m = AddModule()
>>> m(torch.ones(5, dtype=torch.int32)) # integer types for non-quantized Ethos-U delegation
tensor([2, 2, 2, 2, 2], dtype=torch.int32)
```
Keep the inputs and outputs to these modules in mind. When you will lower and run this through alternate means as opposed to running on this Linux machine, you will use the same inputs, and expect the outputs to match with the one shown here.

```{tip}
you need to be aware of data types for running networks on the Ethos-U as it is an integer only co-processor. For this example you use integer types explicitly, for typical use of such a flow networks are built and trained in floating point, and then are quantized from floating point to integer for efficient inference.
```

#### MobileNetV2 Module
[MobileNetV2](https://arxiv.org/abs/1801.04381) is a commonly used network for edge and mobile devices.
It's also available as a default model in [torchvision](https://github.com/pytorch/vision), so you can load it with the sample code below.
```
from torchvision.models import mobilenet_v2  # @manual
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights

mv2 = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
```
For more details, refer to the code snippet [here](https://github.com/pytorch/executorch/blob/2354945d47f67f60d9a118ea1a08eef8ba2364b5/examples/models/mobilenet_v2/model.py#L18).

### Non-delegated Workflow

In the ExecuTorch AoT pipeline, one of the options is to select a backend. ExecuTorch offers a variety of different backends. Selecting backend is optional, it is typically done to target a particular mode of acceleration or hardware for a given model compute requirements. Without any backends, ExecuTorch runtime will fallback to using, available by default, a highly portable set of operators.

It's expected that on platforms with dedicated acceleration like the Ethos-U55, that the non-delegated flow is used for two primary cases:
1. When the network is designed to be very small and best suited to run on the Cortex-M alone.
2. When the network has a mix of operations that can target the NPU and those that can't, e.g. the Ethos-U55 supports integer operations and so floating point softmax will fall back to execute on the CPU.

In this flow, without any backend delegates, to illustrate the portability of the ExecuTorch runtime, as well as of the operator library you will skip specifying the backend during the `.pte` generation.

Following script will serve as a helper utility to help generating the `.pte` file. This is available in the `examples/arm` directory.

```bash
python3 -m examples.arm.aot_arm_compiler --model_name="softmax"
# This should produce ./softmax_arm_ethos-u55-128.pte
```

### Delegated Workflow

Working with Arm, you introduced a new Arm backend delegate for ExecuTorch. This backend is under active development and has a limited set of features available as of writing this.

By including a following step during the ExecuTorch AoT export pipeline to generate the `.pte` file, you can enable this backend delegate.

```python
from executorch.backends.arm.arm_backend import generate_ethosu_compile_spec

graph_module_edge.exported_program = to_backend(
    model.exported_program,
    ArmPartitioner(generate_ethosu_compile_spec("ethos-u55-128")))
```

Similar to the non-delegate flow, the same script will server as a helper utility to help generate the `.pte` file. Notice the `--delegate` option to enable the `to_backend` call.

For Ethos targets:
```bash
python3 -m examples.arm.aot_arm_compiler --model_name="add" --delegate
# This targets the default of ethos-u55-128, see --help for further targets
# should produce ./add_arm_delegate_ethos-u55-128.pte
```

For basic post-training quantization:
```bash
python3 -m examples.arm.aot_arm_compiler --model_name="mv2" --delegate --quantize
# This targets the default of ethos-u55-128, see --help for further targets
# should produce ./mv2_arm_delegate_ethos-u55-128.pte
```


For VGF targets:
```bash
python3 -m examples.arm.aot_arm_compiler --model_name="add" --target=vgf --delegate
# should produce ./add_arm_delegate_vgf.pte
```

For basic post-training quantization:
```bash
python3 -m examples.arm.aot_arm_compiler --model_name="mv2" --target=vgf --delegate --quantize
# should produce ./mv2_arm_delegate_vgf.pte
```

To capture intermediates such as VGF for lower level integration, invoke with the "-i" option:
```bash
python3 -m examples.arm.aot_arm_compiler --model_name="mv2" --target=vgf --delegate --quantize -i ./mv2_output
# should produce ./mv2_arm_delegate_vgf.pte and intermediates in ./mv2_out/
```

<br />

At the end of this, you should have a number of different `.pte` files.

- the SoftmaxModule, without any backend delegates.
- the AddModule, targeting the Arm Ethos-U backend.
- the Quantized MV2Model, targeting the Arm Ethos-U backend.
- the AddModule, targeting the VGF backend.
- the Quantized MV2Model, targeting the VGF backend.

Now let's try to run these `.pte` files on a target.

## Getting a Bare-Metal Executable

In this section, you will go over steps that you need to go through to build the runtime application. This then run on the target device. In the executorch repository you have a functioning script which does the exact same steps. It is located at `executorch/examples/arm/run.sh`. You will use that to build necessary pieces and finally run the previously generated PTE file on an FVP.

By default the `run.sh` will use `arm_test/` as an build and output folder and you will find the build artifacts under it. This can be controlled/overrided with the `--et_build_root` and the `--output` flags if needed.

e.g. running `examples/arm/run.sh --model_name=add --target=ethos-u85-128` will produce a pte and elf file like this:

```bash
arm_test/add/add_arm_delegate_ethos-u85-128.pte
arm_test/add/cmake-out/arm_executor_runner
```
Also before you get started, make sure that you have completed ExecuTorch cmake build setup, and the instructions to setup the development environment described [earlier](#set-up-the-developer-environment).

The block diagram below demonstrates, at the high level, how the various build artifacts are generated and are linked together to generate the final bare-metal executable.

![](arm-delegate-runtime-build.svg)

```{tip}
The `generate_pte_file` function in `run.sh` script produces the `.pte` files based on the models provided through `--model_name` input argument
```

### Generating ExecuTorch Libraries

ExecuTorch's CMake build system produces a set of build pieces which are critical to building the ExecuTorch runtime with-in the bare-metal environment you have for Corstone FVPs from Ethos-U SDK.

[This](using-executorch-building-from-source.md) document provides a detailed overview of each individual build piece. For running either variant of the `.pte` file, you will need a core set of libraries. Here is a list,

- `libexecutorch.a`
- `libportable_kernels.a`
- `libportable_ops_lib.a`

To run a `.pte` file with the Arm backend delegate call instructions, you will need the Arm backend delegate runtime library, that is,

- `libexecutorch_delegate_ethos_u.a`

These libraries are generated by the `backends/arm/scripts/build_executorch.sh` script called from the `run.sh` script.

### Building the executor_runner Bare-Metal Application

The SDK dir is the same one prepared [earlier](#setup-the-arm-ethos-u-software-development). And, you will be passing the `.pte` file (any one of them) generated above.

Note, you have to generate a new `executor-runner` binary if you want to change the model or the `.pte` file. This constraint is from the constrained bare-metal runtime environment you have for Corstone-300/Corstone-320 platforms. The build also generates a kernel registration library for the relevant operators which could not be delegated to the EthosU, see the [Kernel Library Selective Build documentation](https://docs.pytorch.org/executorch/stable/kernel-library-selective-build.html).

This step is executed by the build_executor_runner.sh script, which is invoked from the run.sh in the backends/arm/scripts folder.

```{tip}
The `run.sh` script takes in `--target` option, which provides a way to provide a specific target, Corstone-300(ethos-u55-128) or Corstone-320(ethos-u85-128)
```

## Running on Corstone FVP Platforms

Once the elf is prepared, regardless of the `.pte` file variant is used to generate the bare metal elf. `run.sh` will run the FVP for you via the `backends/arm/scripts/run_fvp.sh` script.

#### Automatic FVP Selection 

- To run a specific test model with the compiler flag and target 
```bash
./run.sh --model_name=mv2 --delegate --quantize --target=ethos-u85-128
```

- To run a specific test model and target 
```bash
./run.sh --model_name=mv2 --delegate --target=ethos-u85-128
```

- To run all the test models iteratively in a loop , simply run
```bash
./run.sh
```

Note that you could use `build_executor_runner.sh` and `run_fvp.sh` scripts in tandem by passing the relevant  --target argument (e.g., --target=ethos-u55-128), the correct FVP binary will be chosen automatically. For more details, see the [section on Runtime Integration](https://docs.pytorch.org/executorch/main/backends-arm-ethos-u.html#runtime-integration).


#### Manual FVP Binary Selection

- If you build for the Ethos delegate U55/U65 target (e.g., using --target=ethos-u55-128 or --target=ethos-u65-256 with `build_executor_runner.sh` and `run_fvp.sh`), you should use the corresponding FVP binary:
  - For U55:
    ```bash
    examples/arm/ethos-u-scratch/FVP-corstone300/models/Linux64_GCC-9.3/FVP_Corstone_SSE-300_Ethos-U55
    ```
  - For U65:
    ```bash
    examples/arm/ethos-u-scratch/FVP-corstone300/models/Linux64_GCC-9.3/FVP_Corstone_SSE-300_Ethos-U65
    ```
- And say if you are not building for an Ethos target, use:
  ```bash
  examples/arm/ethos-u-scratch/FVP-corstone320/models/Linux64_GCC-9.3/FVP_Corstone_SSE-320
  ```

Following is an example usage:

```bash
ethos_u_build_dir=examples/arm/executor_runner/

elf=$(find ${ethos_u_build_dir} -name "arm_executor_runner")

FVP_Corstone_SSE-320                                    \
    -C mps4_board.subsystem.ethosu.num_macs=128         \
    -C mps4_board.visualisation.disable-visualisation=1 \
    -C vis_hdlcd.disable_visualisation=1                \
    -C mps4_board.telnetterminal0.start_telnet=0        \
    -C mps4_board.uart0.out_file='-'                    \
    -C mps4_board.uart0.shutdown_on_eot=1               \
    -a "${elf}"                                         \
    --timelimit 120 || true # seconds- after which sim will kill itself
```

#### Verification of Successful FVP Execution
After running the FVP command, either automatically or manually, you should see output similar to the following on your shell if the execution is successful:

```console
I [executorch:arm_executor_runner.cpp:364] Model in 0x70000000 $
I [executorch:arm_executor_runner.cpp:366] Model PTE file loaded. Size: 4425968 bytes.
I [executorch:arm_executor_runner.cpp:376] Model buffer loaded, has 1 methods
I [executorch:arm_executor_runner.cpp:384] Running method forward
I [executorch:arm_executor_runner.cpp:395] Setup Method allocator pool. Size: 62914560 bytes.
I [executorch:arm_executor_runner.cpp:412] Setting up planned buffer 0, size 752640.
I [executorch:ArmBackendEthosU.cpp:79] ArmBackend::init 0x70000070
I [executorch:arm_executor_runner.cpp:445] Method loaded.
I [executorch:arm_executor_runner.cpp:447] Preparing inputs...
I [executorch:arm_executor_runner.cpp:461] Input prepared.
I [executorch:arm_executor_runner.cpp:463] Starting the model execution...
I [executorch:ArmBackendEthosU.cpp:118] ArmBackend::execute 0x70000070
I [executorch:ArmBackendEthosU.cpp:298] Tensor input/output 0 will be permuted
I [executorch:arm_perf_monitor.cpp:120] NPU Inferences : 1
I [executorch:arm_perf_monitor.cpp:121] Profiler report, CPU cycles per operator:
I [executorch:arm_perf_monitor.cpp:125] ethos-u : cycle_cnt : 1498202 cycles
I [executorch:arm_perf_monitor.cpp:132] Operator(s) total: 1498202 CPU cycles
I [executorch:arm_perf_monitor.cpp:138] Inference runtime: 6925114 CPU cycles total
I [executorch:arm_perf_monitor.cpp:140] NOTE: CPU cycle values and ratio calculations require FPGA and identical CPU/NPU frequency
I [executorch:arm_perf_monitor.cpp:149] Inference CPU ratio: 99.99 %
I [executorch:arm_perf_monitor.cpp:153] Inference NPU ratio: 0.01 %
I [executorch:arm_perf_monitor.cpp:162] cpu_wait_for_npu_cntr : 729 CPU cycles
I [executorch:arm_perf_monitor.cpp:167] Ethos-U PMU report:
I [executorch:arm_perf_monitor.cpp:168] ethosu_pmu_cycle_cntr : 5920305
I [executorch:arm_perf_monitor.cpp:171] ethosu_pmu_cntr0 : 359921
I [executorch:arm_perf_monitor.cpp:171] ethosu_pmu_cntr1 : 0
I [executorch:arm_perf_monitor.cpp:171] ethosu_pmu_cntr2 : 0
I [executorch:arm_perf_monitor.cpp:171] ethosu_pmu_cntr3 : 503
I [executorch:arm_perf_monitor.cpp:178] Ethos-U PMU Events:[ETHOSU_PMU_EXT0_RD_DATA_BEAT_RECEIVED, ETHOSU_PMU_EXT1_RD_DATA_BEAT_RECEIVED, ETHOSU_PMU_EXT0_WR_DATA_BEAT_WRITTEN, ETHOSU_PMU_NPU_IDLE]
I [executorch:arm_executor_runner.cpp:470] model_pte_loaded_size:     4425968 bytes.
I [executorch:arm_executor_runner.cpp:484] method_allocator_used:     1355722 / 62914560  free: 61558838 ( used: 2 % )
I [executorch:arm_executor_runner.cpp:491] method_allocator_planned:  752640 bytes
I [executorch:arm_executor_runner.cpp:493] method_allocator_loaded:   966 bytes
I [executorch:arm_executor_runner.cpp:494] method_allocator_input:    602116 bytes
I [executorch:arm_executor_runner.cpp:495] method_allocator_executor: 0 bytes
I [executorch:arm_executor_runner.cpp:498] temp_allocator_used:       0 / 1048576 free: 1048576 ( used: 0 % )
I [executorch:arm_executor_runner.cpp:152] Model executed successfully.
I [executorch:arm_executor_runner.cpp:156] 1 outputs:
Output[0][0]: -0.749744
Output[0][1]: -0.019224
Output[0][2]: 0.134570
...(Skipped)
Output[0][996]: -0.230691
Output[0][997]: -0.634399
Output[0][998]: -0.115345
Output[0][999]: 1.576386
I [executorch:arm_executor_runner.cpp:177] Program complete, exiting.
I [executorch:arm_executor_runner.cpp:179]
```

```{note}
The `run.sh` script provides various options to select a particular FVP target, use desired models, select portable kernels and can be explored using the `--help` argument
```

## Running on the VGF backend with the standard executor_runner for Linux

Follow typical [Building ExecuTorch with CMake](using-executorch-building-from-source.md) flow to build the linux target, ensuring that the VGF delegate is enabled.

```bash
-DEXECUTORCH_BUILD_VGF=ON
```

A full example buld line is:
```
cmake bash \
    -DCMAKE_INSTALL_PREFIX=cmake-out \
    -DCMAKE_BUILD_TYPE=Release \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
    -DEXECUTORCH_BUILD_XNNPACK=OFF \
    -DEXECUTORCH_BUILD_VULKAN=ON \
    -DEXECUTORCH_BUILD_VGF=ON \
    -DEXECUTORCH_ENABLE_LOGGING=ON \
    -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON \
    -DPYTHON_EXECUTABLE=python \
    -Bcmake-out .
cmake --build cmake-out -j25 --target install --config Release
```

You can then invoke the executor runner on the host machine, which will use the VGF delegate, and requires the vulkan layer drivers we installed with setup.sh.

```bash
./cmake-out/executor_runner -model_path add_arm_delegate_vgf.pte
```


## Takeaways
In this tutorial you have learnt how to use the ExecuTorch software to both export a standard model from PyTorch and to run it on the compact and fully functioned ExecuTorch runtime, enabling a smooth path for offloading models from PyTorch to Arm based platforms.

To recap, there are two major flows:
 * A direct flow which offloads work onto the Cortex-M using libraries built into ExecuTorch.
 * A delegated flow which partitions the graph into sections for Cortex-M and sections which can be offloaded and accelerated on the Ethos-U hardware.

Both of these flows continue to evolve, enabling more use-cases and better performance.

## FAQs
<!----
Describe what common errors users may see and how to resolve them.

* TODO - Binary size and operator Selection
* TODO - Cross-compilation targeting baremetal
* TODO - Debugging on FVP
----->

If you encountered any bugs or issues following this tutorial please file a bug/issue here on [Github](https://github.com/pytorch/executorch/issues/new).
