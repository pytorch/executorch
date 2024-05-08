<!---- Name is a WIP - this reflects better what it can do today ----->
# Building and Running ExecuTorch with ARM Ethos-U Backend

<!----This will show a grid card on the page----->
::::{grid} 2

:::{grid-item-card}  Tutorials we recommend you complete before this:
:class-card: card-prerequisites
* [Introduction to ExecuTorch](./intro-how-it-works.md)
* [Setting up ExecuTorch](./getting-started-setup.md)
* [Building ExecuTorch with CMake](./runtime-build-and-cross-compilation.md)
:::

:::{grid-item-card}  What you will learn in this tutorial:
:class-card: card-prerequisites
In this tutorial you will learn how to export a simple PyTorch model for ExecuTorch Arm Ethos-u backend delegate and run it on a Corstone-300 FVP Simulator.
:::

::::

```{warning}
This ExecuTorch backend delegate is under active development. You may encounter some rough edges and features which may be documented or planned but not implemented.
```

```{tip}
If you are already familiar with this delegate, you may want to jump directly to the examples source dir - [https://github.com/pytorch/executorch/tree/main/examples/arm](https://github.com/pytorch/executorch/tree/main/examples/arm)
```

## Prerequisites

Let's make sure you have everything you need before we get started.

### Hardware

To successfully complete this tutorial, you will need a Linux-based host machine with Arm aarch64 or x86_64 processor architecture.

The target device will be an embedded platform with an Arm Cortex-M55 CPU and Ethos-U55 NPU (ML processor). This tutorial will show you how to run PyTorch models on both.

We will be using a [Fixed Virtual Platform (FVP)](https://www.arm.com/products/development-tools/simulation/fixed-virtual-platforms), simulating a [Corstone-300](https://developer.arm.com/Processors/Corstone-300)(cs300) system. Since we will be using the FVP (think of it as virtual hardware), we won't be requiring any real embedded hardware for this tutorial.

### Software

First, you will need to install ExecuTorch. Please follow the recommended tutorials if you haven't already, to set up a working ExecuTorch development environment.

To generate software which can be run on an embedded platform (real or virtual), we will need a tool chain for cross-compilation and an Arm Ethos-U software development kit, including the Vela compiler for Ethos-U NPUs.

In the following sections we will walk through the steps to download each of the dependencies listed above.

## Set Up the Developer Environment

In this section, we will do a one-time setup, like downloading and installing necessary software, for the platform support files needed to run ExecuTorch programs in this tutorial. There are two approaches available:

1. Method 1: Use the `examples/arm/setup.sh` script to pull each item in an automated fashion (recommended). It is recommended to run the script in a conda environment. Upon successful execution, you can directly go to [the next step](#convert-the-pytorch-model-to-the-pte-file).
2. Method 2: Follow the guide step by step to understand all the components and the logic of the script. You may want to use this method if you intend to change the behavior of the flow significantly.

```{tip}
In the ExecuTorch repository we have a functioning script which follows the exact same steps to speed things up. It is located at `examples/arm/setup.sh`. Feel free to use that instead if it is convenient, or use it as a reference if some of the steps in the manual instruction aren't very clear.
```

As mentioned before, we currently support only Linux based platforms with x86_64 or aarch64 processor architecture. Let’s make sure we are indeed on a supported platform.

```bash
uname -s
# Linux

uname -m
# x86_64 or aarch64
```

Let's create an empty directory, and use this as a top level development directory.

### Download and Set Up the Corstone-300 FVP

Fixed Virtual Platforms (FVPs) are pre-configured, functionally accurate simulations of popular system configurations. Here in this tutorial, we are interested in the Corstone-300 system. We can download this from the Arm website.

```{note}
 By downloading and running the FVP software, you will be agreeing to the FVP [End-user license agreement (EULA)](https://developer.arm.com/downloads/-/arm-ecosystem-fvps/eula).
```

To download, we can either download `Corstone-300 Ecosystem FVP` from [here](https://developer.arm.com/downloads/-/arm-ecosystem-fvps). Alternatively, you can download the same version we tested with like this,

```bash
# for aarch64
curl \
    --output FVP_cs300.tgz \
    'https://developer.arm.com/-/media/Arm%20Developer%20Community/Downloads/OSS/FVP/Corstone-300/FVP_Corstone_SSE-300_11.22_35_Linux64_armv8l.tgz?rev=b083dc5ac9c546899fbb7ccd67b74c17&hash=BFE589289ECF12B07192636382C15C01'

# for x86_64
curl \
    --output FVP_cs300.tgz \
    'https://developer.arm.com/-/media/Arm%20Developer%20Community/Downloads/OSS/FVP/Corstone-300/FVP_Corstone_SSE-300_11.22_20_Linux64.tgz?rev=018659bd574f4e7b95fa647e7836ccf4&hash=22A79103C6FA5FFA7AFF3BE0447F3FF9'
```

Now, extract the `FVP_cs300.tgz` file in a new dir, and run the provided script which will install the FVP.

```bash
./FVP_Corstone_SSE-300.sh          \
   --i-agree-to-the-contained-eula \
   --force                         \
   --destination ./                \
   --quiet                         \
   --no-interactive
```

Once successful, let's make sure the FVP simulator is available on the PATH for later use.

```bash
# for x86-64 hosts
export PATH=${PATH}:<install_dir>/FVP/models/Linux64_GCC-9.3
# for aarch64 hosts
export PATH=${PATH}:<install_dir>/FVP/models/Linux64_armv8l_GCC-9.3/

hash FVP_Corstone_SSE-300_Ethos-U55 # To make sure we are ready to use
```

### Download and Install the Arm GNU AArch32 Bare-Metal Toolchain

Similar to the FVP, we would also need a tool-chain to cross-compile ExecuTorch runtime, executor-runner bare-metal application, as well as the rest of the bare-metal stack for Cortex-M55 CPU available on the Corstone-300 platform.

These toolchains are available [here](https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads). We will be using GCC 12.3 targeting `arm-none-eabi` here for our tutorial. Just like FVP, to download the same version as we tested with in the top-level development dir,

```bash
# for aarch64
curl \
    --output gcc.tar.xz \
    'https://armkeil.blob.core.windows.net/developer/Files/downloads/gnu/12.3.rel1/binrel/arm-gnu-toolchain-12.3.rel1-aarch64-arm-none-eabi.tar.xz'

# for x86_64
curl \
    --output gcc.tar.xz \
    'https://armkeil.blob.core.windows.net/developer/Files/downloads/gnu/12.3.rel1/binrel/arm-gnu-toolchain-12.3.rel1-x86_64-arm-none-eabi.tar.xz'
```

Once downloaded, you can extract its contents in a new dir. Then, let's make sure the toolchain is available on the PATH for later use.

```bash
export PATH=${PATH}:/<install_dir>/arm-gnu-toolchain-12.3.rel1-x86_64-arm-none-eabi/bin
export PATH=${PATH}:/<install_dir>/arm-gnu-toolchain-12.3.rel1-aarch64-arm-none-eabi/bin

hash arm-none-eabi-gcc # To make sure we are ready to use
```

### Setup the Arm Ethos-U Software Development

This git repository is the root directory for all Arm Ethos-U software. It is to help us download required repositories and place them in a tree structure. In the top-level devlopment dir,

```bash
# Download the repo
git clone https://review.mlplatform.org/ml/ethos-u/ethos-u
cd ethos-u

# To align with the version we have tested
git reset --hard 0995223100e3da8011700f58e491f1bf59511e3c

# Download the necessary repos and properly install them
./fetch_externals.py fetch

# Download the Vela compiler
cd .. # To the top-level development dir
git clone https://review.mlplatform.org/ml/ethos-u/ethos-u-vela
```

Once this is done, you should have a working FVP simulator, a functioning toolchain for cross compilation, and the Ethos-U software development setup ready for the bare-metal developement.

#### Applying Local Patches
Since this is under active development, we have some patches for the Arm Ethos-u software development kit. Let's apply them on the download SDK and the Vela compiler.

```bash
cd ethos-u # this is the top level Ethos-U software directory

# Let's patch core_platform repo
cd core_platform
git reset --hard 204210b1074071532627da9dc69950d058a809f4
git am -3 <path_to>/executorch/examples/arm/ethos-u-setup/core_platform/patches/*.patch
cd ../.. # To the top-level development dir
```

### Install the Vela Compiler
Once the patching is done, let's finish the setup by installing the Vela compiler.

```bash
cd ethos-u-vela
pip install .
```

### Install the TOSA reference model
```bash
git clone https://review.mlplatform.org/tosa/reference_model -b v0.80
cd reference_model
git submodule update --init --recursive
mkdir -p build
cd build
cmake ..
n=$(nproc)
make -j"$((n - 5))"
cd reference_model # Within the build directory
# Add tosa_reference_model to the path
export PATH=${PATH}:`pwd`
```

At the end of the setup, if everything goes well, your top level devlopement dir might look something like this,

```bash
.
├── arm-gnu-toolchain-12.3.rel1-x86_64-arm-none-eabi # for x86-64 hosts
├── ethos-u
│   ├── core_platform
│   ├── core_software
│   ├── fetch_externals.py
│   └── [...]
├── ethos-u-vela
├── FVP
│   ├── FVP_Corstone_SSE-300.sh
│   └── [...]
├── FVP_cs300.tgz
├── gcc.tar.xz
└── reference_model
```

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

Running it using the Python environment (on the same development Linux machine), we get the expected output.

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

Running it using the Python environment (on the same development Linux machine), and as expected 1 + 1 indeed produces 2.

```python
>>> m = AddModule()
>>> m(torch.ones(5, dtype=torch.int32)) # integer types for non-quantized Ethos-U delegation
tensor([2, 2, 2, 2, 2], dtype=torch.int32)
```
Keep the inputs and outputs to these modules in mind. When we will lower and run this through alternate means as opposed to running on this Linux machine, we will use the same inputs, and expect the outputs to match with the one shown here.

```{tip}
We need to be aware of data types for running networks on the Ethos-U55 as it is an integer only processor. For this example we use integer types explicitly, for typical use of such a flow networks are built and trained in floating point, and then are quantized from floating point to integer for efficient inference.
```

### Non-delegated Workflow

In the ExecuTorch AoT pipeline, one of the options is to select a backend. ExecuTorch offers a variety of different backends. Selecting backend is optional, it is typically done to target a particular mode of acceleration or hardware for a given model compute requirements. Without any backends, ExecuTorch runtime will fallback to using, available by default, a highly portable set of operators.

It's expected that on platforms with dedicated acceleration like the Ethos-U55, that the non-delegated flow is used for two primary cases:
1. When the network is designed to be very small and best suited to run on the Cortex-M alone.
2. When the network has a mix of operations that can target the NPU and those that can't, e.g. the Ethos-U55 supports integer operations and so floating point softmax will fall back to execute on the CPU.

In this flow, without any backend delegates, to illustrate the portability of the ExecuTorch runtime, as well as of the operator library we will skip specifying the backend during the `.pte` generation.

Following script will serve as a helper utility to help us generate the `.pte` file. This is available in the `examples/arm` directory.

```bash
python3 -m examples.arm.aot_arm_compiler --model_name="softmax"
# This should produce ./softmax.pte
```

### Delegated Workflow

Working with Arm, we introduced a new Arm backend delegate for ExecuTorch. This backend is under active development and has a limited set of features available as of writing this.

By including a following step during the ExecuTorch AoT export pipeline to generate the `.pte` file, we can enable this backend delegate.

```python
from executorch.backends.arm.arm_backend import generate_ethosu_compile_spec

graph_module_edge.exported_program = to_backend(
    model.exported_program,
    ArmPartitioner(generate_ethosu_compile_spec("ethos-u55-128")))
```

Similar to the non-delegate flow, the same script will server as a helper utility to help us generate the `.pte` file. Notice the `--delegate` option to enable the `to_backend` call.

```bash
python3 -m examples.arm.aot_arm_compiler --model_name="add" --delegate
# should produce ./add_arm_delegate.pte
```

At the end of this, we should have two different `.pte` files. First one with the [SoftmaxModule](#softmaxmodule), without any backend delegates. And the second one with the [AddModule](#addmodule), and with Arm Ethos-U backend delegate enabled. Now let's try to run these `.pte` files on a Corstone-300 platform in a bare-metal environment.

## Getting a Bare-Metal Executable

In this section, we will go over steps that you need to go through to build the runtime application. This then run on the target device.

```{tip}
In the executorch repository we have a functioning script which does the exact same steps. It is located at `executorch/examples/arm/run.sh`. Feel free to use that instead if it is convenient, or use it as a reference if some of the steps in the manual instruction aren't very clear.
```

Also before we get started, make sure that you have completed ExecuTorch cmake build setup, and the instructions to setup the development environment described [earlier](#set-up-the-developer-environment).

The block diagram below demonstrates, at the high level, how the various build artifacts are generated and are linked together to generate the final bare-metal executable.

![](./arm-delegate-runtime-build.svg)

### Generating ExecuTorch Libraries

ExecuTorch's CMake build system produces a set of build pieces which are critical for us to include and run the ExecuTorch runtime with-in the bare-metal environment we have for Corstone-300 from Ethos-U SDK.

[This](./runtime-build-and-cross-compilation.md) document provides a detailed overview of each individual build piece. For running either variant of the `.pte` file, we will need a core set of libraries. Here is a list,

- `libexecutorch.a`
- `libportable_kernels.a`
- `libportable_ops_lib.a`

To run a `.pte` file with the Arm backend delegate call instructions, we will need the Arm backend delegate runtime library, that is,

- `libexecutorch_delegate_ethos_u.a`


To generate these libraries, use following commands,

```bash
# Empty and already created
cd <executorch_source_root_dir>

# Use provided cmake toolchain for bare-metal builds
toolchain_cmake=<executorch_source_root_dir>/examples/arm/ethos-u-setup/arm-none-eabi-gcc.cmake

cmake                                                 \
    -DBUCK2=${buck2}                                  \
    -DCMAKE_INSTALL_PREFIX=<executorch_build_dir>     \
    -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF            \
    -DCMAKE_BUILD_TYPE=Release                        \
    -DEXECUTORCH_ENABLE_LOGGING=ON                    \
    -DEXECUTORCH_BUILD_ARM_BAREMETAL=ON               \
    -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON       \
    -DFLATC_EXECUTABLE="$(which flatc)"               \
    -DCMAKE_TOOLCHAIN_FILE="${toolchain_cmake}"       \
    -B<executorch_build_dir>                          \
    <executorch_source_root_dir>

cmake --build <executorch_build_dir> --target install --config Release

cmake                                                 \
    -DCMAKE_INSTALL_PREFIX=<executorch_build_dir>     \
    -DCMAKE_BUILD_TYPE=Release                        \
    -DEXECUTORCH_SELECT_OPS_LIST="aten::_softmax.out" \
    -DCMAKE_TOOLCHAIN_FILE="${toolchain_cmake}"       \
    -B<executorch_build_dir>/examples/arm             \
    <executorch_source_root_dir>/examples/arm

cmake --build <executorch_build_dir>/examples/arm --config Release

```

`EXECUTORCH_SELECT_OPS_LIST` will decide the number of portable operators included in the build and are available at runtime. It must match with `.pte` file's requirements, otherwise you will get `Missing Operator` error at runtime.

For example, here in the command line above, to run SoftmaxModule, we only included the softmax CPU operator. Similarly, to run AddModule in a non-delegated manner you will need add op and so on. As you might have already realized, for the delegated operators, which will be executed by the Arm backend delegate, we do not need to include those operators in this list. This is only for *non-delegated* operators.

### Building the executor_runner Bare-Metal Application

The SDK dir is the same one prepared [earlier](#setup-the-arm-ethos-u-software-development). And, we will be passing the `.pte` file (any one of them) generated above.

Note, you have to generate a new `executor-runner` binary if you want to change the model or the `.pte` file. This constraint is from the constrained bare-metal runtime environment we have for Corstone-300 platform.

```bash

cd <executorch_source_root_dir>
cd examples/arm/executor_runner

cmake                                                    \
    -DCMAKE_TOOLCHAIN_FILE="${toolchain_cmake}"          \
    -DTARGET_CPU=cortex-m55                              \
    -B build                                             \
    -DETHOS_SDK_PATH:PATH=<ethos-u_clone_directory>      \
    -DET_DIR_PATH:PATH=<executorch_source_root_dir>      \
    -DET_BUILD_DIR_PATH:PATH=<executorch_build_dir>      \
    -DET_PTE_FILE_PATH:PATH=<path_to_pte_file_of_choice> \
    -DPYTHON_EXECUTABLE=$(which python3)

cmake --build build -- arm_executor_runner
```

## Running on Corstone-300 FVP Platform

Once the elf is prepared, regardless of the `.pte` file variant is used to generate the bare metal elf, you can run in with following command,

```bash
ethos_u_build_dir=examples/arm/executor_runner/

elf=$(find ${ethos_u_build_dir} -name "arm_executor_runner")

FVP_Corstone_SSE-300_Ethos-U55                          \
    -C ethosu.num_macs=128                              \
    -C mps3_board.visualisation.disable-visualisation=1 \
    -C mps3_board.telnetterminal0.start_telnet=0        \
    -C mps3_board.uart0.out_file='-'                    \
    -a "${elf}"                                         \
    --timelimit 10 # seconds - after which sim will kill itself
```

If successful, the simulator should produce something like the following on the shell,

```console
    Ethos-U rev 136b7d75 --- Apr 12 2023 13:44:01
    (C) COPYRIGHT 2019-2023 Arm Limited
    ALL RIGHTS RESERVED

I executorch:runner.cpp:64] Model PTE file loaded. Size: 960 bytes.
I executorch:runner.cpp:70] Model buffer loaded, has 1 methods
I executorch:runner.cpp:78] Running method forward
I executorch:runner.cpp:95] Setting up planned buffer 0, size 32.
I executorch:runner.cpp:110] Method loaded.
I executorch:runner.cpp:112] Preparing inputs...
I executorch:runner.cpp:114] Input prepared.
I executorch:runner.cpp:116] Starting the model execution...
I executorch:runner.cpp:121] Model executed successfully.
I executorch:runner.cpp:125] 1 outputs:
Output[0][0]: 0.500000
Output[0][1]: 0.500000
Output[0][2]: 0.500000
Output[0][3]: 0.500000
Application exit code: 0.

EXITTHESIM

Info: Simulation is stopping. Reason: CPU time has been exceeded.
```

Here in this example, we ran the `executor_runner` binary with the `softmax.pte` file generated for the [SoftmaxModule](#softmaxmodule), we do see the expected results generated from the baremetal binary running on the Corstone-300 virtual hardware on FVP simulator.

If you rerun the same FVP command with the delegated `.pte` file for the [AddModule](#addmodule), i.e. `add_arm_delegate.pte` - you may get something like following, again the expected results. Pay attention to the messages printed with prefix `ArmBackend::`, they indicate that the backend was sucecssfully initialized and the `add` operator from our AddModule in the `.pte` was exexuted on the Ethos-U55 NPU.

```console
    Ethos-U rev 136b7d75 --- Apr 12 2023 13:44:01
    (C) COPYRIGHT 2019-2023 Arm Limited
    ALL RIGHTS RESERVED

I executorch:runner.cpp:64] Model PTE file loaded. Size: 2208 bytes.
I executorch:runner.cpp:70] Model buffer loaded, has 1 methods
I executorch:runner.cpp:78] Running method forward
I executorch:runner.cpp:95] Setting up planned buffer 0, size 64.
I executorch:ArmBackendEthosU.cpp:51] ArmBackend::init 0x11000050
I executorch:runner.cpp:110] Method loaded.
I executorch:runner.cpp:112] Preparing inputs...
I executorch:runner.cpp:114] Input prepared.
I executorch:runner.cpp:116] Starting the model execution...
I executorch:ArmBackendEthosU.cpp:103] ArmBackend::execute 0x11000050
I executorch:runner.cpp:121] Model executed successfully.
I executorch:runner.cpp:125] 1 outputs:
Output[0][0]: 2
Output[0][1]: 2
Output[0][2]: 2
Output[0][3]: 2
Output[0][4]: 2
Application exit code: 0.

EXITTHESIM

Info: Simulation is stopping. Reason: CPU time has been exceeded.
```

## Takeaways
Through this tutorial we've learnt how to use the ExecuTorch software to both export a standard model from PyTorch and to run it on the compact and fully functioned ExecuTorch runtime, enabling a smooth path for offloading models from PyTorch to Arm based platforms.

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
