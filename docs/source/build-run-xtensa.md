# Building and Running ExecuTorch on Xtensa HiFi4 DSP


In this tutorial we will walk you through the process of getting setup to build ExecuTorch for an Xtensa HiFi4 DSP and running a simple model on it.

[Cadence](https://www.cadence.com/en_US/home.html) is both a hardware and software vendor, providing solutions for many computational workloads, including to run on power-limited embedded devices. The [Xtensa HiFi4 DSP](https://www.cadence.com/en_US/home/tools/ip/tensilica-ip/hifi-dsps/hifi-4.html) is a Digital Signal Processor (DSP) that is optimized for running audio based neural networks such as wake word detection, Automatic Speech Recognition (ASR), etc.

In addition to the chip, the HiFi4 Neural Network Library ([nnlib](https://github.com/foss-xtensa/nnlib-hifi4)) offers an optimized set of library functions commonly used in NN processing that we utilize in this example to demonstrate how common operations can be accelerated.

On top of being able to run on the Xtensa HiFi4 DSP, another goal of this tutorial is to demonstrate how portable ExecuTorch is and its ability to run on a low-power embedded device such as the Xtensa HiFi4 DSP. This workflow does not require any delegates, it uses custom operators and compiler passes to enhance the model and make it more suitable to running on Xtensa HiFi4 DSPs. A custom [quantizer](https://pytorch.org/tutorials/prototype/quantization_in_pytorch_2_0_export_tutorial.html) is used to represent activations and weights as `uint8` instead of `float`, and call appropriate operators. Finally, custom kernels optimized with Xtensa intrinsics provide runtime acceleration.

::::{grid} 2
:::{grid-item-card}  What you will learn in this tutorial:
:class-card: card-prerequisites
* In this tutorial you will learn how to export a quantized model with a linear operation targeted for the Xtensa HiFi4 DSP.
* You will also learn how to compile and deploy the ExecuTorch runtime with the kernels required for running the quantized model generated in the previous step on the Xtensa HiFi4 DSP.
:::
:::{grid-item-card}  Tutorials we recommend you complete before this:
:class-card: card-prerequisites
* [Introduction to ExecuTorch](intro-how-it-works.md)
* [Setting up ExecuTorch](getting-started-setup.md)
* [Building ExecuTorch with CMake](runtime-build-and-cross-compilation.md)
:::
::::

```{note}
The linux part of this tutorial has been designed and tested on Ubuntu 22.04 LTS, and requires glibc 2.34. Workarounds are available for other distributions, but will not be covered in this tutorial.
```

## Prerequisites (Hardware and Software)

In order to be able to succesfully build and run ExecuTorch on a Xtensa HiFi4 DSP you'll need the following hardware and software components.

### Hardware
 - [i.MX RT600 Evaluation Kit](https://www.nxp.com/design/development-boards/i-mx-evaluation-and-development-boards/i-mx-rt600-evaluation-kit:MIMXRT685-EVK)

### Software
 - x86-64 Linux system (For compiling the DSP binaries)
 - [MCUXpresso IDE](https://www.nxp.com/design/software/development-software/mcuxpresso-software-and-tools-/mcuxpresso-integrated-development-environment-ide:MCUXpresso-IDE)
    - This IDE is supported on multiple platforms including MacOS. You can use it on any of the supported platforms as you'll only be using this to flash the board with the DSP images that you'll be building later on in this tutorial.
- [J-Link](https://www.segger.com/downloads/jlink/)
    - Needed to flash the board with the firmware images. You can install this on the same platform that you installed the MCUXpresso IDE on.
    - Note: depending on the version of the NXP board, another probe than JLink might be installed. In any case, flashing is done using the MCUXpresso IDE in a similar way.
 - [MCUXpresso SDK](https://mcuxpresso.nxp.com/en/select?device=EVK-MIMXRT685)
    - Download this SDK to your Linux machine, extract it and take a note of the path where you store it. You'll need this later.
- [Xtensa compiler](https://tensilicatools.com/platform/i-mx-rt600/)
    - Download this to your Linux machine. This is needed to build ExecuTorch for the HiFi4 DSP.
- For cases with optimized kernels, the [nnlib repo](https://github.com/foss-xtensa/nnlib-hifi4).

## Setting up Developer Environment

Step 1. In order to be able to successfully install all the software components specified above users will need to go through the NXP tutorial linked below. Although the tutorial itself walks through a Windows setup, most of the steps translate over to a Linux installation too.

[NXP tutorial on setting up the board and dev environment](https://www.nxp.com/document/guide/getting-started-with-i-mx-rt600-evaluation-kit:GS-MIMXRT685-EVK?section=plug-it-in)

```{note}
Before proceeding forward to the next section users should be able to succesfullly flash the **dsp_mu_polling_cm33** sample application from the tutorial above and notice output on the UART console indicating that the Cortex-M33 and HiFi4 DSP are talking to each other.
```

Step 2. Make sure you have completed the ExecuTorch setup tutorials linked to at the top of this page.

## Working Tree Description

The working tree is:

```
executorch
├── backends
│   └── cadence
│       ├── aot
│       ├── ops_registration
│       ├── tests
│       ├── utils
│       ├── hifi
│       │   ├── kernels
│       │   ├── operators
│       │   └── third-party
│       │       └── hifi4-nnlib
│       └── [other cadence DSP families]
│           ├── kernels
│           ├── operators
│           └── third-party
│               └── [any required lib]
└── examples
    └── cadence
        ├── models
        └── operators
```

***AoT (Ahead-of-Time) Components***:

The AoT folder contains all of the python scripts and functions needed to export the model to an ExecuTorch `.pte` file. In our case, [export_example.py](https://github.com/pytorch/executorch/blob/main/backends/cadence/aot/export_example.py) is an API that takes a model (nn.Module) and representative inputs and runs it through the quantizer (from [quantizer.py](https://github.com/pytorch/executorch/blob/main/backends/cadence/aot/quantizer.py)). Then a few compiler passes, also defined in [quantizer.py](https://github.com/pytorch/executorch/blob/main/backends/cadence/aot/quantizer.py), will replace operators with custom ones that are supported and optimized on the chip. Any operator needed to compute things should be defined in [ops_registrations.py](https://github.com/pytorch/executorch/blob/main/backends/cadence/aot/ops_registrations.py) and have corresponding implemetations in the other folders.

***Operators***:

The operators folder contains two kinds of operators: existing operators from the [ExecuTorch portable library](https://github.com/pytorch/executorch/tree/main/kernels/portable/cpu) and new operators that define custom computations. The former is simply dispatching the operator to the relevant ExecuTorch implementation, while the latter acts as an interface, setting up everything needed for the custom kernels to compute the outputs.

***Kernels***:

The kernels folder contains the optimized kernels that will run on the HiFi4 chip. They use Xtensa intrinsics to deliver high performance at low-power.

## Build

In this step, you will generate the ExecuTorch program from different models. You'll then use this Program (the `.pte` file) during the runtime build step to bake this Program into the DSP image.

***Simple Model***:

The first, simple model is meant to test that all components of this tutorial are working properly, and simply does an add operation. The generated file is called `add.pte`.

```bash
cd executorch
python3 -m examples.portable.scripts.export --model_name="add"
```

***Quantized Operators***:

The other, more complex model are custom operators, including:
  - a quantized [linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) operation. The model is defined [here](https://github.com/pytorch/executorch/blob/main/examples/cadence/operators/quantized_linear_op.py#L28). Linear is the backbone of most Automatic Speech Recognition (ASR) models.
  - a quantized [conv1d](https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html) operation. The model is defined [here](https://github.com/pytorch/executorch/blob/main/examples/cadence/operators/quantized_conv1d_op.py#L36). Convolutions are important in wake word and many denoising models.

In both cases the generated file is called `CadenceDemoModel.pte`.

```bash
cd executorch
python3 -m examples.cadence.operators.quantized_<linear,conv1d>_op
```

***Small Model: RNNT predictor***:

The torchaudio [RNNT-emformer](https://pytorch.org/audio/stable/tutorials/online_asr_tutorial.html) model is an Automatic Speech Recognition (ASR) model, comprised of three different submodels: an encoder, a predictor and a joiner.
The [predictor](https://github.com/pytorch/executorch/blob/main/examples/cadence/models/rnnt_predictor.py) is a sequence of basic ops (embedding, ReLU, linear, layer norm) and can be exported using:

```bash
cd executorch
python3 -m examples.cadence.models.rnnt_predictor
```

The generated file is called `CadenceDemoModel.pte`.

### Runtime

**Building the DSP firmware image**
In this step, you'll be building the DSP firmware image that consists of the sample ExecuTorch runner along with the Program generated from the previous step. This image when loaded onto the DSP will run through the model that this Program consists of.

***Step 1***. Configure the environment variables needed to point to the Xtensa toolchain that you have installed in the previous step. The three environment variables that need to be set include:
```bash
# Directory in which the Xtensa toolchain was installed
export XTENSA_TOOLCHAIN=/home/user_name/cadence/XtDevTools/install/tools
# The version of the toolchain that was installed. This is essentially the name of the directory
# that is present in the XTENSA_TOOLCHAIN directory from above.
export TOOLCHAIN_VER=RI-2021.8-linux
# The Xtensa core that you're targeting.
export XTENSA_CORE=nxp_rt600_RI2021_8_newlib
```

***Step 2***. Clone the [nnlib repo](https://github.com/foss-xtensa/nnlib-hifi4), which contains optimized kernels and primitives for HiFi4 DSPs, with `git clone git@github.com:foss-xtensa/nnlib-hifi4.git`.

***Step 3***. Run the CMake build.
In order to run the CMake build, you need the path to the following:
- The Program generated in the previous step
- Path to the NXP SDK root. This should have been installed already in the [Setting up Developer Environment](#setting-up-developer-environment) section. This is the directory that contains the folders such as boards, components, devices, and other.

```bash
cd executorch
rm -rf cmake-out
# prebuild and install executorch library
cmake -DCMAKE_TOOLCHAIN_FILE=<path_to_executorch>/backends/cadence/cadence.cmake \
    -DCMAKE_INSTALL_PREFIX=cmake-out \
    -DCMAKE_BUILD_TYPE=Debug \
    -DPYTHON_EXECUTABLE=python3 \
    -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON \
    -DEXECUTORCH_BUILD_HOST_TARGETS=ON \
    -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF \
    -DEXECUTORCH_BUILD_PTHREADPOOL=OFF \
    -DEXECUTORCH_BUILD_CPUINFO=OFF \
    -DEXECUTORCH_BUILD_FLATC=OFF \
    -DFLATC_EXECUTABLE="$(which flatc)" \
    -Bcmake-out .

cmake --build cmake-out -j<num_cores> --target install --config Debug
# build cadence runner
cmake -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_TOOLCHAIN_FILE=<path_to_executorch>/examples/backends/cadence.cmake \
    -DCMAKE_PREFIX_PATH=<path_to_executorch>/cmake-out \
    -DMODEL_PATH=<path_to_program_file_generated_in_previous_step> \
    -DNXP_SDK_ROOT_DIR=<path_to_nxp_sdk_root> -DEXECUTORCH_BUILD_FLATC=0 \
    -DFLATC_EXECUTABLE="$(which flatc)" \
    -DNN_LIB_BASE_DIR=<path_to_nnlib_cloned_in_step_2> \
    -Bcmake-out/examples/cadence \
    examples/cadence

cmake --build cmake-out/examples/cadence -j8 -t cadence_executorch_example
```

After having succesfully run the above step you should see two binary files in their CMake output directory.
```bash
> ls cmake-xt/*.bin
cmake-xt/dsp_data_release.bin  cmake-xt/dsp_text_release.bin
```

## Deploying and Running on Device

***Step 1***. You now take the DSP binary images generated from the previous step and copy them over into your NXP workspace created in the [Setting up  Developer Environment](#setting-up-developer-environment) section. Copy the DSP images into the `dsp_binary` section highlighted in the image below.

<img src="_static/img/dsp_binary.png" alt="MCUXpresso IDE" /><br>

```{note}
As long as binaries have been built using the Xtensa toolchain on Linux, flashing the board and running on the chip can be done only with the MCUXpresso IDE, which is available on all platforms (Linux, MacOS, Windows).
```

***Step 2***. Clean your work space

***Step 3***. Click **Debug your Project** which will flash the board with your binaries.

On the UART console connected to your board (at a default baud rate of 115200), you should see an output similar to this:

```bash
> screen /dev/tty.usbmodem0007288234991 115200
Executed model
Model executed successfully.
First 20 elements of output 0
0.165528   0.331055 ...
```

## Conclusion and Future Work

In this tutorial, you have learned how to export a quantized operation, build the ExecuTorch runtime and run this model on the Xtensa HiFi4 DSP chip.

The (quantized linear) model in this tutorial is a typical operation appearing in ASR models, and can be extended to a complete ASR model by creating the model as a new test and adding the needed operators/kernels to [operators](https://github.com/pytorch/executorch/blob/main/backends/cadence/hifi/operators) and [kernels](https://github.com/pytorch/executorch/blob/main/backends/cadence/hifi/kernels).

Other models can be created following the same structure, always assuming that operators and kernels are available.
