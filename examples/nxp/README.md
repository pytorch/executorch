# ExecuTorch Neutron Backend examples
This directory contains examples demonstrating the use of ExecuTorch AoT flow to convert a PyTorch model to ExecuTorch
format and delegate the model computation to eIQ Neutron NPU using the eIQ Neutron Backend.

## Layout
* `experimental/` - contains CifarNet model example.
* `models` - demo models instantiation used in examples.
* `aot_neutron_compile.py` - script with end-to-end ExecuTorch AoT Neutron Backend workflow.
* `README.md` - this file.
* `run_aot_example.sh` - utility script to launch _aot_neutron_compile.py_. Primarily for CI purpose. 
* `setup.sh` - setup script to install Neutron Backend dependencies.

## Setup
Please finish tutorial [Setting up ExecuTorch](https://pytorch.org/executorch/main/getting-started-setup).

Run the setup.sh script to install the neutron-converter:
```commandline
$ ./examples/nxp/setup.sh
```

## Supported models
* CifarNet
* MobileNetV2

## PyTorch Model Delegation to Neutron Backend
First we will start with an example script converting the model. This example show the CifarNet model preparation. 
It is the same model which is part of the `example_cifarnet` in 
[MCUXpresso SDK](https://www.nxp.com/design/design-center/software/development-software/mcuxpresso-software-and-tools-/mcuxpresso-software-development-kit-sdk:MCUXpresso-SDK).

The NXP MCUXpresso software and tools offer comprehensive development solutions designed to help accelerate embedded 
system development of applications based on MCUs from NXP. The MCUXpresso SDK includes a flexible set of peripheral 
drivers designed to speed up and simplify development of embedded applications.

The steps are expected to be executed from the `executorch` root folder.

1. Run the `aot_neutron_compile.py` example with the `cifar10` model 
    ```commandline
    $ python -m examples.nxp.aot_neutron_compile --quantize \
        --delegate --neutron_converter_flavor SDK_25_09 -m cifar10 
    ```

2. It will generate you `cifar10_nxp_delegate.pte` file which can be used with the MCUXpresso SDK `cifarnet_example` 
project, presented [here](https://mcuxpresso.nxp.com/mcuxsdk/latest/html/middleware/eiq/executorch/docs/nxp/topics/example_applications.html#how-to-build-and-run-executorch-cifarnet-example).
This project will guide you through the process of deploying your PTE model to the device.
To get the MCUXpresso SDK follow this [guide](https://mcuxpresso.nxp.com/mcuxsdk/latest/html/middleware/eiq/executorch/docs/nxp/topics/getting_mcuxpresso.html),
use the MCUXpresso SDK v25.09.00. 
