# Getting started with eIQ Neutron NPU ExecuTorch backend

## Prerequisities

### Hardware
For this tutorial, you will need a Linux machine with x86_64 processor architecture. 
This tutorial demonstrates the use of the eIQ Neutron backend with the Neutron behavioral simulator, called NSYS. You don't need any specific development board for this tutorial. 

### Software
First you need to have Python 3.10 - 3.12 installed. 

You need to install the ExecuTorch. Please follow the tutorial to install the ExecuTorch [Setting Up ExecuTorch](../../../getting-started-setup.rst)


In addition to this, you will need to install the eIQ Neutron Simulator, called NSYS,
and the Neutron Converter for generating the byte-code for the eIQ Neutron NPU, 
during the model conversion in ExecuTorch AoT flow. 
To install the eIQ Neutron dependencies, run:
```bash
examples/nxp/setup.sh
```
This will install: 
* Neutron Converter, for converting the Neutron IR to Neutron byte-code 
* eIQ Neutron SDK, containing the eIQ Neutron runtimes (driver and firmware) for various NXP SoC and simulator 
* eIQ NSYS, the Neutron behavioral simulator

## Preparing a Model for NXP eIQ Neutron Backend

This guide demonstrating the use of ExecuTorch AoT flow to convert a PyTorch model to ExecuTorch
format and delegate the model computation to eIQ Neutron NPU using the eIQ Neutron Backend.

### Step 1: Environment Setup

This tutorial is intended to be run from a Linux and uses Conda or Virtual Env for Python environment management. For full setup details and system requirements, see [Getting Started with ExecuTorch](/getting-started).

Create a Conda environment and install the ExecuTorch Python package.
```bash
conda create -y --name executorch python=3.12
conda activate executorch
```
and install the ExecuTorch Python package, either a prebuilt one: 
```bash
conda install executorch
```
or build from source: 
```bash
./install_executorch.sh
```

Also run the `setup.sh` script to install the eIQ Neutron dependencies:
```bash
$ ./examples/nxp/setup.sh
```

### Step 2: Model Preparation and Running the Model on Target

See the example `aot_neutron_compile.py` and its [README](https://github.com/pytorch/executorch/blob/main/examples/nxp/README.md) file.

For the purpose of this tutorial we will use a simple image classification model CifarNet10.
```bash
python -m examples.nxp.aot_neutron_compile --quantize \
    --delegate --neutron_converter_flavor SDK_25_12 -m "cifar10"
```

Also, we will dump few of the images from the Cifar 10 dataset to a folder: 
```bash
python -m examples.nxp.experimental.cifar_net.cifar_net --store-test-data
```
The destination folder is `./cifar10_test_data`.

## Runtime with NSYS
### nxp_executor_runner example
The end-to-end example illustrating the use of the eIQ Neutron NPU is located in `examples/nxp/executor_runner`. 
Before we proceed with the build and execution, let's stop to briefly introduce the example. 

The application runs the executorch on a particular model. 
If a delegated model is provided to the runner, as in this tutorial, the instruction `DelegateCall` having the `NeutronBackend` ID is present in the model. 
This instruction is recognized by ExecuTorch runtime and handed over to the NeutronBackend.
NeutronBackend runs it on the eIQ Neutron NPU, simulated by NSYS. 

The NeutronDriver's API (`NeutronDriver.h` and `NeutronErrors.h`) is the same, regardless of whether it is a physical IP or NSYS. 
What is specific, is the API provided by the `NeutronEnvConfig.h`, to set up the paths to:
* the NSYS,
* the Neutron Firmware to run and
* the NSYS configuration (.ini file): 
```c++
    storeNsysConfigPath(FLAGS_nsys_config.c_str());
    storeFirmwarePath(FLAGS_firmware.c_str());
    storeNsysPath(FLAGS_nsys.c_str());
```

What corresponds to nxp_executor_runner CLI options, `--firmware`, `--nsys`, `--nsys_config` or provided by environmental variable:

| CLI Option      | Environmental variable | Description | 
|-----------------|------------------------|-------------|
| `--firmware`    | NSYS_FIRMWARE_PATH     | Path to the NSYS firmware. The NSYS firmware for each supported platform is provided by the eIQ Neutron SDK package. |
| `--nsys_config` | NSYS_CONFIG_PATH       | Path to the NSYS configuration. For i.MXRT700 SoC available in `examples/nxp/exector_runner/neutron-imxrt700.ini`. For other platform not required. 
| `--nsys` | N/A | Path to the eIQ NSYS simulator executable (`which nsys`) | 

For further details about `eiq_nsys` refer for the provided user guide in the python package. 

### Build the nxp_executor_runner
Use the provided `examples/nxp/executor_runner/CMakeLists.txt`. It can be build separately: 
```bash
mkdir ./examples/nxp/executor_runner/build 
pushd ./examples/nxp/executor_runner/build
cmake ..
make nxp_executor_runner
popd
```
or as part of the ExecuTorch build:
```bash
cd <executorch_root_dir>
mkdir build
cmake .. \
  -DEXECUTORCH_BUILD_NXP_NEUTRON=ON \
  -DEXECUTORCH_BUILD_NXP_NEUTRON_RUNNER=ON \
  -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
  -DEXECUTORCH_BUILD_KERNELS_QUANTIZED_AOT=ON \
make nxp_executorch_runner
```

### Run the nxp_executor_runner

```bash
./examples/nxp/executor_runner/build/nxp_executor_runner \
    --firmware `make -C ./examples/nxp/executor_runner/build locate_neutron_firmware | grep "NeutronFirmware.elf" ` \
    --nsys `which nsys` \
    --nsys_config ./examples/nxp/executor_runner/neutron-imxrt700.ini \
    --model ./cifar10_nxp_delegate.pte \
    --dataset ./cifar10_test_data \
    --output ./cifar10_test_output
```

## Takeways 
For the convenience, you can run the provided utility script doing all the above-mentioned steps for detailed inspection: 
```bash
cd <executorch-root>
./examples/nxp/run.sh
```

## FAQs
If you encounter any bugs or issues following this tutorial please file a bug/issue here on [Github](https://github.com/pytorch/executorch/issues/new) and label as `module: nxp`.  
