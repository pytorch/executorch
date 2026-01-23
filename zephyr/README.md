# Executorch

**ExecuTorch** is PyTorch's unified solution for deploying AI models on-device—from smartphones to microcontrollers—built for privacy, performance, and portability. It powers Meta's on-device AI across **Instagram, WhatsApp, Quest 3, Ray-Ban Meta Smart Glasses**, and [more](https://docs.pytorch.org/executorch/main/success-stories.html).

This folder adds ExecuTorch so it can be build and run in the Zephyr project as a external module. This includes an example under examples/arm/zephyr of running executor runners with Arm&reg; Ethos&trade;-U backend on a Corstone&trade; FVP, targeting the Zephyr RTOS.

# Requirements

Setup a venv (or conda)
```
mkdir <zephyr_build_root>
cd <zephyr_build_root>
python3 -m venv .zephyr_venv
source .zephyr_venv/bin/activate
```

Install requirements
```
pip install west cmake==3.29 pyelftools ninja jsonschema
```

Setup zephyr repo
```
west init --manifest-rev v4.3.0
```

Install Zephyr SDK according to Zephyr's guides and set ZEPHYR_SDK_INSTALL_DIR

```
export ZEPHYR_SDK_INSTALL_DIR=<PATH-TO-ZEPHYR-SDK>
```

# Usage with Zephyr

To pull in ExecuTorch as a Zephyr module, either add it as a West project in the west.yaml file or pull it in by adding a submanifest (e.g. zephyr/submanifests/executorch.yaml)

# Create executorch.yaml under zephyr/submanifests

There is an example executorch.yaml in this folder you can copy or just create a new file <zephyr_build_root>/zephyr/submanifests/executorch.yaml with the following content:

<zephyr_build_root>/zephyr/submanifests/executorch.yaml
```
manifest:
  projects:
    - name: executorch
      url: https://github.com/pytorch/executorch
      revision: main
      path: modules/lib/executorch
```

## Run west config and update:

Add ExecuTorch to Zephyr
```
west config manifest.project-filter -- -.*,+zephyr,+executorch,+cmsis,+cmsis_6,+cmsis-nn,+hal_ethos_u
west update
```

## Setup and install ExecuTorch

Setup ExecuTorch
```
cd modules/lib/executorch/
git submodule sync
git submodule update --init --recursive
./install_executorch.sh
cd ../../..
```

## Prepare Ethos&trade;-U tools like Vela compiler and Corstone&trade; 300/320 FVP

This is needed to convert python models to PTE files for Ethos&trade;-Ux5 and also installs Corstone&trade; 300/320 FVP so you can run and test.

Make sure to read and agree to the Corstone&trade; eula

Install TOSA, vela and FVPs
```
modules/lib/executorch/examples/arm/setup.sh --i-agree-to-the-contained-eula
. modules/lib/executorch/examples/arm/arm-scratch/setup_path.sh
```

# Running a sample application

To run you need to point of the path to the installed Corstone&trade; FVP and you can then use west to build and run. You point out the model PTE file you want to run with -DET_PTE_FILE_PATH= see below.


## Corstone&trade; 300 FVP (Ethos&trade;-U55)

### Setup FVP paths

Set up FVP paths and macs used, this will also set shutdown_on_eot so the FVP auto stops after it has run the example.

Config Zephyr Corstone300 FVP
```
export FVP_ROOT=$PWD/modules/lib/executorch/examples/arm/arm-scratch/FVP-corstone300
export ARMFVP_BIN_PATH=${FVP_ROOT}/models/Linux64_GCC-9.3
export ARMFVP_EXTRA_FLAGS="-C mps3_board.uart0.shutdown_on_eot=1 -C ethosu.num_macs=128"
```

### Ethos-U55

#### Prepare a PTE model file

Prepare the Ethos-U55 PTE model
```
python -m modules.lib.executorch.examples.arm.aot_arm_compiler --model_name=modules/lib/executorch/examples/arm/example_modules/add.py --quantize --delegate -t ethos-u55-128 --output=add_u55_128.pte
```

`--delegate` tells the aot_arm_compiler to use Ethos-U backend and `-t ethos-u55-128` specify the used Ethos-U variant and numbers of macs used, this must match you hardware or FVP config.

#### Build and run

Run the Ethos-U55 PTE model
```
west build -b mps3/corstone300/fvp modules/lib/executorch/examples/arm/zephyr -t run -- -DET_PTE_FILE_PATH=add_u55_128.pte
```

### Cortex-M55

#### Prepare a PTE model file

Prepare the Cortex-M55 PTE model
```
python -m modules.lib.executorch.examples.arm.aot_arm_compiler --model_name=modules/lib/executorch/examples/arm/example_modules/add.py --quantize --output=add_m55.pte
```

#### Build and run

Run the Cortex-M55 PTE model
```
west build -b mps3/corstone300/fvp modules/lib/executorch/examples/arm/zephyr -t run -- -DET_PTE_FILE_PATH=add_m55.pte
```

## Corstone&trade; 320 FVP (Ethos&trade;-U85)

### Setup FVP paths

Set up FVP paths, libs and macs used, this will also set shutdown_on_eot so the FVP auto stops after it has run the example.

Config Zephyr Corstone320 FVP
```
export FVP_ROOT=$PWD/modules/lib/executorch/examples/arm/arm-scratch/FVP-corstone320
export LD_LIBRARY_PATH=${FVP_ROOT}/python/lib:${ARMFVP_BIN_PATH}:${LD_LIBRARY_PATH}
export ARMFVP_BIN_PATH=${FVP_ROOT}/models/Linux64_GCC-9.3
export ARMFVP_EXTRA_FLAGS="-C mps4_board.uart0.shutdown_on_eot=1 -C mps4_board.subsystem.ethosu.num_macs=256"
```

### Ethos-U85

#### Prepare a PTE model file

Prepare the Ethos-U85 PTE model
```
python -m modules.lib.executorch.examples.arm.aot_arm_compiler --model_name=modules/lib/executorch/examples/arm/example_modules/add.py --quantize --delegate -t ethos-u85-256 --output=add_u85_256.pte
```

`--delegate` tells the aot_arm_compiler to use Ethos-U backend and `-t ethos-u85-256` specify the used Ethos-U variant and numbers of macs used, this must match you hardware or FVP config.

#### Build and run

Run the Ethos-U85 PTE model
```
west build -b mps4/corstone320/fvp modules/lib/executorch/examples/arm/zephyr -t run -- -DET_PTE_FILE_PATH=add_u85_256.pte
```

## Notable files

# executorch.yaml

Copy this to <zephyr_build_root>/zephyr/submanifests/

# module.yml

Do not remove this file. As mentioned in the official Zephyr [documenation](https://docs.zephyrproject.org/latest/develop/modules.html), for Executorch to be built as Zephyr module, the file `zephyr/module.yml` must exist at the top level directory in the project.

# Reference

<a href="https://docs.pytorch.org/executorch">Documentation</a>
