# Executorch

**ExecuTorch** is PyTorch's unified solution for deploying AI models on-device—from smartphones to microcontrollers—built for privacy, performance, and portability. It powers Meta's on-device AI across **Instagram, WhatsApp, Quest 3, Ray-Ban Meta Smart Glasses**, and [more](https://docs.pytorch.org/executorch/main/success-stories.html).

This folder adds ExecuTorch so it can be build and run in the Zephyr project as a external module. This includes an example under zephyr/samples/hello-executorch of running executor runners with Arm&reg; Ethos&trade;-U backend on a Corstone&trade; FVP, targeting the Zephyr RTOS.

# Requirements

Setup a venv (or conda)
```
mkdir <zephyr_build_root>
cd <zephyr_build_root>
python3 -m venv .zephyr_venv
source .zephyr_venv/bin/activate
```

Install requirements
<!-- RUN install_reqs -->
```
pip install west "cmake<4.0.0" pyelftools ninja jsonschema
```

Setup zephyr repo
<!-- RUN west_init -->
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

Add ExecuTorch and Ethos-U driver to Zephyr
<!-- RUN west_config -->
```
west config manifest.project-filter -- -.*,+zephyr,+executorch,+cmsis,+cmsis_6,+cmsis-nn,+hal_ethos_u
west update
```

## Setup and install ExecuTorch

Setup ExecuTorch
<!-- RUN install_executorch -->
```
cd modules/lib/executorch/
git submodule sync
git submodule update --init --recursive
./install_executorch.sh
cd ../../..
```

## Prepare Ethos-U tools like Vela compiler and Corstone 300/320 FVP

This is needed to convert python models to PTE files for Ethos-Ux5 and also installs Corstone 300/320 FVP so you can run and test.

Make sure to read and agree to the Corstone eula

Install TOSA, vela and FVPs
<!-- RUN install_arm_tools -->
```
modules/lib/executorch/examples/arm/setup.sh --i-agree-to-the-contained-eula
. modules/lib/executorch/examples/arm/arm-scratch/setup_path.sh
```

# Running sample applications

Build and run instructions for simple Zephyr minimal example setup is documented in
[`zephyr/samples/hello-executorch/README.md`](samples/hello-executorch/README.md).

## Notable files

# executorch.yaml

Copy this to <zephyr_build_root>/zephyr/submanifests/

# module.yml

Do not remove this file. As mentioned in the official Zephyr [documenation](https://docs.zephyrproject.org/latest/develop/modules.html), for Executorch to be built as Zephyr module, the file `zephyr/module.yml` must exist at the top level directory in the project.

# Reference

<a href="https://docs.pytorch.org/executorch">Documentation</a>

## Related Projects

- [ExecuTorch on Zephyr RTOS with CMSIS](https://github.com/Arm-Examples/cmsis-zephyr-executorch) — An alternative project structure demonstrating ExecuTorch on Zephyr using CMSIS Toolbox for build management.
