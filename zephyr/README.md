# Executorch

**ExecuTorch** is PyTorch's unified solution for deploying AI models on-device—from smartphones to microcontrollers—built for privacy, performance, and portability. It powers Meta's on-device AI across **Instagram, WhatsApp, Quest 3, Ray-Ban Meta Smart Glasses**, and [more](https://docs.pytorch.org/executorch/main/success-stories.html).

This folder integrates ExecuTorch as a Zephyr module. It also contains sample
applications under `zephyr/samples/`, including
`zephyr/samples/hello-executorch` for embedding and running a `.pte` model.

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

To pull in ExecuTorch as a Zephyr module, either add it as a West project in
`west.yaml` or include it through a submanifest such as
`zephyr/submanifests/executorch.yaml`.

# Create executorch.yaml under zephyr/submanifests

There is an example `executorch.yaml` in this folder you can copy, or you can
create `<zephyr_build_root>/zephyr/submanifests/executorch.yaml` with the
following content:

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

Add ExecuTorch and Ethos&trade;-U driver to Zephyr
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

## Prepare Ethos-U tools like Vela compiler and Corstone&trade; 300/320 FVP

This installs the tools needed to export Python models to PTE files for
Ethos-Ux5 and also installs the Corstone 300/320 FVPs used by the sample
flows.

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

## Model export config

Zephyr samples can auto-generate a `.pte` file during CMake configure by using
the `CONFIG_EXECUTORCH_EXPORT_PYTHON_*` Kconfig options in `prj.conf` or in a
board-specific `boards/*.conf` file.

- `CONFIG_EXECUTORCH_EXPORT_PYTHON_SCRIPT` selects the Python exporter script to
  run.
- `CONFIG_EXECUTORCH_EXPORT_PYTHON_ARGS` passes extra arguments to that script.
- `CONFIG_EXECUTORCH_EXPORT_PYTHON_DEPENDENCIES` lists local helper files that
  should trigger reconfigure when changed.
- `CONFIG_EXECUTORCH_EXPORT_PYTHON_GENERATED_OUTPUT` tells Zephyr which output
  file the script is expected to write.

These settings can be overridden per board in `boards/<board>.conf`, which is
useful when different targets need different exporter scripts or generated model
file names. The `hello-executorch` sample uses this to select different default
exporters for Corstone-300 and Corstone-320.

## Notable files

# executorch.yaml

Copy this to <zephyr_build_root>/zephyr/submanifests/

# module.yml

Do not remove this file. As described in the official Zephyr
[documentation](https://docs.zephyrproject.org/latest/develop/modules.html),
`zephyr/module.yml` must exist at the top level of the project for ExecuTorch
to be discovered as a Zephyr module.

# Reference

<a href="https://docs.pytorch.org/executorch">Documentation</a>

## Related Projects

- [ExecuTorch on Zephyr RTOS with CMSIS](https://github.com/Arm-Examples/cmsis-zephyr-executorch) — An alternative project structure demonstrating ExecuTorch on Zephyr using CMSIS Toolbox for build management.
