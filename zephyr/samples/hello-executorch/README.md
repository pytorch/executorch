# ExecuTorch Zephyr Samples

This sample uses a tiny `x + x` model to show the minimum pieces needed to run
an ExecuTorch model in a Zephyr application.

The model is exported from Python during CMake configure. This sample keeps a
few different exporter scripts under `models/` so it can demonstrate Cortex-M,
Ethos&trade;-U55, and Ethos&trade;-U85 flows. In a real application you would usually keep
just the exporter you need.

The auto-export path is controlled by these Kconfig options:

- `CONFIG_EXECUTORCH_EXPORT_PYTHON_SCRIPT` selects the exporter script.
- `CONFIG_EXECUTORCH_EXPORT_PYTHON_GENERATED_OUTPUT` names the `.pte` file that
  script writes.
- `CONFIG_EXECUTORCH_EXPORT_PYTHON_ARGS` can be used to pass extra arguments to
  the exporter script.
- `CONFIG_EXECUTORCH_EXPORT_PYTHON_DEPENDENCIES` lists helper files that should
  trigger reconfigure when changed.

The sample sets a default Cortex-M exporter in `prj.conf` and overrides it in
board-specific `boards/*.conf` files for the Corstone&trade; FVP one Ethos-U boards.

You can also bypass auto-export entirely and point the build at a prebuilt model
with `-DET_PTE_FILE_PATH=<model>.pte`.

If you override `CONFIG_EXECUTORCH_EXPORT_PYTHON_*` from the `west build`
command line, remember that they are Kconfig string symbols, so the value must
include embedded double quotes.

For example, to force the Cortex-M exporter on `mps3/corstone300/fvp`:

```
west build -d build-hello-executorch_cortex-m55 -b mps3/corstone300/fvp modules/lib/executorch/zephyr/samples/hello-executorch -t run -- '-DCONFIG_EXECUTORCH_EXPORT_PYTHON_SCRIPT="models/add_cortex-m.py"' '-DCONFIG_EXECUTORCH_EXPORT_PYTHON_GENERATED_OUTPUT="hello_executorch_cortex-m.pte"'
```


## Running a sample application

To run the Ethos-U55/U85 sample flows you need to add the directory of the
installed Corstone FVP. The default flow auto-generates the model from
`prj.conf` and any matching board config. For testing you can also use
`-DET_PTE_FILE_PATH=` to point to a prebuilt model PTE file instead.

The magic to include and use Ethos-U backend is to set `CONFIG_ETHOS_U=y/n`.
This is set automatically for the Ethos-U sample boards in `boards/*.conf`. If
you build for a different board, add a matching board config file or put the
setting directly in `prj.conf`.

## Corstone 300 FVP (Ethos-U55)

### Setup FVP paths

Set up FVP paths and macs used, this will also set `shutdown_on_eot` so the FVP auto stops after it has run the example.

Config Zephyr Corstone300 FVP
<!-- RUN setup_corstone300_fvp -->
```
export FVP_ROOT=$PWD/modules/lib/executorch/examples/arm/arm-scratch/FVP-corstone300
export ARMFVP_BIN_PATH=${FVP_ROOT}/models/Linux64_GCC-9.3
export ARMFVP_EXTRA_FLAGS="-C mps3_board.uart0.shutdown_on_eot=1 -C ethosu.num_macs=128"
```

### Ethos-U55

#### Build and run

Run the Ethos-U55 exporter configured by the project
<!-- RUN test_ethos-u55_build_and_run -->
```
west build -d build-hello-executorch_ethos-u55 -b mps3/corstone300/fvp modules/lib/executorch/zephyr/samples/hello-executorch -t run
```

#### Prepare a PTE model file

To use a prebuilt model instead of the auto-generated one:

Prepare and run a separate Ethos-U55 PTE model
<!-- RUN test_ethos-u55_generate_pte -->
```
python -m modules.lib.executorch.backends.arm.scripts.aot_arm_compiler --model_name=modules/lib/executorch/zephyr/samples/hello-executorch/models/add_ethos-u55.py --quantize --delegate --target=ethos-u55-128 --output=add_u55_128.pte
west build -d build-hello-executorch_ethos-u55 -b mps3/corstone300/fvp modules/lib/executorch/zephyr/samples/hello-executorch -t run -- -DET_PTE_FILE_PATH=add_u55_128.pte
```

`--delegate` tells the `aot_arm_compiler` to use Ethos-U backend and `-t ethos-u55-128` specifies the used Ethos-U variant and numbers of macs used, this must match your hardware or FVP config.

### Cortex-M55 (Corstone 300 FVP)

This sample reuses the Corstone-300 FVP for both Ethos-U55 and Cortex-M55
examples. Because the board config defaults to the Ethos-U55 exporter, the
Cortex-M55 example overrides the exporter on the command line.

#### Build and run

Run the Cortex-M55 exporter
<!-- RUN test_cortex-m55_build_and_run -->
```
west build -d build-hello-executorch_cortex-m55 -b mps3/corstone300/fvp modules/lib/executorch/zephyr/samples/hello-executorch -t run -- '-DCONFIG_EXECUTORCH_EXPORT_PYTHON_SCRIPT="models/add_cortex-m.py"' '-DCONFIG_EXECUTORCH_EXPORT_PYTHON_GENERATED_OUTPUT="hello_executorch_cortex-m.pte"'
```

#### Prepare a PTE model file

To use a prebuilt model instead of the auto-generated one:

Prepare and run the Cortex-M55 PTE model
<!-- RUN test_cortex-m55_generate_pte -->
```
python -m modules.lib.executorch.backends.arm.scripts.aot_arm_compiler --model_name=modules/lib/executorch/zephyr/samples/hello-executorch/models/add_cortex-m.py --quantize --target=cortex-m55+int8 --output=add_m55.pte
west build -d build-hello-executorch_cortex-m55 -b mps3/corstone300/fvp modules/lib/executorch/zephyr/samples/hello-executorch -t run -- -DET_PTE_FILE_PATH=add_m55.pte
```

`--target=cortex-m55+int8` selects the Cortex-M/CMSIS-NN portable kernel path (no NPU delegation). This produces a `.pte` optimized for Cortex-M55 with INT8 quantization.


## Corstone 320 FVP (Ethos-U85)

### Setup FVP paths

Set up FVP paths, libs and macs used, this will also set `shutdown_on_eot` so the FVP auto stops after it has run the example.

These FVP command-line options are passed through the `ARMFVP_EXTRA_FLAGS`
environment variable. The sample does not set `ARMFVP_FLAGS` in its
`CMakeLists.txt`; the base `ARMFVP_FLAGS` come from the selected Zephyr board's
`board.cmake`.

Config Zephyr Corstone320 FVP
<!-- RUN setup_corstone320_fvp -->
```
export FVP_ROOT=$PWD/modules/lib/executorch/examples/arm/arm-scratch/FVP-corstone320
export ARMFVP_BIN_PATH=${FVP_ROOT}/models/Linux64_GCC-9.3
export LD_LIBRARY_PATH=${FVP_ROOT}/python/lib:${ARMFVP_BIN_PATH}:${LD_LIBRARY_PATH}
export ARMFVP_EXTRA_FLAGS="-C mps4_board.uart0.shutdown_on_eot=1 -C mps4_board.subsystem.ethosu.num_macs=256"
```

### Ethos-U85

#### Build and run

Run the Ethos-U85 exporter configured by the project
<!-- RUN test_ethos-u85_build_and_run -->
```
west build -d build-hello-executorch_ethos-u85 -b mps4/corstone320/fvp modules/lib/executorch/zephyr/samples/hello-executorch -t run
```

#### Prepare a PTE model file

To use a prebuilt model instead of the auto-generated one:

Prepare and run a separate Ethos-U85 PTE model
<!-- RUN test_ethos-u85_generate_pte -->
```
python -m modules.lib.executorch.backends.arm.scripts.aot_arm_compiler --model_name=modules/lib/executorch/zephyr/samples/hello-executorch/models/add_ethos-u85.py --quantize --delegate --target=ethos-u85-256 --output=add_u85_256.pte
west build -d build-hello-executorch_ethos-u85 -b mps4/corstone320/fvp modules/lib/executorch/zephyr/samples/hello-executorch -t run -- -DET_PTE_FILE_PATH=add_u85_256.pte
```

`--delegate` tells the `aot_arm_compiler` to use Ethos-U backend and `-t ethos-u85-256` specifies the used Ethos-U variant and numbers of macs used, this must match your hardware or FVP config.


## STM Nucleo n657x0_q

### Run west config and update

You need to add `hal_stm32` driver to Zephyr.
```
west config manifest.project-filter -- -.*,+zephyr,+executorch,+cmsis,+cmsis_6,+cmsis-nn,+hal_stm32
west update
```

### Setup tools

Follow and make sure tools are setup according to this:

https://docs.zephyrproject.org/latest/boards/st/nucleo_n657x0_q/doc/index.html

Test the `samples/hello_world` in that guide to make sure all tools work.

Please note that the ZephyrOS made a fix for the signing tool version v2.21.0 after the v4.3 release in 20 Nov 2025. Make sure to use a later version of ZephyrOS that contains it.
Also note that the signing tool must be in your path for it to auto sign your elf.

```
export PATH=$PATH:~/STMicroelectronics/STM32Cube/STM32CubeProgrammer/bin
```

#### Build and run

Run the Cortex-M55 sample on the board
```
west build -d build-hello-executorch_nucleo_n6 -b nucleo_n657x0_q modules/lib/executorch/zephyr/samples/hello-executorch
west flash
```

This will run the simple add model on the board and print the output on the
serial console.
