# ExecuTorch Zephyr Samples

This document contains build and run instructions for the examples under
`zephyr/samples/`.

## Running a sample application

To run you need to point to the path to the installed Corstone&trade; FVP for Ethos&trade;-U55/U85 and you can then use west to build and run. You point out the model PTE file you want to run with `-DET_PTE_FILE_PATH=` (see below).

The magic to include and use Ethos-U backend is to set `CONFIG_ETHOS_U=y/n`.
This is done in the example depending on the board you build for so if you build for a different board then the ones below you might want to add a board config file, or add this line to the `prj.conf`.

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

#### Prepare a PTE model file

Prepare the Ethos-U55 PTE model
<!-- RUN test_ethos-u55_generate_pte -->
```
python -m modules.lib.executorch.backends.arm.scripts.aot_arm_compiler --model_name=modules/lib/executorch/zephyr/samples/hello-executorch/models/add.py --quantize --delegate --target=ethos-u55-128 --output=add_u55_128.pte
```

`--delegate` tells the `aot_arm_compiler` to use Ethos-U backend and `-t ethos-u55-128` specifies the used Ethos-U variant and numbers of macs used, this must match your hardware or FVP config.

#### Build and run

Run the Ethos-U55 PTE model
<!-- RUN test_ethos-u55_build_and_run -->
```
west build -b mps3/corstone300/fvp modules/lib/executorch/zephyr/samples/hello-executorch -t run -- -DET_PTE_FILE_PATH=add_u55_128.pte
```

### Cortex-M55

#### Prepare a PTE model file

Prepare the Cortex-M55 PTE model
<!-- RUN test_cortex-m55_generate_pte -->
```
python -m modules.lib.executorch.backends.arm.scripts.aot_arm_compiler --model_name=modules/lib/executorch/zephyr/samples/hello-executorch/models/add.py --quantize --target=cortex-m55 --output=add_m55.pte
```

`--target=cortex-m55` plus `--quantize` selects the Cortex-M/CMSIS-NN portable kernel path (no NPU delegation). This produces a `.pte` optimized for Cortex-M55 with INT8 quantization.


#### Build and run

Run the Cortex-M55 PTE model
<!-- RUN test_cortex-m55_build_and_run -->
```
west build -b mps3/corstone300/fvp modules/lib/executorch/zephyr/samples/hello-executorch -t run -- -DET_PTE_FILE_PATH=add_m55.pte
```

## Corstone 320 FVP (Ethos-U85)

### Setup FVP paths

Set up FVP paths, libs and macs used, this will also set `shutdown_on_eot` so the FVP auto stops after it has run the example.

Config Zephyr Corstone320 FVP
<!-- RUN setup_corstone320_fvp -->
```
export FVP_ROOT=$PWD/modules/lib/executorch/examples/arm/arm-scratch/FVP-corstone320
export ARMFVP_BIN_PATH=${FVP_ROOT}/models/Linux64_GCC-9.3
export LD_LIBRARY_PATH=${FVP_ROOT}/python/lib:${ARMFVP_BIN_PATH}:${LD_LIBRARY_PATH}
export ARMFVP_EXTRA_FLAGS="-C mps4_board.uart0.shutdown_on_eot=1 -C mps4_board.subsystem.ethosu.num_macs=256"
```

### Ethos-U85

#### Prepare a PTE model file

Prepare the Ethos-U85 PTE model
<!-- RUN test_ethos-u85_generate_pte -->
```
python -m modules.lib.executorch.backends.arm.scripts.aot_arm_compiler --model_name=modules/lib/executorch/zephyr/samples/hello-executorch/models/add.py --quantize --delegate --target=ethos-u85-256 --output=add_u85_256.pte
```

`--delegate` tells the `aot_arm_compiler` to use Ethos-U backend and `-t ethos-u85-256` specifies the used Ethos-U variant and numbers of macs used, this must match your hardware or FVP config.

#### Build and run

Run the Ethos-U85 PTE model
<!-- RUN test_ethos-u85_build_and_run -->
```
west build -b mps4/corstone320/fvp modules/lib/executorch/zephyr/samples/hello-executorch -t run -- -DET_PTE_FILE_PATH=add_u85_256.pte
```

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

### Prepare a PTE model file

Prepare the Cortex-M55 PTE model
```
python -m modules.lib.executorch.backends.arm.scripts.aot_arm_compiler --model_name=modules/lib/executorch/zephyr/samples/hello-executorch/models/add.py --quantize --target=cortex-m55 --output=add_m55.pte
```

`--target=cortex-m55` plus `--quantize` selects the Cortex-M/CMSIS-NN portable kernel path (no NPU delegation). This produces a `.pte` optimized for Cortex-M55 with INT8 quantization.

#### Build and run

Run the Cortex-M55 PTE model
```
west build -b nucleo_n657x0_q modules/lib/executorch/zephyr/samples/hello-executorch -- -DET_PTE_FILE_PATH=add_m55.pte
west flash
```

This will run the simple add model on your hardware one and print the output on the serial console.
