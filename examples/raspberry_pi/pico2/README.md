# Overview
This document outlines the steps required to run a simple Add Module on the Pico2 microcontroller using executorch.

## (Pre-requisistes) Prepare the Environment for Arm

1. Setup executorch development environment, Also see  <a href="https://docs.pytorch.org/executorch/main/tutorial-arm-ethos-u.html#software"/> for instructions on setting up the environment for Arm.
2. Make sure you have the toolchain configured correctly.

```bash
which arm-none-eabi-gcc
--> return something like executorch/examples/arm/ethos-u-scratch/arm-gnu-toolchain-13.3.rel1-x86_64-arm-none-eabi/bin/arm-none-eabi-gcc
```

## Build Pico2 Firmware with Executorch

This step involves two sub steps

1. Cross Compile Executorch for Arm Cortex M, Pico2 target
2. Build the firmware with the input model provided (If not provided, it will use the default_model.pte)

Use the following command to build the firmware:
``` bash
executorch/examples/rpi/build_firmware_pico.sh --model=<path_to_model.pte>
```

### Flash Firmware

Hold the BOOTSEL button on the Pico2 and connect it to your computer; it will mount as RPI-RP2. Copy the executorch_pico.uf2 file to this drive.

### Verify Execution

Check that the Pico2's LED blinks 10 times at 500 ms interval to confirm successful firmware execution.
The Pico2's LED should blink 10 times at 500 ms intervals, indicating successful firmware execution. If connected via serial, you should see:

```bash
Method loaded [forward]
Output: 13.000000, 136.000000, 24.000000, 131.000000
```

### Debugging via Serial Terminal

On macOS or Linux, open a serial terminal with:

```bash
screen /dev/tty.usbmodem1101 115200
```

Replace /dev/tty.usbmodem1101 with your device path. This terminal shows program logs and errors. If
the LED blinks 10 times at 100 ms intervals, your program hit an error state—check the logs here.

These steps complete running the simple model on Pico2 using ExecuTorch.
