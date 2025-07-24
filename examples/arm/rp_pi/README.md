## Overview
This document outlines the steps required to run a simple Add Module on the Pico2 microcontroller using executorch.
## Steps

### (Pre-requisistes) Prepare the Environment for Arm
1. See <a href="docs/source/backends-arm-ethos-u.md"/> for instructions on setting up the environment for Arm.
2. Make sure you have the toolchain configured correctly.
```bash
which arm-none-eabi-gcc
``` should return something like 'executorch/examples/arm/ethos-u-scratch/arm-gnu-toolchain-13.3.rel1-x86_64-arm-none-eabi/bin/arm-none-eabi-gcc'

### 1. Cross Compile Executorch for Arm Cortex M Target
To begin, navigate to the executorch root directory and execute the following commands:
```bash
mkdir baremetal_build
cd baremetal_build
cmake .. -DCMAKE_TOOLCHAIN_FILE=./examples/arm/rp_pi/pico2/arm-cortex-m0plus-toolchain.cmake -DEXECUTORCH_BUILD_ARM_BAREMETAL=ON -DCMAKE_BUILD_TYPE=Release -DROOT_OPS=aten::add.out
cmake --build . -j$(nproc)
```

### 2. Export PICO_SDK_PATH
Download the Pico SDK from GitHub: https://github.com/raspberrypi/pico-sdk and set the PICO_SDK_PATH environment variable:
```bash
export PICO_SDK_PATH=<path_to_local_pico_sdk_folder>
```

### 3. Build the example for Pico2
Go to the example directory and initiate the build process:
```bash
cd examples/arm/rp_pi/pico2/
rm -rf build
mkdir build
cd build
cmake .. -DPICO_BOARD=pico2 -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```
This will generate the firmware file executorch_pico.uf2.

### 4. Flash the Firmware
Press and hold the BOOTSEL button on the Pico2.
Connect the Pico2 to your computer; it should mount as RPI-RP2.
Copy the executorch_pico.uf2 file to the mounted drive.

### 5. Verify the Firmware
Check that the Pico2's LED blinks 10 times to confirm successful firmware execution.

### 6. (Optional) Check USB Logs on Mac
To view USB logs, use the following command (as an example):
```bash
screen /dev/tty.usbmodem1101 115200
```

These steps complete the process required to run the simple Add Module on the Pico2 microcontroller using executorch.
