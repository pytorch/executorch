## Overview
This document outlines the steps required to run a simple Add Module on the Pico2 microcontroller using executorch.
## Steps

### (Pre-requisistes) Prepare the Environment for Arm
1. See <a href="https://docs.pytorch.org/executorch/main/tutorial-arm.html#set-up-the-developer-environment"/> for instructions on setting up the environment for Arm.
2. Make sure you have the toolchain configured correctly.
```bash
which arm-none-eabi-gcc
``` should return something like 'executorch/examples/arm/ethos-u-scratch/arm-gnu-toolchain-13.3.rel1-x86_64-arm-none-eabi/bin/arm-none-eabi-gcc'

### 1. Cross Compile Executorch for Arm Cortex M Target
To begin, navigate to the executorch root directory and execute the following commands:
```bash
cmake -B cmake-out \
  -DCMAKE_TOOLCHAIN_FILE=examples/arm/ethos-u-setup/arm-none-eabi-gcc.cmake \
  -DTARGET_CPU=cortex-m0plus \
  -DEXECUTORCH_BUILD_ARM_BAREMETAL=ON \
  -DEXECUTORCH_PAL_DEFAULT=minimal \
  -DEXECUTORCH_DTYPE_SELECTIVE_BUILD=ON \
  -DCMAKE_BUILD_TYPE=MinSizeRel \
  -DEXECUTORCH_ENABLE_LOGGING=OFF \
  -DEXECUTORCH_SELECT_ALL_OPS=OFF \
  -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF \
  -DCMAKE_INSTALL_PREFIX=cmake-out .; \
cmake --build cmake-out --target install -j$(nproc);
```

### 2. Export PICO_SDK_PATH
Download the Pico SDK from GitHub: https://github.com/raspberrypi/pico-sdk and set the PICO_SDK_PATH environment variable:
```bash
export PICO_SDK_PATH=<path_to_local_pico_sdk_folder>
```

### 3. Build the example for Pico2
Go to the example directory and initiate the build process:
```bash
cd examples/arm/raspberry_pi/pico2/
rm -rf build
mkdir build
cd build
cmake .. -DPICO_BOARD=pico2 -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```
This step will generate the firmware file executorch_pico.uf2

### 4. Flash the Firmware
Press and hold the BOOTSEL button on the Pico2.
Connect the Pico2 to your computer; it should mount as RPI-RP2.
Copy the executorch_pico.uf2 file to the mounted drive.

### 5. Verify the Firmware
Check that the Pico2's LED blinks 10 times at 500 ms interval to confirm successful firmware execution.
You should see the output (if the serial port is connected, see below for details) :
````bash
Method loaded [forward]
Output: 13.000000, 136.000000, 24.000000, 131.000000
```

### 6. Steps to debug / triage using a serial terminal

On macOS or Linux, run the following command to open a serial terminal for the Pico2:
```bash
screen /dev/tty.usbmodem1101 115200
```

Make sure to replace /dev/tty.usbmodem1101 with the actual device path for your Pico if different.
This will open a serial terminal at 115200 baud rate, where you should see the printf output from your program, including any logs or error messages printed during execution.
If you see the LED blink 10 times at 100 ms interval, that indicates your program reached the error indicator code, so you should also see the corresponding logs in this terminal.

These steps complete the process required to run the simple Add Module on the Pico2 microcontroller using executorch.

### 6. Other Tips

a. The pte_to_header.py script converts binary PTE files into C++ header files containing byte arrays.
```bash
python ./examples/arm/executor_runner/pte_to_header.py -p model.pte
```

b. The following command will generate a simple_ops.txt file with the list of operators used in any given model. This can be used to verify all the operators that should be included in the build.
From the root executor dir,
```bash
python -m executorch.codegen.tools.gen_oplist   --output_path simple_ops.txt   --model_file_path ./model.pte
```

c.
