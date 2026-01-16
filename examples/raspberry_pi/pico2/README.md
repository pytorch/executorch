# Overview

This document outlines the steps required to run a simple MNIST digit recognition neural network on the Pico2 microcontroller using ExecuTorch.

## Demo Model: Hand-crafted MNIST Classifier

The included `export_mlp_mnist.py` (in examples/raspberry_pi/pico2) creates a demonstration model with hand-crafted weights (not production-trained). This tiny MLP recognizes digits 0, 1, 4, and 7 using manually designed feature detectors.
Note: This is a proof-of-concept. For production use, train your model on real MNIST data.

## Bring Your Own Model and Deploy

This demo demonstrates ExecuTorch's ability to bring your own PyTorch model and deploy it to Pico2 with one simple script. The complete pipeline works from any PyTorch model to a runnable binary:

- Use existing demo model (examples/raspberry_pi/pico2/export_mlp_mnist.py) or bring your own model
- Build firmware with one command and pass the model file (.pte) as an argument
- Deploy directly to Pico2

### Important Caveats

- Memory constraints - Models must fit in 520KB SRAM (Pico2)
- Missing operators - Some ops may not be supported
- Selective builds - Include only operators your model uses if you want to reduce binary size

## Memory Constraints & Optimization

- Critical: Pico2 has limited memory
  - 520KB SRAM (on-chip static RAM)
  - 4MB QSPI Flash (onboard storage)

### Always apply optimization techniques on large models that do not fit in Pico2 memory:

Large models will not fit. Keep your `.pte` files small!

- Quantization (INT8, INT4)
- Model pruning
- Operator fusion
- Selective builds (include only needed operators)

For more details , refer to the following guides:

- [ExecuTorch Quantization Optimization Guide](https://docs.pytorch.org/executorch/1.0/quantization-optimization.html)
- [Model Export & Lowering](https://docs.pytorch.org/executorch/1.0/using-executorch-export.html) and
- [Selective Build support](https://docs.pytorch.org/executorch/1.0/kernel-library-selective-build.html)

## (Prerequisites) Prepare the Environment for Arm

Setup executorch development environment. Also see instructions for setting up the environment for Arm.
Make sure you have the toolchain configured correctly. Refer to this [setup](https://docs.pytorch.org/executorch/main/backends-arm-ethos-u.html#development-requirements) for more details.

```bash
which arm-none-eabi-gcc
# Should return: executorch/examples/arm/arm-scratch/arm-gnu-toolchain-13.3.rel1-x86_64-arm-none-eabi/bin/arm-none-eabi-gcc
```

## Build Pico2 Firmware with ExecuTorch

This involves two steps:

### Generate your model:

```bash
cd examples/raspberry_pi/pico2
python export_mlp_mnist.py # Creates balanced_tiny_mlp_mnist.pte
```

### Build firmware:

```bash
# In the dir examples/raspberry_pi/pico2
./build_firmware_pico.sh --model=balanced_tiny_mlp_mnist.pte # This creates executorch_pico.uf2, a firmware image for Pico2
```

### Flash Firmware

Hold the BOOTSEL button on Pico2 and connect to your computer. It mounts as `RPI-RP2`. Copy `executorch_pico.uf2` to this drive.

### Verify Execution

The Pico2 LED blinks 10 times at 500ms intervals for successful execution. Via serial terminal, you'll see:

```bash
...
...
PREDICTED: 4 (Expected: 4) ✅ CORRECT!

==================================================

=== Digit 7 ===
############################
############################
                        ####
                       ####
                      ####
                     ####
                    ####
                   ####
                  ####
                 ####
                ####
               ####
              ####
             ####
            ####
           ####
          ####
         ####
        ####
       ####
      ####
     ####
    ####
   ####
  ####
 ####
####
###

Input stats: 159 white pixels out of 784 total
Running neural network inference...
✅ Neural network results:
  Digit 0: 370.000
  Digit 1: 0.000
  Digit 2: -3.000
  Digit 3: -3.000
  Digit 4: 860.000
  Digit 5: -3.000
  Digit 6: -3.000
  Digit 7: 1640.000 ← PREDICTED
  Digit 8: -3.000
  Digit 9: -3.000

� PREDICTED: 7 (Expected: 7) ✅ CORRECT!

==================================================

🎉 All tests complete! PyTorch neural network works on Pico2!
```

### Debugging via Serial Terminal

On macOS/Linux:

```bash
screen /dev/tty.usbmodem1101 115200
```

Replace `/dev/tty.usbmodem1101` with your device path. If LED blinks 10 times at 100ms intervals, check logs for errors, but if it blinks 10 times at 500ms intervals, it is successful!

Result: A complete PyTorch → ExecuTorch → Pico2 demo MNIST deployment! 🚀
