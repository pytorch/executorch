# Pico2: A simple MNIST Tutorial

Deploy your PyTorch models directly to Raspberry Pi Pico2 microcontroller with ExecuTorch.

## What You'll Build

A 28×28 MNIST digit classifier running on memory constrained, low power microcontrollers:

- Input: ASCII art digits (0, 1, 4, 7)
- Output: Real-time predictions via USB serial
- Memory: <400KB total footprint

## Prerequisites

- [Environment Setup section](https://docs.pytorch.org/executorch/1.0/using-executorch-building-from-source.html)

- Refer to  this link on how to accept 'EULA' agreement and setup toolchain [link](https://docs.pytorch.org/executorch/1.0/backends-arm-ethos-u.html#development-requirements)

- Verify ARM toolchain

```bash
which arm-none-eabi-gcc # --> arm/ethos-u-scratch/arm-gnu-toolchain-13.3.rel1-x86_64-arm-none-eabi/bin/
```

## Step 1: Generate pte from given example Model

- Use the [provided example model](https://github.com/pytorch/executorch/blob/main/examples/raspberry_pi/pico2/export_mlp_mnist.py)

```bash
python export_mlp_mnist.py # Creates balanced_tiny_mlp_mnist.pte
```

- **Note:** This is hand-crafted MNIST Classifier (proof-of-concept), and not production trained. This tiny MLP recognizes digits 0, 1, 4, and 7 using manually designed feature detectors.

## Step 2: Build Firmware for Pico2

```bash
# Generate model (Creates balanced_tiny_mlp_mnist.pte)
cd ./examples/raspberry_pi/pico2
python export_mlp_mnist.py
cd -

# Build Pico2 firmware (one command!)

./examples/raspberry_pi/pico2/build_firmware_pico.sh --model=balanced_tiny_mlp_mnist.pte   # This creates executorch_pico.uf2, a firmware image for Pico2
```

Output: **executorch_pico.uf2** firmware file (examples/raspberry_pi/pico2/build/)

**Note:** '[build_firmware_pico.sh](https://github.com/pytorch/executorch/blob/main/examples/raspberry_pi/pico2/build_firmware_pico.sh)' script converts given model pte to hex array and generates C code for the same via this helper [script](https://github.com/pytorch/executorch/blob/main/examples/raspberry_pi/pico2/pte_to_array.py). This C code is then compiled to generate final .uf2 binary which is then flashed to Pico2.

## Step 3: Flash to Pico2

Hold BOOTSEL button on Pico2
Connect USB → Mounts as ^RPI-RP2^ drive
Drag & drop ^executorch_pico.uf2^ file
Release BOOTSEL → Pico2 reboots with your model

## Step 4: Verify Deployment

**Success indicators:**

- LED blinks 10× at 500ms → Model running ✅
- LED blinks 10× at 100ms → Error, check serial ❌

**View predictions:**

```bash
# Connect serial terminal
screen /dev/tty.usbmodem1101 115200
# Expected output:

Something like:

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
```

## Memory Optimization Tips

### Pico2 Constraints

- 520KB SRAM (runtime memory)
- 4MB Flash (model storage)
- Keep models small:

### Common Issues

- "Memory allocation failed" → Reduce model size and use quantization
- "Operator missing" → Use selective build: ^--operators=add,mul,relu^
- "Import error" → Check ^arm-none-eabi-gcc^ toolchain setup.

In order to resolve some of the issues above, refer to the following guides:

- [ExecuTorch Quantization Optimization Guide](https://docs.pytorch.org/executorch/1.0/quantization-optimization.html)
- [Model Export & Lowering](https://docs.pytorch.org/executorch/1.0/using-executorch-export.html) and
- [Selective Build support](https://docs.pytorch.org/executorch/1.0/kernel-library-selective-build.html)

### Firmware Size Analysis

```bash
cd <root of executorch repo>
ls -al examples/raspberry_pi/pico2/build/executorch_pico.elf
```

- **Overall section sizes**

```bash
arm-none-eabi-size -A examples/raspberry_pi/pico2/build/executorch_pico.elf
```

- **Detailed section breakdown**

```bash
arm-none-eabi-objdump -h examples/raspberry_pi/pico2/build/executorch_pico.elf
```

- **Symbol sizes (largest consumers)**

```bash
arm-none-eabi-nm --print-size --size-sort --radix=d examples/raspberry_pi/pico2/build/executorch_pico.elf | tail -20
```

### Model Memory Footprint

- **Model data specifically**

```bash
arm-none-eabi-nm --print-size --size-sort --radix=d examples/raspberry_pi/pico2/build/executorch_pico.elf | grep -i model
```

- **Check what's in .bss (uninitialized data)**

```bash
arm-none-eabi-objdump -t examples/raspberry_pi/pico2/build/executorch_pico.elf | grep ".bss" | head -10
```

- **Memory map overview**

```bash
arm-none-eabi-readelf -l examples/raspberry_pi/pico2/build/executorch_pico.elf
```

## Next Steps

### Scale up your deployment

- Use real production trained model
- Optimize further → INT8 quantization, pruning

### Happy Inference!

**Result:** PyTorch model → Pico2 deployment in 4 simple steps 🚀
Total tutorial time: ~15 minutes

**Conclusion:** Real-time inference on memory constrained, low power microcontrollers, a complete PyTorch → ExecuTorch → Pico2 demo MNIST deployment
