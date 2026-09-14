# Pico2: A simple MNIST Tutorial

Deploy your PyTorch models directly to Raspberry Pi Pico2 microcontroller with ExecuTorch.

## What You'll Build

A 28×28 MNIST digit classifier running on memory constrained, low power microcontrollers:

- Input: ASCII art digits (0, 1, 4, 7)
- Output: Real-time predictions via USB serial
- Memory: <400KB total footprint
- Two variants: FP32 (portable ops) and INT8 (CMSIS-NN accelerated)

## Prerequisites

- [Environment Setup section](https://docs.pytorch.org/executorch/1.0/using-executorch-building-from-source.html)

- Refer to  this link on how to accept 'EULA' agreement and setup toolchain [link](https://docs.pytorch.org/executorch/1.0/backends-arm-ethos-u.html#development-requirements)

- Verify ARM toolchain

```bash
which arm-none-eabi-gcc # --> arm/arm-scratch/arm-gnu-toolchain-13.3.rel1-x86_64-arm-none-eabi/bin/
```

## Step 1: Generate pte from given example Model

### FP32 model (default)

- Use the [provided example model](https://github.com/pytorch/executorch/blob/main/examples/raspberry_pi/pico2/export_mlp_mnist.py)

```bash
cd examples/raspberry_pi/pico2
python export_mlp_mnist.py # Creates balanced_tiny_mlp_mnist.pte
```

- **Note:** This is hand-crafted MNIST Classifier (proof-of-concept), and not production trained. This tiny MLP recognizes digits 0, 1, 4, and 7 using manually designed feature detectors.

### INT8 quantized model (CMSIS-NN accelerated)

- Use the [CMSIS-NN export script](https://github.com/pytorch/executorch/blob/main/examples/raspberry_pi/pico2/export_mlp_mnist_cmsis.py)

```bash
cd examples/raspberry_pi/pico2
python export_mlp_mnist_cmsis.py # Creates balanced_tiny_mlp_mnist_cmsis.pte
```

This uses the `CortexMQuantizer` to produce INT8 quantized ops that map to CMSIS-NN kernels on Cortex-M33. The model I/O stays float — quantize and dequantize nodes are inserted inside the graph.

## Step 2: Build Firmware for Pico2

### FP32 build

```bash
# Generate model (Creates balanced_tiny_mlp_mnist.pte)
cd ./examples/raspberry_pi/pico2
python export_mlp_mnist.py
cd -

# Build Pico2 firmware (one command!)
./examples/raspberry_pi/pico2/build_firmware_pico.sh --model=balanced_tiny_mlp_mnist.pte
```

### INT8 CMSIS-NN build

```bash
# Generate INT8 quantized model
cd ./examples/raspberry_pi/pico2
python export_mlp_mnist_cmsis.py
cd -

# Build with CMSIS-NN backend
./examples/raspberry_pi/pico2/build_firmware_pico.sh --cmsis --model=balanced_tiny_mlp_mnist_cmsis.pte
```

Output: **executorch_pico.uf2** firmware file (examples/raspberry_pi/pico2/build/)

**Script options:**
| Flag | Description |
|------|-------------|
| `--model=FILE` | Specify model file to embed (relative to pico2/) |
| `--cmsis` | Build with CMSIS-NN INT8 kernels for Cortex-M33 acceleration |
| `--clean` | Clean build directories and exit; run separately before building if needed |

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

📊 Memory usage after method load:
   Method allocator: 45632 / 204800 bytes used
   Activation pool: 204800 bytes allocated

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
⏱️  Inference time: 245 us
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

🎯 PREDICTED: 7 (Expected: 7) ✅ CORRECT!

==================================================

📊 Inference latency summary:
  Digit 0: 312 us
  Digit 1: 198 us
  Digit 4: 267 us
  Digit 7: 245 us
  Average: 255 us
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

## CMSIS-NN INT8 Acceleration

The Pico2 uses an RP2350 SoC with a Cortex-M33 core. The CMSIS-NN library provides optimized INT8 kernels that leverage the Cortex-M33's DSP instructions for faster inference compared to FP32 portable ops.

### How it works

1. `export_mlp_mnist_cmsis.py` uses `CortexMQuantizer` to quantize the model to INT8
2. The model I/O remains float — quantize/dequantize nodes are inserted inside the graph
3. `--cmsis` flag builds ExecuTorch with the Cortex-M backend and links CMSIS-NN kernels
4. At runtime, quantized linear ops dispatch to CMSIS-NN instead of portable kernels

### When to use CMSIS-NN

- Lower latency on supported ops (linear, conv2d)
- Smaller model size (INT8 weights vs FP32)
- Trade-off: slight accuracy loss from quantization

## Next Steps

### Scale up your deployment

- Use real production trained model
- Optimize further → INT8 quantization with CMSIS-NN, pruning

### Happy Inference!

**Result:** PyTorch model → Pico2 deployment in 4 simple steps 🚀
Total tutorial time: ~15 minutes

**Conclusion:** Real-time inference on memory constrained, low power microcontrollers, a complete PyTorch → ExecuTorch → Pico2 demo MNIST deployment
