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

### Adapting to Other Baremetal Architectures

While this example targets the Pico2 board, the same pattern — embedding the `.pte` model as a C array, using `BufferDataLoader`, and statically allocating memory — can be adapted to other baremetal targets (e.g., RISC-V) by providing your own CMake toolchain file. The key requirement is a correct selective build (see below) so all operators your model needs are included.

### Important Caveats

- Memory constraints - Models must fit in 520KB SRAM (Pico2)
- Missing operators - If you get "Operator missing" (error 20) at runtime, your build is missing operators that the model needs. Use `EXECUTORCH_SELECT_OPS_MODEL` (see below) to auto-detect the required operators from your `.pte` file.
- Selective builds - Include only operators your model uses to reduce binary size

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

### Selective Build: Choosing the Right Operators

When cross-compiling ExecuTorch for baremetal targets, you need to register the operators your model uses. There are three approaches:

1. **`EXECUTORCH_SELECT_OPS_MODEL` (recommended)** — Point to your `.pte` file and the build system auto-detects all required operators:
   ```bash
   cmake ... -DEXECUTORCH_SELECT_OPS_MODEL=/path/to/model.pte
   ```
   This is the most reliable approach because it reads the exact operators from the serialized model, including any operators introduced by compiler passes or edge IR lowering that may not be obvious from the original PyTorch model.

2. **`EXECUTORCH_SELECT_OPS_LIST`** — Manually specify operators by name:
   ```bash
   cmake ... -DEXECUTORCH_SELECT_OPS_LIST="aten::addmm.out,aten::relu.out,..."
   ```
   This requires you to know the exact operator names (including `.out` suffixes). If you miss any, you'll get "Operator missing" (error 20) at runtime.

3. **All portable operators (no selective build)** — Omit any `EXECUTORCH_SELECT_OPS_*` options when configuring CMake. This registers all portable operators, which is simple but produces larger binaries, an important consideration on memory-constrained targets.

The `build_firmware_pico.sh` script uses `EXECUTORCH_SELECT_OPS_MODEL` by default when a model file is provided.

For more details, refer to the following guides:

- [ExecuTorch Quantization Optimization Guide](https://docs.pytorch.org/executorch/1.0/quantization-optimization.html)
- [Model Export & Lowering](https://docs.pytorch.org/executorch/1.0/using-executorch-export.html)
- [Selective Build support](https://docs.pytorch.org/executorch/1.0/kernel-library-selective-build.html)

## (Prerequisites) Prepare the Environment for Arm

Setup executorch development environment. Also see instructions for setting up the environment for Arm.
Make sure you have the toolchain configured correctly. Refer to this [setup](https://docs.pytorch.org/executorch/main/backends/arm-ethos-u/arm-ethos-u-overview.html#development-requirements) for more details.

```bash
which arm-none-eabi-gcc
# Should return: executorch/examples/arm/arm-scratch/arm-gnu-toolchain-13.3.rel1-x86_64-arm-none-eabi/bin/arm-none-eabi-gcc
```

## Build Pico2 Firmware with ExecuTorch

This involves two steps:

### Generate your model:

**FP32 model (default):**
```bash
cd examples/raspberry_pi/pico2
python export_mlp_mnist.py # Creates balanced_tiny_mlp_mnist.pte
```

**INT8 quantized model (CMSIS-NN accelerated):**
```bash
cd examples/raspberry_pi/pico2
python export_mlp_mnist_cmsis.py # Creates balanced_tiny_mlp_mnist_cmsis.pte
```

### Build firmware:

**FP32 build:**
```bash
# In the dir examples/raspberry_pi/pico2
./build_firmware_pico.sh --model=balanced_tiny_mlp_mnist.pte
```

**INT8 CMSIS-NN build:**
```bash
# In the dir examples/raspberry_pi/pico2
./build_firmware_pico.sh --cmsis --model=balanced_tiny_mlp_mnist_cmsis.pte
```

**Script options:**
| Flag | Description |
|------|-------------|
| `--model=FILE` | Specify model file to embed (relative to pico2/) |
| `--cmsis` | Build with CMSIS-NN INT8 kernels for Cortex-M33 acceleration |
| `--clean` | Clean build directories and exit (run separately before building) |

### Flash Firmware

Hold the BOOTSEL button on Pico2 and connect to your computer. It mounts as `RPI-RP2`. Copy `executorch_pico.uf2` to this drive.

### Verify Execution

The Pico2 LED blinks 10 times at 500ms intervals for successful execution. Via serial terminal, you'll see:

```bash
...
...
🎯 PREDICTED: 4 (Expected: 4) ✅ CORRECT!

==================================================

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

🎉 All tests complete! ExecuTorch inference of neural network works on Pico2!
```

### Debugging via Serial Terminal

On macOS/Linux:

```bash
screen /dev/tty.usbmodem1101 115200
```

Replace `/dev/tty.usbmodem1101` with your device path. If LED blinks 10 times at 100ms intervals, check logs for errors, but if it blinks 10 times at 500ms intervals, it is successful!

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

Result: A complete PyTorch → ExecuTorch → Pico2 demo MNIST deployment!
