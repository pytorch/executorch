# Zephyr: MobileNetV2 with Ethos-U NPU (Corstone FVP & Alif E8)

Run a quantized MobileNetV2 image classifier on the
Alif Ensemble E8 DevKit using
ExecuTorch, Zephyr RTOS, and the Arm Ethos-U55 NPU. The same build flow also
works on the Arm Corstone-320 FVP for development without hardware.

## What You'll Build

- A quantized INT8 MobileNetV2 model fully delegated to the Ethos-U55 NPU
  (110 ops, ~19 ms inference on Alif E8)
- A Zephyr RTOS application that loads the `.pte` model, runs inference on a
  static test image, and prints the top-5 ImageNet predictions over UART

## Prerequisites

### Hardware (choose one)

| Target | Description |
|--------|-------------|
| **Alif Ensemble E8 DevKit** | Cortex-M55 HP core + Ethos-U55 (256 MACs), 4.5 MB HP SRAM, MRAM |
| **Corstone-320 FVP** | Virtual platform simulating Cortex-M85 + Ethos-U85 (no hardware needed, Linux only) |

### Software

- Linux x86_64 (FVP and Arm toolchain are Linux-only; macOS can export models
  but cannot run the FVP or flash)
- Python 3.10+
- Alif SE Tools for flashing (Alif hardware only)

## Step 1: Set Up the Zephyr Workspace

Create a workspace, install `west`, and initialize the Zephyr tree:

```bash
mkdir ~/zephyr_workspace && cd ~/zephyr_workspace
python3 -m venv .venv && source .venv/bin/activate
pip install west "cmake<4.0.0" pyelftools ninja jsonschema
west init --manifest-rev v4.3.0
```

Install the Zephyr SDK (compiler toolchain):

```bash
wget https://github.com/zephyrproject-rtos/sdk-ng/releases/download/v0.17.4/zephyr-sdk-0.17.4_linux-x86_64.tar.xz
tar -xf zephyr-sdk-0.17.4_linux-x86_64.tar.xz && rm -f zephyr-sdk-0.17.4_linux-x86_64.tar.xz
./zephyr-sdk-0.17.4/setup.sh -c -t arm-zephyr-eabi
export ZEPHYR_SDK_INSTALL_DIR=$(realpath ./zephyr-sdk-0.17.4)
```

## Step 2: Add ExecuTorch as a Zephyr Module

Copy the submanifest, configure `west` to pull only the modules we need, and
update:

```bash
mkdir -p zephyr/submanifests
cat > zephyr/submanifests/executorch.yaml << 'EOF'
manifest:
  projects:
    - name: executorch
      url: https://github.com/pytorch/executorch
      revision: main
      path: modules/lib/executorch
EOF

west config manifest.project-filter -- -.*,+zephyr,+executorch,+cmsis,+cmsis_6,+cmsis-nn,+hal_ethos_u
west update
```

For Alif boards, also add the Alif HAL:

```bash
west config manifest.project-filter -- -.*,+zephyr,+executorch,+cmsis,+cmsis_6,+cmsis-nn,+hal_ethos_u,+hal_alif
west update
```

## Step 3: Install ExecuTorch and Arm Tools

```bash
cd modules/lib/executorch
git submodule sync && git submodule update --init --recursive
./install_executorch.sh
cd ../../..
```

Install the Arm toolchain, Vela compiler, and Corstone FVPs:

```bash
modules/lib/executorch/examples/arm/setup.sh --i-agree-to-the-contained-eula
source modules/lib/executorch/examples/arm/arm-scratch/setup_path.sh
```

## Step 4: Export the MobileNetV2 Model

Export a quantized INT8 MobileNetV2 with Ethos-U delegation. Choose the target
that matches your hardware:

**For Alif E8 (Ethos-U55 with 256 MACs):**

```bash
python -m modules.lib.executorch.backends.arm.scripts.aot_arm_compiler \
    --model_name=mv2_untrained \
    --quantize --delegate \
    --target=ethos-u55-256 \
    --output=mv2_ethosu.pte
```

**For Corstone-320 FVP (Ethos-U85 with 256 MACs):**

```bash
python -m modules.lib.executorch.backends.arm.scripts.aot_arm_compiler \
    --model_name=mv2_untrained \
    --quantize --delegate \
    --target=ethos-u85-256 \
    --output=mv2_u85_256.pte
```

The `--delegate` flag routes all compatible ops through the Ethos-U backend.
The Vela compiler converts the TOSA intermediate representation into an
optimized command stream for the NPU. Use `mv2` instead of `mv2_untrained` for
meaningful predictions (requires torchvision pretrained weights).

## Step 5: Build the Zephyr Application

**For Alif E8:**

```bash
west build -b alif_e8_dk/ae822fa0e5597xx0/rtss_hp \
    -S ethos-u55-enable \
    modules/lib/executorch/zephyr/samples/mv2-ethosu -- \
    -DET_PTE_FILE_PATH=mv2_ethosu.pte
```

**For Corstone-320 FVP:**

```bash
west build -b mps4/corstone320/fvp \
    modules/lib/executorch/zephyr/samples/mv2-ethosu -- \
    -DET_PTE_FILE_PATH=mv2_u85_256.pte
```

## Step 6a: Run on Corstone-320 FVP

Set up the FVP paths and run:

```bash
export FVP_ROOT=$PWD/modules/lib/executorch/examples/arm/arm-scratch/FVP-corstone320
export ARMFVP_BIN_PATH=${FVP_ROOT}/models/Linux64_GCC-9.3
export LD_LIBRARY_PATH=${FVP_ROOT}/python/lib:${ARMFVP_BIN_PATH}:${LD_LIBRARY_PATH}
export ARMFVP_EXTRA_FLAGS="-C mps4_board.uart0.shutdown_on_eot=1 -C mps4_board.subsystem.ethosu.num_macs=256"

west build -t run
```

MV2 inference is cycle-accurate on the FVP and takes 10-20 minutes of wall
clock. You should see output like:

```
========================================
ExecuTorch MobileNetV2 Classification Demo
========================================
Ethos-U backend registered successfully
Model loaded, has 1 methods
Inference completed in <N> ms
--- Classification Results ---
Top-5 predictions:
  [1] class <id>: <score>
  ...
MobileNetV2 Demo Complete
========================================
```

## Step 6b: Flash and Run on Alif E8

### Flash with Alif SE Tools

Use the Alif SE Tools to program the binary into the E8's MRAM. Create a
`zephyr.json` in the build output directory:

```bash
cat > build/zephyr/zephyr.json << 'EOF'
{
    "HP_img_class": {
        "binary": "zephyr.bin",
        "version": "1.0.0",
        "mramAddress": "0x80008000",
        "cpu_id": "M55_HP",
        "flags": ["boot"],
        "signed": false
    }
}
EOF
```

> **Important:** Use `mramAddress: "0x80008000"` (FLASH_LOAD_OFFSET=0x8000),
> **not** the default `0x80200000`. The default offset does not leave enough
> MRAM for the ~3.5 MB MV2 model blob.

> **Note:** Your SE Tools install may include additional `zephyr.json` entries
> (e.g., a `DEVICE` block referencing `app-device-config.json`). Copy those
> from the SE Tools template if your board configuration requires them.

Generate the table of contents and flash using the SE Tools:

```bash
cd build/zephyr
python <path-to-alif-se-tools>/app-gen-toc.py
python <path-to-alif-se-tools>/app-write-mram.py
cd ../..
```

Refer to the Alif SE Tools documentation for installation and detailed usage.

### Connect Serial Console

Connect to UART4 at 115200 baud. On Linux:

```bash
picocom -b 115200 /dev/ttyUSB0
```

Press the reset button on the E8 DevKit. You should see:

```
Booting Zephyr OS build ff8b8697c0f5 ***

========================================
ExecuTorch MobileNetV2 Classification Demo
========================================

I [executorch:main.cpp] Ethos-U backend registered successfully
I [executorch:main.cpp] Model PTE at 0x8004b290, Size: 3490912 bytes
I [executorch:main.cpp] Model loaded, has 1 methods
I [executorch:main.cpp] Running method: forward
I [executorch:main.cpp] Method allocator pool size: 1572864 bytes.
I [executorch:main.cpp] Setting up planned buffer 0, size 752640.
I [executorch:main.cpp] Loading method...
I [executorch:main.cpp] Method 'forward' loaded successfully
I [executorch:main.cpp] Preparing input: static RGB image (150528 bytes)
I [executorch:main.cpp]
--- Starting inference ---
I [executorch:main.cpp] Inference completed in 19 ms
I [executorch:main.cpp]
--- Classification Results ---
I [executorch:main.cpp] Top-5 predictions:
I [executorch:main.cpp]   [1] class 0: 0.0000
I [executorch:main.cpp]   [2] class 1: 0.0000
I [executorch:main.cpp]   [3] class 2: 0.0000
I [executorch:main.cpp]   [4] class 3: 0.0000
I [executorch:main.cpp]   [5] class 4: 0.0000
I [executorch:main.cpp]
========================================
I [executorch:main.cpp] MobileNetV2 Demo Complete
I [executorch:main.cpp] Model size: 3490912 bytes
I [executorch:main.cpp] Input: 224x224x3 RGB image (150528 bytes)
I [executorch:main.cpp] Output: 1000 ImageNet classes (top-5 shown)
I [executorch:main.cpp] Inference time: 19 ms
I [executorch:main.cpp] ========================================
```

All predictions show `0.0000` because `mv2_untrained` has random weights.
Use `mv2` (with torchvision pretrained weights) for meaningful class scores.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Linker: `region 'FLASH' overflowed` | Model PTE too large for ITCM | Use the DDR overlay (FVP) or verify mramAddress (Alif) |
| Linker: `region 'RAM' overflowed` | Pools + model copy exceed SRAM | Set `CONFIG_ET_ARM_MODEL_PTE_DMA_ACCESSIBLE=y` to skip the SRAM copy |
| FVP hangs after "Ethos-U backend registered" | Cycle-accurate MV2 simulation is slow | Wait 10-20 min, or use Corstone-320 (faster than 300) |
| No serial output on Alif | Wrong UART or baud rate | Use UART4 at 115200 baud |
| `app-write-mram.py` fails | Wrong mramAddress | Use `0x80008000`, not `0x80200000` |
| Runtime: method allocator OOM | Pool size too small | Increase `CONFIG_EXECUTORCH_METHOD_ALLOCATOR_POOL_SIZE` in the Zephyr config used for the build (for example `prj.conf`, an `OVERLAY_CONFIG`, or a board `.conf`) |

## Memory Layout

| Region | Corstone-320 FVP | Alif E8 |
|--------|-----------------|---------|
| Code + .rodata | ITCM (512 KB) | MRAM |
| .data + .bss + pools | ISRAM (4 MB) | HP SRAM (4.5 MB) |
| Model PTE (~3.5 MB) | DDR (16 MB, via overlay) | MRAM (DMA-accessible) |
| NPU delegation | Ethos-U85 (256 MACs) | Ethos-U55 (256 MACs) |

## Using Claude Code with Zephyr

If you use [Claude Code](https://docs.anthropic.com/en/docs/claude-code), the
ExecuTorch repo ships a `/zephyr` skill that can help with:

- **Workspace setup** — scaffolds the Zephyr workspace, west manifests, and SDK install
- **Board bringup** — generates DTS overlays, board confs, and linker snippets for new boards
- **Memory debugging** — diagnoses linker overflow errors and runtime allocation failures,
  with the exact pool sizes your model needs

Type `/zephyr` in Claude Code while working in the ExecuTorch repo to activate
it. Related skills: `/export` for model conversion, `/cortex-m` for baremetal
Cortex-M builds, `/executorch-kb` for backend-specific debugging.

## Next Steps

- Swap `mv2_untrained` for `mv2` (with torchvision) to get real ImageNet predictions
- Try other models: `resnet18`, or bring your own `.py` model file
- Explore the [hello-executorch sample](https://github.com/pytorch/executorch/tree/main/zephyr/samples/hello-executorch) for a minimal starting point
- See the {doc}`Ethos-U Getting Started tutorial <backends/arm-ethos-u/tutorials/ethos-u-getting-started>` for the baremetal (non-Zephyr) flow
