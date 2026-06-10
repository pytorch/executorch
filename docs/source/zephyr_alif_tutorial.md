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
- Python 3.12+
- Alif SE Tools for flashing (Alif hardware only)

## Step 1: Set Up the Zephyr Workspace

Create a workspace, install `west`, and initialize the Zephyr tree:

```bash
mkdir ~/zephyr_workspace && cd ~/zephyr_workspace
python3 -m venv .venv && source .venv/bin/activate
pip install west "cmake<4.0.0" pyelftools ninja jsonschema
west init --manifest-rev v4.4.0
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
west packages pip --install
```

For Alif boards, also add the Alif HAL:

```bash
west config manifest.project-filter -- -.*,+zephyr,+executorch,+cmsis,+cmsis_6,+cmsis-nn,+hal_ethos_u,+hal_alif
west update
west packages pip --install
```

Install the Zephyr SDK (compiler toolchain):

```bash
west sdk install --gnu-toolchains arm-zephyr-eabi
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

**For Alif E8 (HP Ethos-U55 with 256 MACs):**

```bash
python -m modules.lib.executorch.backends.arm.scripts.aot_arm_compiler \
    --model_name=mv2 \
    --quantize --delegate \
    --target=ethos-u55-256 \
    --output=mv2_ethosu.pte
```

If rtss_he is used instead of rtss_hp below use `--target=ethos-u55-128` to match the hardware.

**For Corstone-320 FVP (Ethos-U85 with 256 MACs):**

```bash
python -m modules.lib.executorch.backends.arm.scripts.aot_arm_compiler \
    --model_name=mv2 \
    --quantize --delegate \
    --target=ethos-u85-256 \
    --output=mv2_u85_256.pte
```

The `--delegate` flag routes all compatible ops through the Ethos-U backend.
The Vela compiler converts the TOSA intermediate representation into an
optimized command stream for the NPU. Use `mv2_untrained` instead of `mv2` if
you do not want to depend on torchvision pretrained weights.

## Step 5: Build the Zephyr Application

**For Alif E8:**

```bash
west build -d build -b ensemble_e8_dk/ae822fa0e5597ls0/rtss_hp \
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

### Flash with west

```bash
west flash
```


### Flash with Alif SE Tools

If you do not want to use J-Link to write a raw image, or if `west flash` does
not work on your board, use Alif SE Tools to program the E8 MRAM through the
Secure Enclave Application Table of Contents (ATOC) flow.

Use a SE Tools release that matches the SES firmware printed on the SE UART at
reset. For example, if the boot log says `SES A0 v1.109.0`, download and use the `1.109`
SE Tools directory.

The Alif E8 sample build generates the binary consumed by SE Tools:

- `build/zephyr/zephyr.bin`

Create `build/images/zephyr.json` and `build/images/app-device-config.json` in
the SE Tools directory:

```bash
ZEPHYR_BUILD=~/zephyr_workspace/build/zephyr

cd <PATH_TO_ALIF_SETOOLS>/app-release-exec-linux_FW_1.109.00_DEV

ln -sf "$ZEPHYR_BUILD/zephyr.bin" build/images/zephyr.bin

cat > build/images/zephyr.json << 'EOF'
{
    "DEVICE": {
        "binary": "app-device-config.json",
        "version": "0.5.0",
        "signed": true
    },
    "MV2": {
        "binary": "zephyr.bin",
        "version": "1.0.0",
        "mramAddress": "0x80008000",
        "cpu_id": "M55_HP",
        "flags": ["boot"],
        "signed": false
    }
}
EOF

cat > build/images/app-device-config.json << 'EOF'
{
  "metadata": {
    "device": "AE822FA0E5597LS0",
    "project": "",
    "external_clock_sources": [
      {
        "id": "OSC_HFXO",
        "enabled": true,
        "frequency": 38400000
      },
      {
        "id": "OSC_LFXO",
        "enabled": false,
        "frequency": 32768
      }
    ]
  },
  "firewall": {"firewall_components": []},
  "pinmux": {"configurations": []},
  "miscellaneous": []
}
EOF
```

Make sure `cpu_id` matches your target: use `M55_HP` for
`ensemble_e8_dk/ae822fa0e5597ls0/rtss_hp` and `M55_HE` for
`ensemble_e8_dk/ae822fa0e5597ls0/rtss_he`.

`zephyr.json` places the image at `0x80008000`, matching the board config's
`CONFIG_FLASH_LOAD_OFFSET=0x8000`:

> **Important:** Keep `mramAddress: "0x80008000"` and
> `CONFIG_FLASH_LOAD_OFFSET=0x8000` in sync. If these differ, SES may accept the
> ATOC but the M55 image will not start correctly.

Generate the ATOC package from the SE Tools directory. The executable releases
expect image files under their own `build/images` directory, so stage the Zephyr
outputs there with symlinks:

```bash
./app-gen-toc -f "build/images/zephyr.json"
```

Note that some versions use app-gen-toc.py

This creates `build/AppTocPackage.bin` in the SE Tools directory.


The DK-E8 routes the J-Link VCOM to different UARTs using `SW4`:

- Set `SW4=SE` for SE Tools and the Secure Enclave boot log at 57600 baud.
- Set `SW4=U2` for the Zephyr RTSS-HE console at 115200 baud.
- Set `SW4=U4` for the Zephyr RTSS-HP console at 115200 baud.

To flash set `SW4=SE` and run

```bash
./app-write-mram
```
Note that some versions use app-write-mram.py

On some boards the running application prevents the ISP handshake. In that case,
enter hard maintenance mode first. Keep `SW4=SE`, close any serial terminal, and
use the same baud rate as the SE boot log (`57600` on DK-E8):

```bash
./maintenance -c /dev/ttyACM0 -b 57600
```

Select:

```text
1 - Device Control
1 - Hard maintenance mode
```

When the tool prints `Waiting for Target..[RESET Platform]`, press the physical
reset button on the board. Wait for the tool to return to the menu.

Verify that hard maintenance really succeeded before flashing. From the returned
menu, select:

```text
Enter - return to the top-level menu
2 - Device Information
4 - Device enquiry
```

Continue only if the device enquiry reports `Maintenance Mode = Enabled`. The
maintenance tool may still show menus after `Device did not respond`; that does
not mean the target is connected. If device enquiry also reports `Target did not
respond`, power-cycle the board, keep `SW4=SE`, press reset, and repeat the hard
maintenance step.

After `Maintenance Mode = Enabled`, exit the maintenance tool and flash without
resetting out of maintenance mode:

```bash
./app-write-mram -b 57600 -nr -s
```

If prompted for the serial port, enter the J-Link VCOM port, for example
`/dev/ttyACM0`.

You can verify the flash and boot if needed with `SW4` set to `SE`, connected
to the SE boot UART at 57600 baud:

```bash
picocom -b 57600 /dev/ttyACM0
```

Press the reset button on the E8 DevKit.
The package has been accepted and booted when you see `ATOC ok` and a table row
for `M55-HP` or `M55-HE` whose flags include `B`.

### Connect Serial Console

After flashing, make sure `SW4` is set to `U4` for RTSS-HP or `U2` for RTSS-HE,
open the serial terminal, and press reset:

```bash
picocom -b 115200 /dev/ttyACM0
```

Press the reset button on the E8 DevKit. You should see:

```
*** Booting Zephyr OS build v4.4.0 ***

========================================
ExecuTorch MobileNetV2 Classification Demo
========================================

I [executorch:main.cpp] Ethos-U backend registered successfully
I [executorch:main.cpp] Model PTE at 0x800774a0, Size: 3380576 bytes
I [executorch:main.cpp] Model loaded, has 1 methods
I [executorch:main.cpp] Running method: forward
I [executorch:main.cpp] Method allocator pool size: 1572864 bytes.
I [executorch:main.cpp] Setting up planned buffer 0, size 752640.
I [executorch:main.cpp] Loading method...
I [executorch:main.cpp] Method 'forward' loaded successfully
I [executorch:main.cpp] Preparing input: static RGB image (150528 bytes)
I [executorch:main.cpp] Input tensor: scalar_type=Float, numel=150528, nbytes=602112
I [executorch:main.cpp] Converting uint8 input (150528 elements) to float32
I [executorch:main.cpp]
--- Starting inference ---
I [executorch:main.cpp] Inference completed in 30 ms
I [executorch:main.cpp]
--- Classification Results ---
Top-5 predictions:
I [executorch:main.cpp]   [1] class 92: 3.2362
I [executorch:main.cpp]   [2] class 21: 3.0362
I [executorch:main.cpp]   [3] class 127: 2.8180
I [executorch:main.cpp]   [4] class 22: 2.5998
I [executorch:main.cpp]   [5] class 18: 2.5089
I [executorch:main.cpp]
========================================
I [executorch:main.cpp] MobileNetV2 Demo Complete
I [executorch:main.cpp] Model size: 3380576 bytes
I [executorch:main.cpp] Input: [1, 3, 224, 224] NCHW RGB tensor (150528 bytes)
I [executorch:main.cpp] Output: 1000 classes (top-5 shown)
I [executorch:main.cpp] Inference time: 30 ms
I [executorch:main.cpp] ========================================
```

Use `mv2_untrained` instead of `mv2` if you want to avoid downloading
pretrained weights. In that case the class scores are arbitrary and may all be
close to `0.0000`.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Linker: `region 'FLASH' overflowed` | Model PTE too large for ITCM | Use the DDR overlay (FVP) or verify mramAddress (Alif) |
| Linker: `region 'RAM' overflowed` | Pools + model copy exceed SRAM | Set `CONFIG_ET_ARM_MODEL_PTE_DMA_ACCESSIBLE=y` to skip the SRAM copy |
| FVP hangs after "Ethos-U backend registered" | Cycle-accurate MV2 simulation is slow | Wait 10-20 min, or use Corstone-320 (faster than 300) |
| No Zephyr serial output on Alif | `SW4` still routes VCOM to SE UART | Move `SW4` to `U4` and use 115200 baud |
| No SE Tools response | Serial terminal is open, wrong `SW4` setting, or app blocks ISP | Close terminal, set `SW4=SE`, or enter hard maintenance mode |
| `app-write-mram` fails or app does not boot | `mramAddress` and link address differ | Use `0x80008000`, not `0x80200000` with `CONFIG_FLASH_LOAD_OFFSET=0x8000` |
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

- Try other models: `resnet18`, or bring your own `.py` model file
- Explore the [hello-executorch sample](https://github.com/pytorch/executorch/tree/main/zephyr/samples/hello-executorch) for a minimal starting point
- See the {doc}`Ethos-U Getting Started tutorial <backends/arm-ethos-u/tutorials/ethos-u-getting-started>` for the baremetal (non-Zephyr) flow
