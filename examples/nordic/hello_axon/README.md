# Hello AXON — ExecuTorch + Nordic AXON NPU

Minimal example: train a PyTorch model, compile it for the AXON NPU,
and run inference on the nRF54LM20DK.

## What it does

1. Trains a small 3-layer FC model to approximate sin(x)
2. Quantizes to INT8 via ExecuTorch PT2E
3. Delegates FC layers to the AXON NPU backend
4. Exports as `.pte` + AXON command buffer headers
5. Builds Zephyr firmware with the model embedded
6. Runs inference on the nRF54LM20DK — AXON NPU executes the FC layers

## Two Python environments

This example uses **two separate Python environments** for different
stages. This is necessary because the nRF Connect SDK (NCS) ships its
own Python (3.12) with its own `PYTHONHOME` and `PYTHONPATH`, which
conflict with PyTorch and ExecuTorch's Python packages.

| Stage | Python | Why |
|-------|--------|-----|
| **Model export** (`export_model.py`) | Your own Python (3.10+) via `uv` | Needs PyTorch, ExecuTorch, tosa-tools — packages that don't exist in the NCS Python |
| **Firmware build** (`west build`) | NCS toolchain Python (3.12) | Needs Zephyr's cmake modules and the NCS build system |

The `setup_export_env.sh` script creates an isolated `.venv/` in this
directory with all export dependencies. It uses
[uv](https://docs.astral.sh/uv/) to manage the environment. The NCS
toolchain Python is used only by `west build` and is activated by
sourcing `nrf-connect-sdk-env.sh`.

**Important:** Do not source `nrf-connect-sdk-env.sh` in the same
terminal where you run `export_model.py`. The NCS environment sets
`PYTHONHOME` which overrides Python's standard library path and causes
import errors in any non-NCS Python. The `run_export.sh` wrapper
handles this automatically by unsetting `PYTHONHOME` before invoking
`uv`.

## Prerequisites

- **nRF54LM20DK** development kit
- **nRF Connect SDK (NCS)** installed — provides `west`, Zephyr, and
  the ARM cross-compiler. See [Nordic's install guide](https://docs.nordicsemi.com/bundle/ncs-latest/page/nrf/installation.html).
- **Nordic sdk-edge-ai** — contains the AXON compiler library.
  Set `SDK_EDGE_AI_PATH` to its location.
- **uv** — Python package manager. Install with `pip install uv`.

## Step-by-step

### 1. Set up the export environment (one time)

```bash
cd examples/nordic/hello_axon
./setup_export_env.sh
```

This creates `.venv/` with PyTorch (CPU), ExecuTorch, tosa-tools, and
torchao. It also generates `run_export.sh` — a wrapper that sets the
correct `PYTHONPATH` for ExecuTorch.

### 2. Export the model

```bash
SDK_EDGE_AI_PATH=~/sdk-edge-ai ./run_export.sh
```

This trains sin(x), quantizes to INT8, compiles FC layers to AXON
command buffers, and produces:

| Output | Description |
|--------|-------------|
| `build/hello_axon.pte` | ExecuTorch program file |
| `src/model_pte.h` | Model embedded as a C array (16-byte aligned) |
| `src/generated/axon_subgraph_*.h` | AXON command buffers per layer |
| `src/generated/axon_subgraphs_table.h` | Delegate lookup table |

### 3. Build firmware

Open a **new terminal** (or unrelated to the export step), then:

```bash
# Activate the NCS toolchain (provides west, arm-zephyr-eabi-gcc, cmake)
source ~/ncs-workspace/nrf-connect-sdk-env.sh

# Build from the executorch root directory
cd <executorch-root>
west build -b nrf54lm20dk/nrf54lm20b/cpuapp examples/nordic/hello_axon \
    --no-sysbuild -- \
    -DZEPHYR_EXTRA_MODULES="$(pwd);$SDK_EDGE_AI_PATH"
```

### 4. Flash and verify

```bash
west flash

# Serial console (115200 baud):
#   Linux:  screen /dev/ttyACM0 115200
#   macOS:  screen /dev/cu.usbmodem* 115200
```

### Expected output

```
Hello AXON - ExecuTorch + Nordic AXON NPU
Board: nrf54lm20dk/nrf54lm20b/cpuapp
AXON NPU: enabled
Loading model (2084 bytes)...
Program loaded, 1 method(s)
Method: forward
AxonBackend::init (delegate 0, processed=36 bytes)
  AXON model 'hello_axon_...' bound (out: 1x1x1 byte_width=1)
Method loaded
Running inference (x=1.57, expected sin~1.0)...
Inference: 20876 cycles (163 us @ 128 MHz)
  output[0] = 0.987485
Done.
```

## Architecture

```
 setup_export_env.sh          (Python venv with PyTorch + ExecuTorch)
         |
         v
 run_export.sh                (unsets PYTHONHOME, sets PYTHONPATH)
         |
         v
 export_model.py              (train → quantize → AXON compile → .pte)
         |
         +-- build/hello_axon.pte
         +-- src/model_pte.h              (embedded C array)
         +-- src/generated/axon_subgraph_*.h   (AXON command buffers)
         +-- src/generated/axon_subgraphs_table.h

 nrf-connect-sdk-env.sh       (NCS toolchain Python + west + compiler)
         |
         v
 west build                   (Zephyr + ExecuTorch + AXON delegate)
         |
         v
 zephyr.hex                   (firmware with model + runtime)
         |
         v
 nRF54LM20DK                  (AXON NPU executes FC layers)
```

## File structure

```
hello_axon/
├── setup_export_env.sh       # One-time: create .venv with export deps
├── run_export.sh             # Generated: export wrapper (sets PYTHONPATH)
├── export_model.py           # Train, quantize, export sin(x) model
├── pyproject.toml            # Python project (base deps for uv)
├── CMakeLists.txt            # Zephyr firmware build
├── prj.conf                  # Zephyr project config
├── boards/
│   └── nrf54lm20dk_...conf   # AXON NPU board config
├── src/
│   ├── main.c                # Entry point (pure C)
│   ├── inference.cpp         # ExecuTorch runtime (C++)
│   ├── model_pte.h           # Generated: embedded .pte
│   └── generated/            # Generated: AXON command buffers
├── build/                    # Build output (gitignored)
└── .venv/                    # Export Python env (gitignored)
```
