# Nordic AXON NPU Backend for ExecuTorch

ExecuTorch backend for Nordic Semiconductor's **AXON NPU** on the
nRF54LM20B (ARM Cortex-M33 + hardware neural network accelerator).

Compiles PyTorch models to AXON command buffers via TOSA, then executes
them on the NPU at inference time, offloading compute-intensive layers
from the CPU.

## Architecture

```
PyTorch Model
    │
    ▼
torch.export ─── ExecuTorch Edge Lowering
    │
    ▼
AxonPartitioner ─── identifies ops for AXON delegation
    │
    ▼
TOSABackend._preprocess() ─── shared ARM TOSA lowering
    │
    ▼
tosa_reader ─── parse TOSA flatbuffer
    │
    ▼
axon_compiler ─── convert TOSA layers to AXON layer descriptors
    │
    ▼
axon_binary ─── pack intermediate binary (cffi structs)
    │
    ▼
Nordic compiler lib ─── produce AXON command buffers (.h headers)
    │
    ▼
.pte file + generated headers ─── deploy to nRF54LM20DK
```

## Supported Hardware

| Device | NPU | Status |
|--------|-----|--------|
| nRF54LM20DK (nRF54LM20B) | AXON (~300 MACs, 3-8 GOPS) | Supported |

## Supported Operations

### AXON-accelerated (hardware)

| Operation | Max dimensions | Notes |
|-----------|---------------|-------|
| Fully Connected | 2048 in/out | INT8 weights + bias |
| Conv2D | 16x16 filter, stride ≤ 31 | INT8, with padding |
| Depthwise Conv2D | 16x16 filter | INT8 |
| Average Pool 2D | 32x32 filter | |
| Max Pool 2D | 32x32 filter | |
| Add | element-wise | INT8 |
| Multiply | element-wise | INT8 |
| ReLU / ReLU6 | fused with preceding layer | Zero overhead |
| Leaky ReLU | fused with preceding layer | |

### Op extensions (AXON + CPU hybrid)

| Operation | Preceding layer output | CPU callback |
|-----------|----------------------|--------------|
| Sigmoid | INT16 q3.12 | `axon_op_extension_sigmoid` |
| Tanh | INT16 q3.12 | `axon_op_extension_tanh` |
| Softmax | INT32 q11.12 | Nordic's reference implementation |

## Prerequisites

- **nRF54LM20DK**: Nordic's development kit with the AXON NPU.
- **nRF Connect SDK (NCS)**: Nordic's Zephyr-based SDK. Install via
  [nRF Connect for Desktop](https://www.nordicsemi.com/Products/Development-tools/nRF-Connect-for-Desktop)
  or manually:
  ```bash
  # Install NCS toolchain
  nrfutil sdk-manager install --ncs-version v3.3.0-preview3

  # Initialize west workspace
  west init -m https://github.com/nrfconnect/sdk-nrf --mr v3.3.0-preview3 ~/ncs-workspace
  cd ~/ncs-workspace && west update

  # Generate toolchain environment script
  nrfutil sdk-manager toolchain env --as-script sh --ncs-version v3.3.0-preview3 > nrf-connect-sdk-env.sh
  ```
- **Nordic sdk-edge-ai**: Contains the AXON compiler library (proprietary,
  not redistributed). Available to nRF54LM20DK owners via Nordic's
  [Edge AI documentation](https://docs.nordicsemi.com/bundle/addon-edge-ai_latest/page/index.html).
  ```bash
  git clone <sdk-edge-ai-url> ~/sdk-edge-ai
  export SDK_EDGE_AI_PATH=~/sdk-edge-ai
  ```
- **ExecuTorch**: This repository (with the ARM TOSA backend).
- **Python packages** (for model export, separate from NCS Python):
  ```bash
  pip install cffi numpy tosa-tools
  ```

Verify the setup:
```bash
bash backends/nordic/scripts/setup.sh
```

## Quick Start

### One-command flow

```bash
# Source NCS toolchain
source ~/ncs-workspace/nrf-connect-sdk-env.sh

# Export model, build firmware, flash — all in one
./backends/nordic/scripts/run.sh
```

### Step-by-step flow

#### Step 1: Export a model

```bash
# Use a Python environment with ExecuTorch installed (NOT the NCS Python)
python examples/nordic/hello_axon/export_model.py
```

This trains a small sin(x) model, quantizes it to INT8, compiles the
FC layers to AXON command buffers, and produces:
- `build/hello_axon.pte` — ExecuTorch program
- `src/model_pte.h` — embedded model as C array
- `src/generated/axon_subgraph_*.h` — AXON command buffers
- `src/generated/axon_subgraphs_table.h` — delegate lookup table

### Step 2: Build firmware

```bash
# Source the NCS toolchain environment
source ~/ncs-workspace/nrf-connect-sdk-env.sh

# Build for nRF54LM20DK
west build -b nrf54lm20dk/nrf54lm20b/cpuapp examples/nordic/hello_axon \
  --no-sysbuild -- \
  -DZEPHYR_EXTRA_MODULES="$(pwd);$SDK_EDGE_AI_PATH"
```

### Step 3: Flash and verify

```bash
west flash

# Open serial console (115200 baud)
# Linux: screen /dev/ttyACM0 115200
# macOS: screen /dev/cu.usbmodem* 115200
```

Expected output:
```
Hello AXON - ExecuTorch + Nordic AXON NPU
AXON NPU: enabled
Loading model (2084 bytes)...
AxonBackend::init (delegate 0, processed=36 bytes)
  AXON model 'hello_axon_...' bound (out: 1x1x1 byte_width=1)
Running inference (x=1.57, expected sin~1.0)...
Inference: 20871 cycles (163 us @ 128 MHz)
  output[0] = 0.997794
Done.
```

### Using the AXON backend in your own model

```python
from executorch.backends.nordic.axon import (
    AxonQuantizer,
    AxonCompileSpec,
    AxonPartitioner,
)
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig

# 1. Quantize
quantizer = AxonQuantizer()  # Symmetric INT8, per-channel weights
exported = torch.export.export(model.eval(), example_input, strict=False)
prepared = prepare_pt2e(exported.module(), quantizer)
prepared(*calibration_data)  # Calibrate
quantized = convert_pt2e(prepared)
re_exported = torch.export.export(quantized, example_input, strict=False)

# 2. Partition to AXON
compile_spec = AxonCompileSpec(
    model_name="my_model",
    axon_generated_dir="build/generated",
)
partitioner = AxonPartitioner(compile_spec)
edge = to_edge_transform_and_lower(
    re_exported,
    partitioner=[partitioner],
    compile_config=EdgeCompileConfig(_check_ir_validity=False),
)

# 3. Save .pte
edge.to_executorch().save("build/my_model.pte")
```

## Directory Structure

```
backends/nordic/
├── axon/                    # Core backend
│   ├── backend.py           # AxonBackend (BackendDetails)
│   ├── compile_spec.py      # AxonCompileSpec
│   ├── partitioner.py       # AxonPartitioner (extends TOSAPartitioner)
│   └── codegen.py           # Marker format, naming, header generation
├── axon_compiler.py         # TOSA → AXON layer conversion
├── axon_binary.py           # Binary builder for Nordic compiler
├── tosa_reader.py           # TOSA flatbuffer parser
├── operator_support/        # AXON hardware constraint checks
├── runtime/                 # C++ delegate for on-device execution
│   ├── AxonBackend.cpp      # BackendInterface implementation
│   ├── AxonBackend.h        # Profiling API
│   └── axon_op_extensions.c # Sigmoid/tanh CPU callbacks
├── test/                    # Unit tests (23 tests, no SDK required)
├── CMakeLists.txt           # Build configuration
└── README.md                # This file
```

## How It Works

The backend follows the same **composition pattern** as the Ethos-U backend:

1. **Partitioner** identifies INT8-quantized operations that AXON supports
2. **TOSABackend** (shared with Ethos-U) lowers the subgraph to TOSA IR
3. **AXON compiler** converts TOSA → AXON layer descriptors → intermediate binary
4. **Nordic compiler lib** (external) produces command buffers as C headers
5. **C++ delegate** on-device parses the marker, looks up the compiled model,
   and calls `nrf_axon_nn_model_infer_sync()` for hardware execution

Multi-subgraph models are supported: each delegated subgraph gets a unique
name (content-hash based), and a generated lookup table maps names to compiled
models at runtime.

## Running Tests

```bash
# TOSA lowering tests (no SDK required)
pytest backends/nordic/test/test_tosa_lowering.py -v

# Full compilation tests (requires SDK_EDGE_AI_PATH)
pytest backends/nordic/test/test_axon_compile.py -v
```

## Tutorials and Examples

See [ioteai/axon-ai](https://github.com/ioteai/axon-ai) for Jupyter
notebooks, Docker setup, and detailed guides.

## License

Copyright (c) 2026 iote.ai. BSD 3-Clause License — see the root
[LICENSE](../../LICENSE) file.
