# Config Generator Python

Python tools for extracting convolution layer parameters from neural network models and generating optimized C header configurations for DMA-tiled execution on the Xtensa XRC Vision DSP (XAI CNN runtime).

## Prerequisites

The script requires the Python venv in the executorch tree and must be run from a **bash** terminal (not csh):

```bash
# The venv is at <executorch>/.venv/
# All paths below are relative to the executorch root.

# Option 1: call the venv python directly (works from any shell)
.venv/bin/python3 backends/cadence/vision/config_generator/generate_layer_configs.py ...

# Option 2: activate the venv in a bash shell
bash
source .venv/bin/activate
python3 backends/cadence/vision/config_generator/generate_layer_configs.py ...
```

> **Note:** The default terminal on this machine is `csh`. Inline python commands
> and `source ... && ...` chains will fail in csh. Always use `bash` or invoke
> the venv python by its full path.

## Quick Start

```bash
# Run from the executorch root directory: cd <executorch>

# From a single ExecuTorch .pte binary
.venv/bin/python3 backends/cadence/vision/config_generator/generate_layer_configs.py \
    --pte operator_and_model_testing/resnet18/pte/resnet18_quantized.pte \
    --output backends/cadence/vision/config_generator/conv_layer_configs.h \
    --dram0 62976 --dram1 62976

# From multiple .pte files (layers are deduplicated automatically)
.venv/bin/python3 backends/cadence/vision/config_generator/generate_layer_configs.py \
    --pte operator_and_model_testing/resnet18/pte/resnet18_quantized.pte \
         operator_and_model_testing/resnet50/pte/resnet50_quantized.pte \
    --output backends/cadence/vision/config_generator/conv_layer_configs_combined.h \
    --dram0 62976 --dram1 62976

# From a torchvision model (requires torchvision installed in venv)
.venv/bin/python3 backends/cadence/vision/config_generator/generate_layer_configs.py \
    --model resnet18 --input-size 1,3,64,64 \
    --output backends/cadence/vision/config_generator/conv_layer_configs.h \
    --dram0 32768 --dram1 32768
```

### Full working commands

```bash
# cd to the executorch root first
cd <path-to-executorch>

# ResNet18 with 62976 bytes per DRAM bank
.venv/bin/python3 backends/cadence/vision/config_generator/generate_layer_configs.py \
    --pte operator_and_model_testing/resnet18/pte/resnet18_quantized.pte \
    --output backends/cadence/vision/config_generator/conv_layer_configs_62k_pte.h \
    --dram0 62976 --dram1 62976

# ResNet18 + ResNet50 combined
.venv/bin/python3 backends/cadence/vision/config_generator/generate_layer_configs.py \
    --pte operator_and_model_testing/resnet18/pte/resnet18_quantized.pte \
         operator_and_model_testing/resnet50/pte/resnet50_quantized.pte \
    --output backends/cadence/vision/config_generator/conv_layer_configs_62k_combined.h \
    --dram0 62976 --dram1 62976
```

---

## `generate_layer_configs.py` — Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--model`, `-m` | — | Comma or `+`-separated torchvision model names (e.g. `resnet18+resnet50`) |
| `--pte` | — | Path to an ExecuTorch `.pte` binary; bootstraps `exir._serialize` from the local source tree — no pip install needed |
| `--flatc` | cmake-out default | Path to `flatc` binary (auto-detected; only relevant with `--pte`) |
| `--input-size` | `1,3,64,64` | Input tensor shape `N,C,H,W` (only used with `--model`) |
| `--output`, `-o` | `conv_layer_configs.h` | Output C header file |
| `--dram0` | `32768` | DRAM0 size in bytes |
| `--dram1` | `32768` | DRAM1 size in bytes |
| `--cache-mode` | off | Append `_cache` to every kernel name |

---

## Output

The generated header contains:

- `conv_layer_config_t` struct with ~60 fields (buffer sizes, tile dimensions, DRAM0/1 placement, kernel name, quantization params)
- `CONV_LAYER_CONFIGS[]` static array — one entry per unique layer
- `get_layer_config()`, `get_layer_config_by_params()`, `get_layer_config_by_key()` inline accessors

---

## Directory Structure

```
config_generator_python/
├── generate_layer_configs.py    # Main entry point
├── generate_idma_buffers.py     # Core tiling / buffer sizing engine
├── extract_layers_from_pte.py   # .pte/.onnx → JSON (intermediate step)
├── config/                      # Pre-generated headers
└── bin/                         # Compare / test utilities
```
