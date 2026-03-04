# Config Generator Python

Python tools for extracting convolution layer parameters from neural network models and generating optimized C header configurations for DMA-tiled execution on the Xtensa XRC Vision DSP (XAI CNN runtime).

## Quick Start

### From a torchvision model (ResNet-18 + ResNet-50)

```bash
python generate_layer_configs.py --model resnet18+resnet50 --input-size 1,3,64,64 \
    --output conv_layer_configs.h --dram0 32768 --dram1 32768
```

### From an existing CSV

```bash
python generate_layer_configs.py resnet18_conv_list.csv \
    --output conv_layer_configs.h --dram0 32768 --dram1 32768
```

### From a `.pte` or `.onnx` model

```bash
python extract_layers_from_pte.py model.pte --output layers_config.json
python generate_layer_configs.py layers_config.json --dram0 32768 --dram1 32768
```

This reads convolution parameters, calculates optimal IDMA buffer tiling for the specified DRAM sizes, and writes the resulting C lookup table to `conv_layer_configs.h`.

---

## Scripts

| Script | Description |
|--------|-------------|
| `generate_layer_configs.py` | **Main entry point.** Reads layer parameters from a model, CSV, or JSON; computes buffer/tiling configs; and emits `conv_layer_configs.h` with a struct lookup table. |
| `generate_idma_buffers.py` | Core buffer sizing and tiling engine. Calculates IDMA ping-pong buffer sizes, optimal tile configurations, and DRAM0/DRAM1 placement strategies. |
| `csv_to_config.py` | Converts a ResNet-style CSV convolution list into Python config dictionaries. |
| `extract_layers_from_pte.py` | Extracts convolution layer parameters from ExecuTorch `.pte` or ONNX `.onnx` model files (outputs JSON). |
| `extract_conv_params.py` | Lower-level model parameter extraction supporting `.pte`, `.onnx`, `.pt`, and `.pth` files. |
| `demo_generate_configs.py` | End-to-end demo that runs `generate_layer_configs.py` on `resnet18_conv_list.csv`. |

---

## Detailed Usage

### `generate_layer_configs.py`

The main script supports three input sources: torchvision models, CSV files, or JSON files.

**Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `input_file` (positional) | — | Input `.csv` or `.json` file (not needed with `--model`) |
| `--model`, `-m` | — | Comma or `+`-separated model names (e.g. `resnet18`, `resnet50`, `resnet18+resnet50`) |
| `--input-size` | `1,3,64,64` | Model input tensor shape `N,C,H,W` |
| `--output`, `-o` | `conv_layer_configs.h` | Output C header file path |
| `--dram0` | `32768` | DRAM0 size in bytes |
| `--dram1` | `32768` | DRAM1 size in bytes |

**What it generates:** A C header containing:
- `conv_layer_config_t` struct typedef with ~60+ fields (buffer sizes, tile dimensions, DRAM placements, kernel name, quantization params)
- `CONV_LAYER_CONFIGS[]` static array with one entry per unique layer
- Accessor functions: `get_layer_config()`, `get_layer_config_by_params()`, `get_layer_config_by_key()`

### `generate_idma_buffers.py`

Core tiling engine used internally by `generate_layer_configs.py`.

**Key functions:**

| Function | Description |
|----------|-------------|
| `find_max_tile_config(...)` | Finds the maximum `n_tile_size` and `output_rows` that fit in DRAM. Two-phase search: Phase 1 scans tile sizes for best balanced config; Phase 2 maximizes output rows. |
| `calculate_buffer_sizes_with_rows(...)` | Calculates all buffer sizes (input, coeff, output, bias, outscale) and tile dimension parameters for a given output-row count. |
| `calculate_buffer_placement(...)` | Optimal ping-pong buffer placement across DRAM0/DRAM1. Tries 6 strategies and picks the one with best utilization. All sizes aligned to 64 bytes. |
| `calculate_conv_params(...)` | Calculates quantized convolution parameters (output_shift, output_scale, relu_max, etc.) |

**Constants:** `DRAM_SIZE_0 = 32768`, `DRAM_SIZE_1 = 32768` (defaults)

### `csv_to_config.py`

Standalone utility for converting CSV convolution lists to Python config dicts.

```bash
python csv_to_config.py resnet18_conv_list.csv              # Print all layers
python csv_to_config.py resnet18_conv_list.csv --unique      # Unique kernel/stride combos only
python csv_to_config.py resnet18_conv_list.csv -o configs.py # Write to file
```

**Expected CSV format:** Tab-delimited with columns: `(index)`, `input`, `kernel`, `stride`, `padding`, `dilation`, `transposed`, `output_padding`, `groups`, `output`. Shape values are comma-separated (e.g. `1,64,112,112`).

### `extract_layers_from_pte.py`

Extracts conv layers from ExecuTorch `.pte` or ONNX `.onnx` models.

```bash
python extract_layers_from_pte.py model.pte -o layers.json --format json
python extract_layers_from_pte.py model.onnx -o layers.json
```

| Flag | Default | Description |
|------|---------|-------------|
| `model_file` (positional) | required | Path to `.pte` or `.onnx` model |
| `--output`, `-o` | `layers_config.json` | Output file path |
| `--format` | `json` | Output format: `json` or `python` |

### `extract_conv_params.py`

Lower-level extraction supporting multiple model formats with auto-detection.

```bash
python extract_conv_params.py model.onnx -o conv_configs.py -f auto
```

| Flag | Default | Description |
|------|---------|-------------|
| `model_path` (positional) | required | Path to model file |
| `--output`, `-o` | `conv_configs.py` | Output config file |
| `--format`, `-f` | `auto` | Model format: `pte`, `onnx`, `pytorch`, or `auto` |

---

## Typical Workflow

1. **Obtain a convolution list** – export from a model or use an existing CSV.
2. **Generate the config header** – run `generate_layer_configs.py` with desired DRAM sizes.
3. **Include in C code** – add `#include "conv_layer_configs.h"` and the runtime dispatcher calls `conv_execute_kernel()` using the config struct.

```
  Model (.pte / .onnx / torchvision)
       │
       ▼
  extract_layers_from_pte.py  ──→  layers_config.json
       │                                   │
       │         CSV file ─────────────────┤
       │                                   │
       ▼                                   ▼
  generate_layer_configs.py  ◄── generate_idma_buffers.py
       │
       ▼
  conv_layer_configs.h  ──→  C runtime (conv_kernel_dispatcher.c)
```

---

## Output Format

The generated `conv_layer_configs.h` contains:

- **Struct definition** (`conv_layer_config_t`): buffer sizes, tile dimensions, DRAM0/DRAM1 placement, kernel executor name, stride, padding, quantization parameters, input zero-point.
- **Config array** (`CONV_LAYER_CONFIGS[]`): one entry per unique convolution layer configuration.
- **Lookup functions**: match layers by index, by `(ic, oc, kh, kw, stride, oh, ow)` parameters, or by a string key.

---

## Directory Structure

```
config_generator_python/
├── generate_layer_configs.py    # Main entry point
├── generate_idma_buffers.py     # Core tiling engine
├── csv_to_config.py             # CSV → config dicts
├── extract_layers_from_pte.py   # .pte/.onnx → JSON
├── extract_conv_params.py       # Low-level model extraction
├── demo_generate_configs.py     # End-to-end demo
├── bin/
│   ├── compare_configs.exe      # Config comparison tool
│   └── test_conv_layer_lookup.exe  # Lookup test tool
└── README.md
```
