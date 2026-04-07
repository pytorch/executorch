# Config Generator Python

Python tools for extracting convolution layer parameters from neural network models and generating optimized C header configurations for DMA-tiled execution on the Xtensa XRC Vision DSP (XAI CNN runtime).

## Quick Start

```bash
# From an ExecuTorch .pte binary (no pip install required)
python generate_layer_configs.py --pte resnet18_quantized.pte \
    --output conv_layer_configs_62k5.h --dram0 31250 --dram1 31250
    
# working cmd
# python3 backends/cadence/vision/config_generator/generate_layer_configs.py   --pte /home/sraut/ext_test/executorch/operator_and_model_testing/resnet50/pte/resnet50_quantized.pte   --output backends/cadence/vision/config_generator/test_32k.h --dram0 31250 --dram1 31250 

# From a torchvision model
python generate_layer_configs.py --model resnet18 --input-size 1,3,64,64 \
    --output conv_layer_configs.h --dram0 32768 --dram1 32768
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
