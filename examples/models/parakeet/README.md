# Parakeet TDT Export for ExecuTorch

Export [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) speech recognition model to ExecuTorch.

## Installation

```bash
pip install -r install_requirements.txt
```

## Export

Export the model:
```bash
python export_parakeet_tdt.py
```

Test transcription on an audio file and compare eager vs lowered results:
```bash
python export_parakeet_tdt.py --audio /path/to/audio.wav
```

### Export Arguments

| Argument | Description |
|----------|-------------|
| `--output-dir` | Output directory for exports (default: `./parakeet_tdt_exports`) |
| `--backend` | Backend for acceleration: `portable`, `xnnpack`, `metal`, `cuda`, `cuda-windows` (default: `portable`) |
| `--dtype` | Data type: `fp32`, `bf16`, `fp16` (default: `fp32`). Metal backend supports `fp32` and `bf16` only (no `fp16`). |
| `--audio` | Path to audio file for transcription test |

**Note:** The preprocessor is always lowered with the portable backend regardless of the `--backend` setting.

### Quantization

The export script supports quantizing encoder and decoder linear layers using [torchao](https://github.com/pytorch/ao).

#### Quantization Arguments

| Argument | Description |
|----------|-------------|
| `--qlinear_encoder` | Quantization config for encoder linear layers: `4w`, `8w`, `8da4w`, `8da8w` |
| `--qlinear_encoder_group_size` | Group size for encoder linear quantization (default: 32) |
| `--qlinear_encoder_packing_format` | Packing format for encoder: `tile_packed_to_4d` |
| `--qlinear` | Quantization config for decoder linear layers: `4w`, `8w`, `8da4w`, `8da8w` |
| `--qlinear_group_size` | Group size for decoder linear quantization (default: 32) |
| `--qlinear_packing_format` | Packing format for decoder: `tile_packed_to_4d` |
| `--qembedding` | Quantization config for decoder embedding layer: `4w`, `8w` |
| `--qembedding_group_size` | Group size for embedding quantization (default: 0 = per-axis) |

#### Quantization Configs

| Config | Description |
|--------|-------------|
| `4w` | 4-bit weight only quantization |
| `8w` | 8-bit weight only quantization |
| `8da4w` | 8-bit dynamic activation, 4-bit weight |
| `8da8w` | 8-bit dynamic activation, 8-bit weight |

#### Example: 4-bit Weight Quantization with Tile Packing

```bash
python export_parakeet_tdt.py \
    --backend cuda \
    --qlinear_encoder 4w \
    --qlinear_encoder_group_size 32 \
    --qlinear_encoder_packing_format tile_packed_to_4d \
    --qlinear 4w \
    --qlinear_group_size 32 \
    --qlinear_packing_format tile_packed_to_4d \
    --qembedding 8w \
    --output-dir ./parakeet_quantized
```

**Note:** The `tile_packed_to_4d` packing format is optimized for CUDA.

### Metal Export (macOS)

```bash
python export_parakeet_tdt.py --backend metal --output-dir ./parakeet_metal
```

This generates:
- `model.pte` - The compiled Parakeet TDT model (includes Metal kernel blob)
- `tokenizer.model` - SentencePiece tokenizer

### CUDA Export (Linux)

```bash
python export_parakeet_tdt.py --backend cuda --output-dir ./parakeet_cuda
```

This generates:
- `model.pte` - The compiled Parakeet TDT model
- `aoti_cuda_blob.ptd` - CUDA kernel blob required at runtime
- `tokenizer.model` - SentencePiece tokenizer

## C++ Runner

### Building

From the executorch root directory:

```bash
# CPU/XNNPACK build
make parakeet-cpu

# Metal build (macOS)
make parakeet-metal

# CUDA build (Linux)
make parakeet-cuda
```

### Running

From the executorch root directory:

```bash
# CPU/XNNPACK
./cmake-out/examples/models/parakeet/parakeet_runner \
  --model_path examples/models/parakeet/parakeet_tdt_exports/model.pte \
  --audio_path /path/to/audio.wav \
  --tokenizer_path examples/models/parakeet/parakeet_tdt_exports/tokenizer.model

# Metal
DYLD_LIBRARY_PATH=/usr/lib ./cmake-out/examples/models/parakeet/parakeet_runner \
  --model_path examples/models/parakeet/parakeet_metal/model.pte \
  --audio_path /path/to/audio.wav \
  --tokenizer_path examples/models/parakeet/parakeet_metal/tokenizer.model

# CUDA (include .ptd data file)
./cmake-out/examples/models/parakeet/parakeet_runner \
  --model_path examples/models/parakeet/parakeet_cuda/model.pte \
  --data_path examples/models/parakeet/parakeet_cuda/aoti_cuda_blob.ptd \
  --audio_path /path/to/audio.wav \
  --tokenizer_path examples/models/parakeet/parakeet_cuda/tokenizer.model
```

### Runner Arguments

| Argument | Description |
|----------|-------------|
| `--model_path` | Path to Parakeet model (.pte) |
| `--audio_path` | Path to input audio file (.wav) |
| `--tokenizer_path` | Path to tokenizer file (default: `tokenizer.json`) |
| `--data_path` | Path to data file (.ptd) for delegate data (required for CUDA) |
| `--timestamps`     | Timestamp output mode: `none\|token\|word\|segment\|all` (default: `segment`) |
