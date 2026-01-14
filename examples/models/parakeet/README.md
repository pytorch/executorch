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

### Metal Export (macOS)

```bash
python export_parakeet_tdt.py --backend metal --output-dir ./parakeet_metal
```

This generates:
- `parakeet_tdt.pte` - The compiled model
- `aoti_metal_blob.ptd` - Metal kernel blob required at runtime
- `tokenizer.model` - SentencePiece tokenizer

## C++ Runner

### Building

First, build ExecuTorch with the appropriate preset from the executorch root directory:

```bash
# For CPU/XNNPACK
cmake --workflow --preset llm-release

# For Metal (macOS)
cmake --workflow --preset llm-release-metal
```

Then build the parakeet runner:

```bash
cd examples/models/parakeet

# CPU/XNNPACK build
cmake --workflow --preset parakeet-cpu

# Metal build
cmake --workflow --preset parakeet-metal
```

Available presets:
- `parakeet-cpu` - CPU-only build
- `parakeet-cuda` - CUDA acceleration (Linux/Windows)
- `parakeet-metal` - Metal acceleration (macOS)

### Running

From the executorch root directory:

```bash
# CPU/XNNPACK
./cmake-out/examples/models/parakeet/parakeet_runner \
  --model_path examples/models/parakeet/parakeet_tdt_exports/parakeet_tdt.pte \
  --audio_path /path/to/audio.wav \
  --tokenizer_path examples/models/parakeet/parakeet_tdt_exports/tokenizer.model

# Metal (include .ptd data file)
DYLD_LIBRARY_PATH=/usr/lib ./cmake-out/examples/models/parakeet/parakeet_runner \
  --model_path examples/models/parakeet/parakeet_metal/parakeet_tdt.pte \
  --data_path examples/models/parakeet/parakeet_metal/aoti_metal_blob.ptd \
  --audio_path /path/to/audio.wav \
  --tokenizer_path examples/models/parakeet/parakeet_metal/tokenizer.model
```

### Runner Arguments

| Argument | Description |
|----------|-------------|
| `--model_path` | Path to Parakeet model (.pte) |
| `--audio_path` | Path to input audio file (.wav) |
| `--tokenizer_path` | Path to tokenizer file (default: `tokenizer.json`) |
| `--data_path` | Path to data file (.ptd) for delegate data (optional, required for Metal/CUDA) |
| `--timestamps`     | Timestamp output mode: `none\|token\|word\|segment\|all` (default: `segment`) |
