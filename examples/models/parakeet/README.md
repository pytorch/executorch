# Parakeet TDT Export for ExecuTorch

Export [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) speech recognition model to ExecuTorch.

## Installation

```bash
pip install nemo_toolkit[asr] torchaudio
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
| `--backend` | Backend for acceleration: `portable`, `xnnpack`, `cuda`, `cuda-windows` (default: `portable`) |
| `--audio` | Path to audio file for transcription test |

**Note:** The preprocessor is always lowered with the portable backend regardless of the `--backend` setting.

## C++ Runner

### Building

First, build ExecuTorch with the LLM preset from the executorch root directory:

```bash
cmake --workflow --preset llm-release
```

Then build the parakeet runner:

```bash
cd examples/models/parakeet
cmake --workflow --preset parakeet-cpu
```

Available presets:
- `parakeet-cpu` - CPU-only build
- `parakeet-cuda` - CUDA acceleration (Linux/Windows)
- `parakeet-metal` - Metal acceleration (macOS)

### Running

From the executorch root directory:

```bash
./cmake-out/examples/models/parakeet/parakeet_runner \
  --model_path examples/models/parakeet/parakeet_tdt_exports/parakeet_tdt.pte \
  --audio_path /path/to/audio.wav \
  --tokenizer_path examples/models/parakeet/tokenizer.model
```

### Runner Arguments

| Argument | Description |
|----------|-------------|
| `--model_path` | Path to Parakeet model (.pte) |
| `--audio_path` | Path to input audio file (.wav) |
| `--tokenizer_path` | Path to tokenizer file (default: `tokenizer.json`) |
