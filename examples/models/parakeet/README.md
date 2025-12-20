# Parakeet TDT Export for ExecuTorch

Export [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) speech recognition model to ExecuTorch.

## Installation

```bash
pip install nemo_toolkit[asr] torchaudio
```

## Export

Export the model (portable backend):
```bash
python export_parakeet_tdt.py
```

Export with a specific backend:
```bash
python export_parakeet_tdt.py --backend xnnpack      # CPU acceleration
python export_parakeet_tdt.py --backend cuda         # CUDA acceleration
python export_parakeet_tdt.py --backend cuda-windows # CUDA on Windows
```

Test transcription on an audio file:
```bash
python export_parakeet_tdt.py --audio /path/to/audio.wav
```

### Export Arguments

| Argument | Description |
|----------|-------------|
| `--output-dir` | Output directory for exports (default: `./parakeet_tdt_exports`) |
| `--backend` | Backend for acceleration: `portable`, `xnnpack`, `cuda`, `cuda-windows` (default: `portable`) |
| `--audio` | Path to audio file for transcription test |

## C++ Runner

### Building

First, build ExecuTorch with the LLM preset:

```bash
cd executorch
cmake --workflow --preset llm-release
```

Then build the parakeet runner:

```bash
cd examples/models/parakeet
cmake --workflow --preset parakeet-cpu
```

For Metal (macOS):
```bash
cd examples/models/parakeet
cmake --workflow --preset parakeet-metal
```

For CUDA (Linux/Windows):
```bash
cd examples/models/parakeet
cmake --workflow --preset parakeet-cuda
```

### Running

```bash
./cmake-out/examples/models/parakeet/parakeet_runner \
  --model_path parakeet.pte \
  --processor_path preprocessor.pte \
  --audio_path audio.wav
```

### Runner Arguments

| Argument | Description |
|----------|-------------|
| `--model_path` | Path to Parakeet model (.pte) |
| `--processor_path` | Path to preprocessor .pte for mel spectrogram extraction |
| `--audio_path` | Path to input audio file (.wav) |
| `--tokenizer_path` | Path to tokenizer file (for token-to-text conversion) |
