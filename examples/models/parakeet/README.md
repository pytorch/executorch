# Parakeet TDT Export for ExecuTorch

Export [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) speech recognition model to ExecuTorch.

## Installation

```bash
pip install nemo_toolkit[asr] torchaudio
```

## Usage

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

### Arguments

| Argument | Description |
|----------|-------------|
| `--output-dir` | Output directory for exports (default: `./parakeet_tdt_exports`) |
| `--backend` | Backend for acceleration: `portable`, `xnnpack`, `cuda`, `cuda-windows` (default: `portable`) |
| `--audio` | Path to audio file for transcription test |
