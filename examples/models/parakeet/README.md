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
| `--backend` | Backend for acceleration: `portable`, `xnnpack`, `metal`, `cuda`, `cuda-windows` (default: `xnnpack`) |
| `--dtype` | Data type: `fp32`, `bf16`, `fp16` (default: `fp32`). Metal backend supports `fp32` and `bf16` only (no `fp16`). |
| `--audio` | Path to audio file for transcription test |

**Note:** The preprocessor is always lowered with the portable backend regardless of the `--backend` setting.

### Quantization

The export script supports quantizing encoder and decoder linear layers using [torchao](https://github.com/pytorch/ao).

#### Quantization Arguments

| Argument | Description |
|----------|-------------|
| `--qlinear_encoder` | Quantization config for encoder linear layers: `4w`, `8w`, `8da4w`, `8da8w`, `fpa4w` |
| `--qlinear_encoder_group_size` | Group size for encoder linear quantization (default: 32) |
| `--qlinear_encoder_packing_format` | Packing format for encoder: `tile_packed_to_4d` |
| `--qlinear` | Quantization config for decoder linear layers: `4w`, `8w`, `8da4w`, `8da8w`, `fpa4w` |
| `--qlinear_group_size` | Group size for decoder linear quantization (default: 32) |
| `--qlinear_packing_format` | Packing format for decoder: `tile_packed_to_4d` |
| `--qembedding` | Quantization config for decoder embedding layer: `4w`, `8w` |
| `--qembedding_group_size` | Group size for embedding quantization (default: 0 = per-axis) |

#### Quantization Configs

| Config | Description | Backends |
|--------|-------------|----------|
| `4w` | 4-bit weight only quantization | CUDA |
| `8w` | 8-bit weight only quantization | CUDA |
| `8da4w` | 8-bit dynamic activation, 4-bit weight | CUDA |
| `8da8w` | 8-bit dynamic activation, 8-bit weight | CUDA |
| `fpa4w` | Floating point activation, 4-bit weight | Metal |

#### Example: Dynamic Quantization for XNNPACK

```bash
python export_parakeet_tdt.py \
    --backend xnnpack \
    --qlinear_encoder 8da4w \
    --qlinear_encoder_group_size 32 \
    --qlinear 8da4w \
    --qlinear_group_size 32 \
    --output-dir ./parakeet_quantized_xnnpack
```

#### Example: 4-bit Weight Quantization with Tile Packing for CUDA

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

#### Example: Metal 4-bit Quantization

```bash
python export_parakeet_tdt.py \
    --backend metal \
    --qlinear_encoder fpa4w \
    --qlinear_encoder_group_size 32 \
    --qlinear fpa4w \
    --qlinear_group_size 32 \
    --output-dir ./parakeet_metal_quantized
```

**Note:** Metal 4-bit quantization requires torchao built with experimental MPS (Metal) ops.

You can install torchao with Metal support from the `ao` repo:
```bash
USE_CPP=1 TORCHAO_BUILD_EXPERIMENTAL_MPS=1 pip install . --no-build-isolation
```

Alternatively, you can build torchao with Metal support while installing ExecuTorch:
```bash
EXECUTORCH_BUILD_KERNELS_TORCHAO=1 TORCHAO_BUILD_EXPERIMENTAL_MPS=1 ./install_executorch.sh
```

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

### CUDA-Windows Export

Before running `cuda-windows` export, make sure these requirements are set up:
- `x86_64-w64-mingw32-g++` is installed and on `PATH` (mingw-w64 cross-compiler).
- `WINDOWS_CUDA_HOME` points to the extracted Windows CUDA package directory.

Example setup on Ubuntu:

```bash
# 1) Install cross-compiler + extraction tools
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  g++-mingw-w64-x86-64-posix mingw-w64-tools p7zip-full wget

# 2) Verify cross-compiler
x86_64-w64-mingw32-g++ --version

# 3) Download and extract Windows CUDA installer package
CUDA_VERSION=12.8.1
CUDA_DRIVER_VERSION=572.61
CUDA_INSTALLER="cuda_${CUDA_VERSION}_${CUDA_DRIVER_VERSION}_windows.exe"
CUDA_URL="https://developer.download.nvidia.com/compute/cuda/${CUDA_VERSION}/local_installers/${CUDA_INSTALLER}"

mkdir -p /opt/cuda-windows
cd /opt/cuda-windows
wget -q "${CUDA_URL}" -O "${CUDA_INSTALLER}"
7z x "${CUDA_INSTALLER}" -oextracted -y

# 4) Point WINDOWS_CUDA_HOME to extracted Windows CUDA payload
export WINDOWS_CUDA_HOME=/opt/cuda-windows/extracted/cuda_cudart/cudart
```

```bash
python export_parakeet_tdt.py --backend cuda-windows --output-dir ./parakeet_cuda_windows
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

On Windows (PowerShell), use CMake workflow presets directly:

```powershell
cmake --workflow --preset llm-release-cuda
Push-Location examples/models/parakeet
cmake --workflow --preset parakeet-cuda
Pop-Location
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

Windows (PowerShell):

```powershell
.\cmake-out\examples\models\parakeet\Release\parakeet_runner.exe `
  --model_path C:\path\to\parakeet_cuda_windows\model.pte `
  --data_path C:\path\to\parakeet_cuda_windows\aoti_cuda_blob.ptd `
  --audio_path C:\path\to\audio.wav `
  --tokenizer_path C:\path\to\parakeet_cuda_windows\tokenizer.model
```

If your generator is single-config, the runner may be at `.\cmake-out\examples\models\parakeet\parakeet_runner.exe` instead.

### Runner Arguments

| Argument | Description |
|----------|-------------|
| `--model_path` | Path to Parakeet model (.pte) |
| `--audio_path` | Path to input audio file (.wav) |
| `--tokenizer_path` | Path to tokenizer file (default: `tokenizer.json`) |
| `--data_path` | Path to data file (.ptd) for delegate data (required for CUDA/CUDA-Windows) |
| `--timestamps`     | Timestamp output mode: `none\|token\|word\|segment\|all` (default: `segment`) |
