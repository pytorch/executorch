# ExecuTorch on Raspberry Pi

## TLDR

This tutorial demonstrates how to deploy **Llama models on Raspberry Pi 4/5 devices** using ExecuTorch:

- **Prerequisites**: Linux host machine, Python 3.11-3.15, conda environment, Raspberry Pi 4/5
- **Setup**: Automated cross-compilation using `setup.sh` script for ARM toolchain installation
- **Export**: Convert Llama models to optimized `.pte` format with quantization options
- **Deploy**: Transfer binaries to Raspberry Pi and configure runtime libraries
- **Optimize**: Build optimization and performance tuning techniques
- **Result**: Efficient on-device Llama inference

## Prerequisites and Hardware Requirements

### Host Machine Requirements

**Operating System**: Linux x86_64 (Ubuntu 20.04+ or CentOS Stream 9+)

**Software Dependencies**:

- **Python 3.11-3.15** (ExecuTorch requirement)
- **conda** or **venv** for environment management
- **CMake 3.26+**
- **Git** for repository cloning

### Target Device Requirements

**Supported Devices**: **Raspberry Pi 4** and **Raspberry Pi 5** with **64-bit OS**

**Memory Requirements**:

- **RAM & Storage** Varies by model size and optimization level
- **64-bit Raspberry Pi OS** Bookworm (Debian 12) or Trixie (Debian 13). Bullseye is not supported.

### Verification Commands

Verify your host machine compatibility:
```bash
# Check OS and architecture
uname -s  # Should output: Linux
uname -m  # Should output: x86_64

# Check Python version
python3 --version  # Should be 3.11-3.15

# Check required tools
hash cmake git md5sum 2>/dev/null || echo "Missing required tools"

cmake --version  # Should be 3.26+ at minimum

## Development Environment Setup

### Clone ExecuTorch Repository

First, clone the ExecuTorch repository with the Raspberry Pi support:

```bash
# Create project directory
mkdir ~/executorch-rpi && cd ~/executorch-rpi &&  git clone -b release/1.0 https://github.com/pytorch/executorch.git &&
cd executorch
```

### Create Conda Environment

```bash
# Create conda environment
conda create -yn executorch python=3.11
conda activate executorch

# Upgrade pip
pip install --upgrade pip
```

Alternative: Virtual Environment
If you prefer Python's built-in virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

Refer to → {doc}`getting-started` for more details.

## Cross-Compilation Toolchain Step

Check the Raspberry Pi OS release on your Pi:

```bash
ssh <rpi-user>@<rpi-ip> grep VERSION_CODENAME /etc/os-release
# VERSION_CODENAME=bookworm
```

`--os` builds against that release's glibc so the binaries run on the Pi as is. It needs `mmdebstrap` on the host:

```bash
sudo apt-get install mmdebstrap debian-archive-keyring
```

Run the following script on your Linux host machine with the codename from above (`bookworm` or `trixie`):

```bash
# Pi 5 running Raspberry Pi OS Bookworm
examples/raspberry_pi/setup.sh pi5 --os bookworm
```

Without `--os`, the binaries are built against the toolchain's bundled glibc and fail on a Pi with an older glibc (e.g. Bookworm) with `version 'GLIBC_2.38' not found`. Each run does a clean build.

On successful completion, you should see the following output:

```bash
==== Verifying Build Outputs ====
[INFO] Checking required binaries...
[SUCCESS] ✓ llama_main (6.9M)
[SUCCESS] ✓ libllama_runner.so (17M)
[SUCCESS] ✓ libextension_module.a (159K) - statically linked
[SUCCESS] All required binaries built successfully!
[INFO] Binaries require glibc >= 2.36
[SUCCESS] Compatible with Debian bookworm (glibc 2.36)

==== Setup Complete! ====

✓ ExecuTorch cross-compilation setup completed successfully!
```

## Model Preparation and Export

### Download Llama Models

Download the Llama model from Hugging Face or any other source, and make sure that following files exist.

- consolidated.00.pth (model weights)
- params.json (model config)
- tokenizer.model (tokenizer)

### Export Llama to ExecuTorch Format

After downloading the Llama model, export it to ExecuTorch format using the provided script:

```bash

#### Set these paths to point to the exported files. Following is an example instruction to export a llama model

LLAMA_QUANTIZED_CHECKPOINT=path/to/consolidated.00.pth
LLAMA_PARAMS=path/to/params.json

python -m extension.llm.export.export_llm \
  --config examples/models/llama/config/llama_xnnpack_spinquant.yaml \
  +base.model_class="llama3_2" \
  +base.checkpoint="${LLAMA_QUANTIZED_CHECKPOINT:?}" \
  +base.params="${LLAMA_PARAMS:?}"
```

The file llama3_2.pte will be generated at the place where you run the command

- For more details see [Option A: Download and Export Llama3.2 1B/3B Model](https://github.com/pytorch/executorch/blob/main/examples/models/llama/README.md#option-a-download-and-export-llama32-1b3b-model)
- Also refer to → {doc}`llm/export-llm` for more details.

## Raspberry Pi Deployment

### Transfer Binaries to Raspberry Pi

After successful cross-compilation, transfer the required files:

```bash
##### Set Raspberry Pi details
export RPI_UN="pi"  # Your Raspberry Pi username
export RPI_IP="your-rpi-ip-address"

##### Create deployment directory on Raspberry Pi
ssh $RPI_UN@$RPI_IP 'mkdir -p ~/executorch-deployment'
##### Copy main executable
scp cmake-out/examples/models/llama/llama_main $RPI_UN@$RPI_IP:~/executorch-deployment/
##### Copy runtime library
scp cmake-out/examples/models/llama/runner/libllama_runner.so $RPI_UN@$RPI_IP:~/executorch-deployment/
##### Copy model file
scp llama3_2.pte $RPI_UN@$RPI_IP:~/executorch-deployment/
scp ./tokenizer.model $RPI_UN@$RPI_IP:~/executorch-deployment/
```

### Configure Runtime Libraries on Raspberry Pi

SSH into your Raspberry Pi and configure the runtime:

#### Set up library environment

```bash
cd ~/executorch-deployment
echo 'export LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH' > setup_env.sh
chmod +x setup_env.sh

#### Make executable

chmod +x llama_main
```

## Dry Run

```bash
source setup_env.sh
./llama_main --help
```

Make sure that the output does not have any GLIBC / other library mismatch errors. If you see `GLIBC_x.y not found`, rerun the cross-compilation step with `--os` set to the Pi's release. For other errors, see the troubleshooting steps below.

## Troubleshooting

### Issue 1: Library Not Found

**Problem:** Required libraries are not found at runtime.

**Error Symptoms:**
```bash
./llama_main: error while loading shared libraries: libllama_runner.so: cannot open shared object file
```

**Solution:**
```bash
# Ensure you're in the correct directory and environment is set
cd ~/executorch-deployment
source setup_env.sh
./llama_main --help
```

**Root Cause:** Either `LD_LIBRARY_PATH` is not set or you're not in the deployment directory.

---

### Issue 2: Tokenizer JSON Parsing Warnings

**Problem:** Warning messages about JSON parsing errors after running the llama_main binary.

**Error Symptoms:**

```bash
E tokenizers:hf_tokenizer.cpp:60] Error parsing json file: [json.exception.parse_error.101]
```

**Solution:** These warnings can be safely ignored. They don't affect model inference.

---


## Quick Test Command

After resolving issues, test with:

```bash
cd ~/executorch-deployment
source setup_env.sh
./llama_main --model_path ./llama3_2.pte --tokenizer_path ./tokenizer.model --seq_len 128 --prompt "What is the meaning of life?"
```

## Debugging Tools

Enable ExecuTorch logging:

```bash
# Set log level for debugging
export ET_LOG_LEVEL=Info
./llama_main --model_path ./model.pte --verbose
```

## Final Run command

```bash
cd ~/executorch-deployment
source setup_env.sh
./llama_main --model_path ./llama3_2.pte --tokenizer_path ./tokenizer.model --seq_len 128 --prompt "What is the meaning of life?"
```

Happy Inferencing!
