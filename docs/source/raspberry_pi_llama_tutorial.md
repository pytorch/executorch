# ExecuTorch on Raspberry Pi

## TLDR

This tutorial demonstrates how to deploy **Llama models on Raspberry Pi 4/5 devices** using ExecuTorch:

- **Prerequisites**: Linux host machine, Python 3.10-3.13, conda environment, Raspberry Pi 4/5
- **Setup**: Automated cross-compilation using `setup.sh` script for ARM toolchain installation
- **Export**: Convert Llama models to optimized `.pte` format with quantization options
- **Deploy**: Transfer binaries to Raspberry Pi and configure runtime libraries
- **Optimize**: Build optimization and performance tuning techniques
- **Result**: Efficient on-device Llama inference

## Prerequisites and Hardware Requirements

### Host Machine Requirements

**Operating System**: Linux x86_64 (Ubuntu 20.04+ or CentOS Stream 9+)

**Software Dependencies**:

- **Python 3.10-3.13** (ExecuTorch requirement)
- **conda** or **venv** for environment management
- **CMake 3.29.6+**
- **Git** for repository cloning

### Target Device Requirements

**Supported Devices**: **Raspberry Pi 4** and **Raspberry Pi 5** with **64-bit OS**

**Memory Requirements**:

- **RAM & Storage** Varies by model size and optimization level
- **64-bit Raspberry Pi OS** (Bullseye or newer)

### Verification Commands

Verify your host machine compatibility:
```bash
# Check OS and architecture
uname -s  # Should output: Linux
uname -m  # Should output: x86_64

# Check Python version
python3 --version  # Should be 3.10-3.13

# Check required tools
hash cmake git md5sum 2>/dev/null || echo "Missing required tools"

cmake --version  # Should be 3.29.6+ at minimum

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
conda create -yn executorch python=3.12.0
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

Run the following script on your Linux host machine:

```bash
# Run the Raspberry Pi setup script for Pi 5
examples/raspberry_pi/setup.sh pi5
```

On successful completion, you should see the following output:

```bash
[100%] Linking CXX executable llama_main
[100%] Built target llama_main
[SUCCESS] LLaMA runner built successfully

==== Verifying Build Outputs ====
[SUCCESS] ✓ llama_main (6.1M)
[SUCCESS] ✓ libllama_runner.so (4.0M)
[SUCCESS] ✓ libextension_module.a (89K) - static library

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

Make sure that the output does not have any GLIBC / other library mismatch errors in the output. If you see any, follow the troubleshooting steps below.

## Troubleshooting

### Issue 1: GLIBC Version Mismatch

**Problem:** The binary was compiled with a newer GLIBC version (2.38) than what's available on your Raspberry Pi (2.36).

**Error Symptoms:**

```bash
./llama_main: /lib/aarch64-linux-gnu/libm.so.6: version `GLIBC_2.38' not found (required by ./llama_main)
./llama_main: /lib/aarch64-linux-gnu/libc.so.6: version `GLIBC_2.38' not found (required by ./llama_main)
./llama_main: /lib/aarch64-linux-gnu/libstdc++.so.6: version `CXXABI_1.3.15' not found (required by ./llama_main)
./llama_main: /lib/aarch64-linux-gnu/libc.so.6: version `GLIBC_2.38' not found (required by /lib/libllama_runner.so)
```

**There are two potential solutions:**

- **Solution A**: Modify the Pi to match the binary (run on Pi)

- **Solution B**: Modify the binary to match the Pi (run on host)

#### Solution A: Upgrade GLIBC on Raspberry Pi (Recommended)

1. **Check your current GLIBC version:**

```bash
ldd --version
# Output: ldd (Debian GLIBC 2.36-9+rpt2+deb12u12) 2.36
```

2. **⚠️ Compatibility Warning and Safety Check:**

```bash
# Just check and warn - don't do the upgrade
current_glibc=$(ldd --version | head -n1 | grep -o '[0-9]\+\.[0-9]\+')
required_glibc="2.38"

echo "Current GLIBC: $current_glibc"
echo "Required GLIBC: $required_glibc"

if [[ $(echo "$current_glibc < $required_glibc" | bc -l) -eq 1 ]]; then
    echo ""
    echo "⚠️  WARNING: Your GLIBC version is too old"
    echo "   You need to upgrade to continue with the next steps"
    echo "   Consider using Solution B (rebuild binary) for better safety"
    echo ""
else
    echo "✅ Your GLIBC version is already compatible"
fi
```

**NOTE:** If the output shows "⚠️  WARNING: Your GLIBC version is too old", proceed with either Upgrade / Step #3 below (or) Solution B. Otherwise skip the next step as your device is __already compatible__ and directly go to Step#4.

3. **Upgrade to newer GLIBC:**

```bash
# Add Debian unstable repository
echo "deb http://deb.debian.org/debian sid main contrib non-free" | sudo tee -a /etc/apt/sources.list

# Update package lists
sudo apt update

# Install newer GLIBC packages
sudo apt-get -t sid install libc6 libstdc++6

# Reboot system
sudo reboot
```

4. **Verify compatibility after reboot:**

```bash
cd ~/executorch-deployment
source setup_env.sh

# Test that the binary works
if ./llama_main --help &>/dev/null; then
    echo "✅ GLIBC upgrade successful - binary is compatible"
else
    echo "❌ GLIBC upgrade failed - binary still incompatible"
    echo "Consider rolling back or refer to documentation for troubleshooting"
fi
```

5. **Test the fix:**

```bash
cd ~/executorch-deployment
source setup_env.sh
./llama_main --model_path ./llama3_2.pte --tokenizer_path ./tokenizer.model --seq_len 128 --prompt "Hello"
```

**Important Notes:**

- Select "Yes" when prompted to restart services
- Press Enter to keep current version for configuration files
- Backup important data before upgrading

#### Solution B: Rebuild with Raspberry Pi's GLIBC (Advanced)

If you prefer not to upgrade your Raspberry Pi system:

1. **Copy Pi's filesystem to host machine:**

```bash
# On Raspberry Pi - install rsync
ssh pi@<your-rpi-ip>
sudo apt update && sudo apt install rsync
exit

# On host machine - copy Pi's filesystem
mkdir -p ~/rpi5-sysroot
rsync -aAXv --exclude={"/proc","/sys","/dev","/run","/tmp","/mnt","/media","/lost+found"} \
    pi@<your-rpi-ip>:/ ~/rpi5-sysroot
```

2. **Update CMake toolchain file:**
```bash
# Edit arm-toolchain-pi5.cmake
# Replace this line:
# set(CMAKE_SYSROOT "${TOOLCHAIN_PATH}/aarch64-none-linux-gnu/libc")

# With this:
set(CMAKE_SYSROOT "/home/yourusername/rpi5-sysroot")
set(CMAKE_FIND_ROOT_PATH "${CMAKE_SYSROOT}")
```

3. **Rebuild binaries:**
```bash
# Clean and rebuild
rm -rf cmake-out
./examples/raspberry_pi/rpi_setup.sh pi5 --force-rebuild

# Verify GLIBC version
strings ./cmake-out/examples/models/llama/llama_main | grep GLIBC_
# Should show max GLIBC_2.36 (matching your Pi)
```

---

### Issue 2: Library Not Found

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

### Issue 3: Tokenizer JSON Parsing Warnings

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
