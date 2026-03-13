# Docker Build for ExecuTorch

This directory contains scripts for building ExecuTorch Docker images locally,
without requiring access to Meta's CI infrastructure.

## Quick Start (Pre-built Images)

Pre-built images are available on Docker Hub:

```bash
# Basic image (Ubuntu 22.04 + PyTorch + build tools)
docker pull youngmeta/executorch:basic
docker run -it --platform linux/amd64 youngmeta/executorch:basic bash

# Android image (includes Android NDK for cross-compilation)
docker pull youngmeta/executorch:android
docker run -it --platform linux/amd64 youngmeta/executorch:android bash
```

After starting the container, clone and install ExecuTorch:

```bash
# Inside the container
git clone https://github.com/pytorch/executorch.git
cd executorch
./install_executorch.sh

# Export a model (example)
python -m examples.xnnpack.aot_compiler --model_name="mv2" --delegate
```

## Building Images Locally

If you need to build the images yourself:

```bash
cd docker

# Build basic image with clang12 and PyTorch
./build.sh executorch-ubuntu-22.04-clang12 -t executorch:basic

# Build image with Android NDK support
./build.sh executorch-ubuntu-22.04-clang12-android -t executorch:android
```

## Supported Images

| Image Name | Description |
|------------|-------------|
| `executorch-ubuntu-22.04-gcc11` | GCC 11 build environment |
| `executorch-ubuntu-22.04-clang12` | Clang 12 build environment |
| `executorch-ubuntu-22.04-clang12-android` | Clang 12 + Android NDK |
| `executorch-ubuntu-22.04-arm-sdk` | Arm SDK for embedded targets |
| `executorch-ubuntu-22.04-qnn-sdk` | Qualcomm QNN backend support |
| `executorch-ubuntu-22.04-mediatek-sdk` | MediaTek backend support |

## What's Included

The built image includes:
- Ubuntu 22.04
- Python 3.10 (via Conda)
- PyTorch (built from source)
- Build tools (CMake, Ninja, etc.)
- ExecuTorch dependencies

## Running the Image

```bash
# Start container with local executorch mounted
docker run -it --platform linux/amd64 \
  -v /path/to/executorch:/executorch \
  -w /executorch \
  youngmeta/executorch:basic bash

# Inside container, install ExecuTorch
./install_executorch.sh

# Export a model
python -m examples.xnnpack.aot_compiler --model_name="mv2" --delegate
```

## For QNN Backend Development

```bash
docker run -it --platform linux/amd64 \
  -v /path/to/executorch:/executorch \
  -v /path/to/qnn_sdk_2.37:/qnn-sdk \
  -e QNN_SDK_ROOT=/qnn-sdk \
  -e LD_LIBRARY_PATH=/qnn-sdk/lib/x86_64-linux-clang \
  -w /executorch \
  youngmeta/executorch:basic bash

# Inside container, build QNN backend
./backends/qualcomm/scripts/build.sh
```

## Apple Silicon (M1/M2/M3) Notes

On Apple Silicon Macs, use `--platform linux/amd64` when running the container.
The images are built for x86_64 because:
- PyTorch build has ARM compiler bugs with SVE + BFloat16 on Clang 12
- QNN SDK only provides x86_64 Linux libraries

The x86_64 emulation is slower but ensures compatibility.

## Building Images Locally

The local build script:
1. Skips sccache entirely (no S3/AWS credentials needed)
2. Limits parallel jobs to reduce memory usage
3. Patches the Dockerfile and install scripts on-the-fly
4. Automatically uses `--platform linux/amd64` on Apple Silicon Macs

Build times: 1-2 hours on Apple Silicon, faster on native x86_64 Linux.

## Troubleshooting

If the build fails:
1. Ensure Docker has enough memory (16GB+ recommended)
2. Check disk space (build requires ~20GB)
3. Try with `--no-cache` flag if you encounter stale layer issues
