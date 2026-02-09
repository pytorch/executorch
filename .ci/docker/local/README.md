# Local Docker Build for ExecuTorch

This directory contains scripts for building ExecuTorch Docker images locally,
without requiring access to Meta's CI infrastructure (S3 sccache).

## Usage

```bash
cd .ci/docker/local

# Build basic image with clang12 and PyTorch
./build.sh executorch-ubuntu-22.04-clang12 -t executorch-hackathon:basic

# Build image with Android NDK support
./build.sh executorch-ubuntu-22.04-clang12-android -t executorch-hackathon:android
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

## Differences from CI Build

This local build script:
1. Uses local sccache instead of S3 (no AWS credentials needed)
2. Removes `--no-cache` flag for faster rebuilds
3. Patches the Dockerfile and install scripts on-the-fly

## Running the Image

```bash
# Start container
docker run -it executorch-hackathon:basic bash

# Inside container, clone and build ExecuTorch
git clone https://github.com/pytorch/executorch.git
cd executorch
./install_executorch.sh

# Export a model
python -m examples.xnnpack.aot_compiler --model_name="mv2" --delegate
```

## Troubleshooting

If the build fails:
1. Ensure Docker has enough memory (8GB+ recommended)
2. Check disk space (build requires ~20GB)
3. Try with `--no-cache` flag if you encounter stale layer issues
