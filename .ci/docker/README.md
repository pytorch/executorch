# Docker images for ExecuTorch CI

This directory contains everything needed to build the Docker images
that are used in ExecuTorch CI. The content of this directory are copied
from PyTorch CI https://github.com/pytorch/pytorch/tree/main/.ci/docker.
It also uses the same directory structure as PyTorch.

## Contents

* `build.sh` -- dispatch script to launch all builds
* `common` -- scripts used to execute individual Docker build stages
* `ubuntu` -- Dockerfile for Ubuntu image for CPU build and test jobs

## Usage

```bash
# Generic usage
./build.sh "${IMAGE_NAME}" "${DOCKER_BUILD_PARAMETERS}"

# Build a specific image
./build.sh executorch-ubuntu-22.04-clang12 -t myimage:latest

# Set CLANG version (see build.sh) and build image
CLANG_VERSION=11 ./build.sh executorch-ubuntu-22.04-clang11 -t myimage:latest
```

The clang12 Linux image uses the published November 2 torch, torchvision and
torchaudio source commits in `ci_commit_pins/*-linux-clang12.txt`. The shared
macOS pin and other images' sources are unchanged. The build checks the installed
torch identity and schema-1.0 reporter after installing the domain libraries.

These pins participate in the existing global Linux image hash. A pin change
can require other Linux images to rebuild, even though only clang12 reads it.
Validate the rebuilt image before enabling test-report uploads.
