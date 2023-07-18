# Docker images for Executorch CI

This directory contains everything needed to build the Docker images
that are used in Executorch CI.

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

