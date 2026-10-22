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

## Reusing the pinned PyTorch installer

From an ExecuTorch checkout, activate a venv or conda environment and run:

```bash
bash .ci/scripts/install_pinned_pytorch.sh
bash .ci/scripts/install_pinned_pytorch.sh --verify-only
```

The installer reads `.ci/docker/ci_commit_pins/pytorch.txt`, resolves named refs,
and reuses an installed build with the same commit. Set `PYTHON_EXECUTABLE` to
select another interpreter, or `TORCH_VERSION` to override the source ref as the
Docker build does. A mismatched build is replaced, including when downgrading.
Install mode also installs the domain packages when the torch build is reused.
`--verify-only` checks the installed PyTorch commit and LAPACK support, returning
nonzero on a mismatch or failed check. It does not install packages or change the
active environment. Named refs are resolved with `git ls-remote`, without cloning.
This is not a check of every ExecuTorch build or test dependency.

Use it inside a CI container to check that its prebuilt PyTorch matches the pin
in the current checkout, or locally to check the selected venv/conda environment
before running tests:

```bash
PYTHON_EXECUTABLE="$PWD/env/bin/python" \
  bash .ci/scripts/install_pinned_pytorch.sh --verify-only
```

It can also run after dependency installation to detect replacement of the
pinned build. Linux and macOS CI verify after installing ExecuTorch, before
building the test runner. On Linux this also checks that the Docker image
provides the commit required by the checkout.

The shared implementation lives in `common/install_pinned_pytorch.sh` so it is
also available in the Docker build context. Docker and macOS CI use this same
implementation. System build dependencies (including BLAS/LAPACK) must already
be installed. Windows is not supported.
