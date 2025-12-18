#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Install PyTorch with CUDA support from prebuilt wheels
# This is used for the cuda-windows Docker image to get a specific CUDA version

set -ex

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"

# Default CUDA version if not specified
CUDA_VERSION="${CUDA_VERSION:-12.8}"

# Ensure PYTHON_VERSION is set (should be set by Dockerfile ENV)
if [ -z "${PYTHON_VERSION}" ]; then
    echo "ERROR: PYTHON_VERSION environment variable is not set"
    exit 1
fi

echo "Using Python version: ${PYTHON_VERSION}"

# Convert CUDA version to PyTorch wheel suffix (e.g., 12.8 -> cu128)
CUDA_SUFFIX="cu$(echo ${CUDA_VERSION} | tr -d '.')"

echo "Installing PyTorch with CUDA ${CUDA_VERSION} (${CUDA_SUFFIX})..."

# Install PyTorch from nightly with specific CUDA version into the conda environment
pip_install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/nightly/${CUDA_SUFFIX}"

# Verify installation
conda_run python -c "import torch; print(f'PyTorch {torch.__version__} installed with CUDA {torch.version.cuda}')"

echo "PyTorch CUDA installation complete"
