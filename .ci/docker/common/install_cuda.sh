#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Install Linux CUDA toolkit
# This installs nvcc and other CUDA development tools needed for compiling CUDA code

set -ex

# CUDA version must be specified (e.g., 12.8)
CUDA_VERSION="${CUDA_VERSION:?CUDA_VERSION must be set}"

# Convert version format (e.g., 12.8 -> 12-8 for package names)
CUDA_VERSION_DASH=$(echo "${CUDA_VERSION}" | tr '.' '-')

# Add NVIDIA package repository
apt-get update
apt-get install -y --no-install-recommends \
    gnupg2 \
    ca-certificates \
    wget

# Download and install the CUDA keyring
wget -q "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb" -O /tmp/cuda-keyring.deb
dpkg -i /tmp/cuda-keyring.deb
rm /tmp/cuda-keyring.deb

apt-get update

# Install CUDA toolkit (nvcc and development libraries)
# We install a minimal set of packages needed for compilation:
# - cuda-nvcc: The CUDA compiler
# - cuda-cudart-dev: CUDA runtime development files
# - cuda-nvrtc-dev: CUDA runtime compilation library
# - libcublas-dev: cuBLAS development files
# - libcusparse-dev: cuSPARSE development files
# - libcufft-dev: cuFFT development files
apt-get install -y --no-install-recommends \
    "cuda-nvcc-${CUDA_VERSION_DASH}" \
    "cuda-cudart-dev-${CUDA_VERSION_DASH}" \
    "cuda-nvrtc-dev-${CUDA_VERSION_DASH}" \
    "libcublas-dev-${CUDA_VERSION_DASH}" \
    "libcusparse-dev-${CUDA_VERSION_DASH}" \
    "libcufft-dev-${CUDA_VERSION_DASH}"

# Clean up
apt-get clean
rm -rf /var/lib/apt/lists/*

# Verify installation
/usr/local/cuda-${CUDA_VERSION}/bin/nvcc --version

echo "CUDA ${CUDA_VERSION} toolkit installation complete"
echo "CUDA_HOME=/usr/local/cuda-${CUDA_VERSION}"
