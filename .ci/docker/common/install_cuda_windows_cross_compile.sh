#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Install mingw-w64 cross-compiler and Windows CUDA toolkit for cross-compilation

set -ex

INSTALL_DIR="${WINDOWS_CUDA_INSTALL_DIR:-/opt/cuda-windows}"

# Mapping of CUDA versions to their corresponding driver versions for Windows installers
# Source: https://developer.nvidia.com/cuda-toolkit-archive
declare -A CUDA_DRIVER_MAP=(
    ["12.6"]="12.6.3:561.17"
    ["12.8"]="12.8.1:572.61"
    ["12.9"]="12.9.1:576.57"
)

install_mingw() {
    echo "Installing mingw-w64 cross-compiler..."

    apt-get update
    # Install the POSIX threads version of mingw-w64 which supports C++11 threading
    # primitives (std::mutex, std::condition_variable, std::shared_mutex).
    # The default win32 threads version does not support these.
    apt-get install -y --no-install-recommends \
        g++-mingw-w64-x86-64-posix \
        mingw-w64-tools \
        p7zip-full \
        wget

    # Verify installation shows POSIX threads
    x86_64-w64-mingw32-g++ --version

    # Cleanup
    apt-get clean
    rm -rf /var/lib/apt/lists/*

    echo "mingw-w64 installation complete (POSIX threads version)"
}

get_torch_cuda_version() {
    # Query PyTorch for its CUDA version using conda environment
    conda run -n "py_${PYTHON_VERSION}" python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo ""
}

install_windows_cuda() {
    # Get CUDA version from torch
    TORCH_CUDA_VERSION=$(get_torch_cuda_version)

    if [ -z "${TORCH_CUDA_VERSION}" ] || [ "${TORCH_CUDA_VERSION}" = "None" ]; then
        echo "ERROR: Could not detect CUDA version from PyTorch."
        echo "Make sure PyTorch with CUDA support is installed before running this script."
        exit 1
    fi

    echo "Detected PyTorch CUDA version: ${TORCH_CUDA_VERSION}"

    # Extract major.minor version (e.g., "12.8" from "12.8.1" or "12.8")
    CUDA_MAJOR_MINOR=$(echo "${TORCH_CUDA_VERSION}" | cut -d. -f1,2)

    # Look up the full version and driver version
    if [ -z "${CUDA_DRIVER_MAP[${CUDA_MAJOR_MINOR}]}" ]; then
        echo "ERROR: CUDA version ${CUDA_MAJOR_MINOR} is not in the known version map."
        echo "Known versions: ${!CUDA_DRIVER_MAP[*]}"
        exit 1
    fi

    CUDA_INFO="${CUDA_DRIVER_MAP[${CUDA_MAJOR_MINOR}]}"
    CUDA_VERSION=$(echo "${CUDA_INFO}" | cut -d: -f1)
    CUDA_DRIVER_VERSION=$(echo "${CUDA_INFO}" | cut -d: -f2)

    echo "Using CUDA ${CUDA_VERSION} with driver ${CUDA_DRIVER_VERSION}"

    echo "Installing Windows CUDA toolkit ${CUDA_VERSION}..."

    mkdir -p "${INSTALL_DIR}"
    cd "${INSTALL_DIR}"

    CUDA_INSTALLER="cuda_${CUDA_VERSION}_${CUDA_DRIVER_VERSION}_windows.exe"
    CUDA_URL="https://developer.download.nvidia.com/compute/cuda/${CUDA_VERSION}/local_installers/${CUDA_INSTALLER}"

    # Check if already downloaded and extracted
    if [ -d "${INSTALL_DIR}/extracted/cuda_cudart" ]; then
        echo "Windows CUDA toolkit already installed, skipping download..."
        return 0
    fi

    echo "Downloading CUDA installer from ${CUDA_URL}..."
    wget -q "${CUDA_URL}" -O "${CUDA_INSTALLER}"

    echo "Extracting CUDA toolkit..."
    7z x "${CUDA_INSTALLER}" -o"extracted" -y

    # Fix permissions so ci-user can access the files
    chmod -R a+rX "${INSTALL_DIR}"

    # Clean up installer to save space
    rm -f "${CUDA_INSTALLER}"

    echo "Windows CUDA toolkit installation complete"
    echo "WINDOWS_CUDA_HOME=${INSTALL_DIR}/extracted/cuda_cudart/cudart"
}

# Parse command line arguments
INSTALL_MINGW=false
INSTALL_CUDA=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --mingw)
            INSTALL_MINGW=true
            shift
            ;;
        --cuda)
            INSTALL_CUDA=true
            shift
            ;;
        --all)
            INSTALL_MINGW=true
            INSTALL_CUDA=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--mingw] [--cuda] [--all]"
            exit 1
            ;;
    esac
done

# Default to installing everything if no options specified
if [ "${INSTALL_MINGW}" = false ] && [ "${INSTALL_CUDA}" = false ]; then
    INSTALL_MINGW=true
    INSTALL_CUDA=true
fi

if [ "${INSTALL_MINGW}" = true ]; then
    install_mingw
fi

if [ "${INSTALL_CUDA}" = true ]; then
    install_windows_cuda
fi

echo "Installation complete"
