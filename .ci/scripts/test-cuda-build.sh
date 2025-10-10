#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -exu

CUDA_VERSION=${1:-"12.6"}

echo "=== Testing ExecuTorch CUDA ${CUDA_VERSION} Build ==="

# Function to build and test ExecuTorch with CUDA support
test_executorch_cuda_build() {
    local cuda_version=$1

    echo "Building ExecuTorch with CUDA ${cuda_version} support..."
    echo "ExecuTorch will automatically detect CUDA and install appropriate PyTorch wheel"

    # Check available resources before starting
    echo "=== System Information ==="
    echo "Available memory: $(free -h | grep Mem | awk '{print $2}')"
    echo "Available disk space: $(df -h . | tail -1 | awk '{print $4}')"
    echo "CPU cores: $(nproc)"
    echo "CUDA version check:"
    nvcc --version || echo "nvcc not found"
    nvidia-smi || echo "nvidia-smi not found"

    # Set CMAKE_ARGS to enable CUDA build - ExecuTorch will handle PyTorch installation automatically
    export CMAKE_ARGS="-DEXECUTORCH_BUILD_CUDA=ON"

    echo "=== Starting ExecuTorch Installation ==="
    # Install ExecuTorch with CUDA support with timeout and error handling
    timeout 5400 ./install_executorch.sh || {
        local exit_code=$?
        echo "ERROR: install_executorch.sh failed with exit code: $exit_code"
        if [ $exit_code -eq 124 ]; then
            echo "ERROR: Installation timed out after 90 minutes"
        fi
        exit $exit_code
    }

    echo "SUCCESS: ExecuTorch CUDA build completed"

    # Verify the installation
    echo "=== Verifying ExecuTorch CUDA Installation ==="

    # Test that ExecuTorch was built successfully
    python -c "
import executorch
print('SUCCESS: ExecuTorch imported successfully')
"

    # Test CUDA availability and show details
    python -c "
try:
    import torch
    print('INFO: PyTorch version:', torch.__version__)
    print('INFO: CUDA available:', torch.cuda.is_available())

    if torch.cuda.is_available():
        print('SUCCESS: CUDA is available for ExecuTorch')
        print('INFO: CUDA version:', torch.version.cuda)
        print('INFO: GPU device count:', torch.cuda.device_count())
        print('INFO: Current GPU device:', torch.cuda.current_device())
        print('INFO: GPU device name:', torch.cuda.get_device_name())

        # Test basic CUDA tensor operation
        device = torch.device('cuda')
        x = torch.randn(10, 10).to(device)
        y = torch.randn(10, 10).to(device)
        z = torch.mm(x, y)
        print('SUCCESS: CUDA tensor operation completed on device:', z.device)
        print('INFO: Result tensor shape:', z.shape)

        print('SUCCESS: ExecuTorch CUDA integration verified')
    else:
        print('WARNING: CUDA not detected, but ExecuTorch built successfully')
        exit(1)
except Exception as e:
    print('ERROR: ExecuTorch CUDA test failed:', e)
    exit(1)
"

    echo "SUCCESS: ExecuTorch CUDA ${cuda_version} build and verification completed successfully"
}

# Main execution
echo "Current working directory: $(pwd)"
echo "Directory contents:"
ls -la

# Run the CUDA build test
test_executorch_cuda_build "${CUDA_VERSION}"
