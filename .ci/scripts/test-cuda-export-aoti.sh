#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -exu

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"

CUDA_VERSION=${1:-"12.6"}

echo "=== Testing ExecutorTorch CUDA AOTI Export ${CUDA_VERSION} ==="

# Function to test CUDA AOTI export functionality
test_cuda_aoti_export() {
    local cuda_version=$1

    echo "Testing CUDA AOTI export with CUDA ${cuda_version} support..."

    # Check available resources before starting
    echo "=== System Information ==="
    echo "Available memory: $(free -h | grep Mem | awk '{print $2}')"
    echo "Available disk space: $(df -h . | tail -1 | awk '{print $4}')"
    echo "CPU cores: $(nproc)"
    echo "CUDA version check:"
    nvcc --version || echo "nvcc not found"
    nvidia-smi || echo "nvidia-smi not found"

    # Set up environment for CUDA builds
    export CMAKE_ARGS="-DEXECUTORCH_BUILD_CUDA=ON"

    echo "=== Installing ExecutorTorch with CUDA support ==="
    # Install ExecutorTorch with CUDA support with timeout and error handling
    timeout 5400 ./install_executorch.sh || {
        local exit_code=$?
        echo "ERROR: install_executorch.sh failed with exit code: $exit_code"
        if [ $exit_code -eq 124 ]; then
            echo "ERROR: Installation timed out after 90 minutes"
        fi
        exit $exit_code
    }

    echo "SUCCESS: ExecutorTorch CUDA installation completed"

    # Verify the installation
    echo "=== Verifying ExecutorTorch CUDA Installation ==="

    # Test that ExecutorTorch was built successfully
    python -c "
import executorch
print('SUCCESS: ExecutorTorch imported successfully')
"

    # Test CUDA availability and show details
    python -c "
try:
    import torch
    print('INFO: PyTorch version:', torch.__version__)
    print('INFO: CUDA available:', torch.cuda.is_available())

    if torch.cuda.is_available():
        print('SUCCESS: CUDA is available for ExecutorTorch')
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

        print('SUCCESS: ExecutorTorch CUDA integration verified')
    else:
        print('WARNING: CUDA not detected, but ExecutorTorch built successfully')
        exit(1)
except Exception as e:
    print('ERROR: ExecutorTorch CUDA test failed:', e)
    exit(1)
"

    echo "=== Running CUDA AOTI Export Tests ==="
    # Run the CUDA AOTI export tests using the Python script
    python .ci/scripts/test_cuda_export_aoti.py \
        --models linear conv2d add resnet18 \
        --export-mode export_aoti_only \
        --timeout 600 \
        --cleanup

    echo "SUCCESS: ExecutorTorch CUDA AOTI export ${cuda_version} tests completed successfully"
}

# Main execution
echo "Current working directory: $(pwd)"
echo "Directory contents:"
ls -la

# Run the CUDA AOTI export test
test_cuda_aoti_export "${CUDA_VERSION}"
