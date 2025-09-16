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

echo "=== Testing ExecutorTorch CUDA ${CUDA_VERSION} Build ==="

# Function to build and test ExecutorTorch with CUDA support
test_executorch_cuda_build() {
    local cuda_version=$1

    echo "Building ExecutorTorch with CUDA ${cuda_version} support..."
    echo "ExecutorTorch will automatically detect CUDA and install appropriate PyTorch wheel"

    # Set CMAKE_ARGS to enable CUDA build - ExecutorTorch will handle PyTorch installation automatically
    export CMAKE_ARGS="-DEXECUTORCH_BUILD_CUDA=ON"

    # Install ExecutorTorch with CUDA support - this will automatically:
    # 1. Detect CUDA version using nvcc
    # 2. Install appropriate PyTorch wheel for the detected CUDA version
    # 3. Build ExecutorTorch with CUDA support
    ./install_executorch.sh

    echo "SUCCESS: ExecutorTorch CUDA build completed"

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

    echo "SUCCESS: ExecutorTorch CUDA ${cuda_version} build and verification completed successfully"
}

# Main execution
echo "Current working directory: $(pwd)"
echo "Directory contents:"
ls -la

# Run the CUDA build test
test_executorch_cuda_build "${CUDA_VERSION}"
