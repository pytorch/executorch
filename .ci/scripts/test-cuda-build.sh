#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -exu

CUDA_VERSION=${1:-"13.0"}

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

    # The CUDA delegate ships as its own shared library. Nothing else here would
    # notice if it were built into more than one place, and a process with two
    # copies of the delegate has two copies of its state, so check that the
    # installed tree defines it exactly once.
    python -c "
import shutil
import subprocess
import sys
from pathlib import Path

if shutil.which('nm') is None:
    print('INFO: nm unavailable, skipping the delegate duplication check')
    sys.exit(0)

import executorch

# A namespace package has no __file__, so derive the directory from the loader's
# search path instead.
locations = list(getattr(executorch, '__path__', []) or [])
if not locations:
    print('INFO: cannot locate the installed package, skipping the check')
    sys.exit(0)
package = Path(locations[0])
# A symbol the delegate itself defines. The shim layer's symbols stay resolvable
# even if the delegate stops being packaged, so probing one of those would prove
# the shim is present rather than that delegate code exists exactly once.
symbol = 'executorch::backends::cuda::CudaBackend::execute'
libraries = [p for p in package.rglob('*.so*') if p.is_file() and not p.is_symlink()]
definers = []
for library in libraries:
    result = subprocess.run(
        ['nm', '-DC', str(library)], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        continue
    for line in result.stdout.splitlines():
        parts = line.split(maxsplit=2)
        if len(parts) == 3 and parts[1] in 'TtWVu' and parts[2].startswith(symbol):
            definers.append(str(library.relative_to(package)))
            break

# The file is checked as well as the symbol. A missing delegate would already fail the
# symbol count above, but naming the file gives a clearer message about what is wrong.
delegates = [
    p for p in package.rglob('libexecutorch_cuda_backend.so*')
    if p.is_file() and not p.is_symlink()
]
if len(delegates) != 1:
    print(f'ERROR: expected one shipped CUDA delegate library, found {delegates}')
    sys.exit(1)
print(f'SUCCESS: one CUDA delegate library at {delegates[0].relative_to(package)}')

if not definers:
    # This runs inside a job that just built with CUDA enabled, so a missing
    # delegate means the build or the packaging stopped producing it. Treating
    # that as nothing to check would let the regression through.
    print('ERROR: CUDA was enabled but no shipped library defines the delegate')
    sys.exit(1)
if len(definers) != 1:
    print(f'ERROR: expected one library to define the CUDA delegate, found {definers}')
    sys.exit(1)
print(f'SUCCESS: exactly one CUDA delegate across {len(libraries)} shipped libraries')
" || exit $?

    # Loading it is what the symbol scan above cannot prove. A broken runtime
    # path, an undefined symbol, or a mismatched CUDA dependency all pass a name
    # check and fail here.
    #
    # Narrowed before the interpreter starts, not inside it: the loader reads this
    # variable once at process start, so changing it later would not affect what the
    # library is allowed to find. Only the entries holding a CUDA runtime are kept,
    # because the wheel does not bundle it. Everything else is dropped so the check
    # cannot pass on a machine whose environment happens to cover a dependency the
    # wheel should have carried itself.
    cuda_search_path=""
    IFS=':' read -ra _search_entries <<< "${LD_LIBRARY_PATH:-}"
    for _entry in "${_search_entries[@]}"; do
      # Any CUDA library, not just the runtime: the nvidia pip packages put each one in
      # its own directory, so matching only libcudart would drop the directory holding
      # libcurand and the load would still fail. Patterns are looped over rather than
      # brace-expanded, which compgen does not apply.
      if [ -z "${_entry}" ]; then
        continue
      fi
      for _pattern in libcudart libcurand libcublas libcudnn; do
        if compgen -G "${_entry}/${_pattern}*.so*" > /dev/null; then
          cuda_search_path="${cuda_search_path:+${cuda_search_path}:}${_entry}"
          break
        fi
      done
    done

    LD_LIBRARY_PATH="${cuda_search_path}" python -c "
import ctypes, os, sys
from pathlib import Path

# Imported before anything is loaded by hand. When CUDA comes from torch's bundled
# nvidia wheels rather than a system install, importing torch is what puts those
# libraries in reach; without it the loads below fail for a reason that is not a
# packaging defect.
try:
    import torch  # noqa: F401
except ImportError:
    pass

import executorch

package = Path(getattr(executorch, '__path__', [None])[0])
delegates = [
    p for p in package.rglob('libexecutorch_cuda_backend.so*')
    if p.is_file() and not p.is_symlink()
]
if len(delegates) != 1:
    print(f'ERROR: expected one shipped CUDA delegate, found {delegates}')
    sys.exit(1)

for library in [delegates[0]] + sorted(package.rglob('libaoti_cuda_shims.so*')):
    if not library.is_file() or library.is_symlink():
        continue
    try:
        ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
    except OSError as error:
        print(f'ERROR: {library.relative_to(package)} does not load: {error}')
        sys.exit(1)
    print(f'loaded {library.relative_to(package)}')
print('SUCCESS: the CUDA delegate and its shim load from the installed package')
" || exit $?

    echo "SUCCESS: ExecuTorch CUDA ${cuda_version} build and verification completed successfully"
}

# Main execution
echo "Current working directory: $(pwd)"
echo "Directory contents:"
ls -la

# Run the CUDA build test
test_executorch_cuda_build "${CUDA_VERSION}"
