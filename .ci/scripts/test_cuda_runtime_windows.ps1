# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Builds and runs the CUDA backend's runtime C++ tests on a Windows GPU runner, the
# counterpart of the unittest-cuda-runtime job in cuda.yml. The Windows runners carry
# data-center GPUs in TCC mode, which have no memory pools, so this is also what covers the
# allocator's synchronous fallback; the Linux job covers the pool path.
$ErrorActionPreference = "Stop"

# Windows PowerShell 5.1 does not stop on a failing native command.
function Invoke-Native {
    param([Parameter(Mandatory = $true)][scriptblock]$Command)
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "exit code ${LASTEXITCODE}: $Command"
    }
}

& "$PSScriptRoot\install_nvidia_driver_windows.ps1"

$cudaNvcc = Join-Path $env:CUDA_HOME "bin\nvcc.exe"
if (-not (Test-Path $cudaNvcc)) { throw "CUDA compiler not found at '$cudaNvcc'" }
$env:CUDACXX = $cudaNvcc
$env:PATH = "$env:CUDA_HOME\bin\x64;$env:CUDA_HOME\bin;$env:PATH"
# The runner's GPU has no memory pools, which is what makes this job cover the
# allocator's fallback; fail rather than skip those tests if it ever has them.
$env:EXECUTORCH_CUDA_TEST_REQUIRE_NO_MEMORY_POOLS = "1"

$tests = @(
    "test_cuda_allocator",
    "test_cuda_mutable_state",
    "test_cuda_weight_cache",
    "test_cuda_sort_rand",
    "test_cuda_guard",
    "test_cuda_stream_guard"
)
$numCores = [Math]::Max([Environment]::ProcessorCount - 1, 1)
Invoke-Native {
    cmake --preset llm-release-cuda -DEXECUTORCH_BUILD_TESTS=ON -DCMAKE_CXX_STANDARD=20 `
        -T "cuda=$env:CUDA_HOME" "-DCMAKE_CUDA_COMPILER=$cudaNvcc" "-DCUDAToolkit_ROOT=$env:CUDA_HOME"
}
Invoke-Native { cmake --build cmake-out --config Release -j $numCores --target $tests }
foreach ($test in $tests) {
    Invoke-Native { ctest --test-dir cmake-out -C Release -R "^$test`$" --output-on-failure -V }
}
