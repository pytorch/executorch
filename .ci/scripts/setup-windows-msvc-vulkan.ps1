# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Build-validation for the Vulkan backend under MSVC on Windows. Mirrors
# setup-windows-msvc.ps1 but installs glslc (the Vulkan shader compiler) and
# configures/builds the vulkan_backend target. This is a bring-up job: it exists
# to surface MSVC portability issues in the Vulkan/volk/VMA code, so it may need
# iteration.

conda create --yes --quiet -n et python=3.12
conda activate et

# Install cmake
conda install -y cmake

# Activate the VS environment - this is required for MSVC to work.
& "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64

# Install glslc (via conda-forge shaderc) and put it on PATH in this process.
.ci/scripts/setup-vulkan-windows-deps.ps1

# Install CI requirements
pip install -r .ci/docker/requirements-ci.txt

$buildDir = "cmake-out-vulkan"
if (Test-Path -Path $buildDir) {
    Remove-Item -Path $buildDir -Recurse -Force
}
New-Item -Path $buildDir -ItemType Directory

cmake -S . -B $buildDir `
    -DCMAKE_BUILD_TYPE=Release `
    -DEXECUTORCH_BUILD_VULKAN=ON `
    -DPYTHON_EXECUTABLE=python

if ($LASTEXITCODE -ne 0) {
    Write-Host "CMake configuration failed. Exit code: $LASTEXITCODE."
    exit $LASTEXITCODE
}

cmake --build $buildDir --config Release --target vulkan_backend -j16

if ($LASTEXITCODE -ne 0) {
    Write-Host "Vulkan backend MSVC build failed. Exit code: $LASTEXITCODE."
    exit $LASTEXITCODE
}

Write-Host "Vulkan backend MSVC build completed successfully!"
