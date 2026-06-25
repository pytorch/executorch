# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Install glslc (the Vulkan shader compiler) on Windows via conda-forge's
# shaderc package, and make sure it is on PATH. glslc is the only build-time
# Vulkan dependency -- the Vulkan headers and the volk loader come from the
# in-tree submodules -- so this avoids depending on the heavyweight LunarG SDK
# installer. Requires conda to be available (the callers create/activate an env).

$ErrorActionPreference = "Stop"

Write-Host "Installing shaderc (provides glslc) from conda-forge..."
conda install -y -c conda-forge shaderc
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to install shaderc from conda-forge (exit ${LASTEXITCODE})"
    exit 1
}

$glslc = Get-Command glslc -ErrorAction SilentlyContinue
if (-not $glslc) {
    Write-Error "glslc not found on PATH after installing shaderc"
    exit 1
}

# Expose glslc to the current process and, when running as a GitHub Actions step,
# to subsequent steps.
$glslcDir = Split-Path -Parent $glslc.Source
$env:PATH = "${glslcDir};${env:PATH}"
if ($env:GITHUB_PATH) {
    Add-Content -Path $env:GITHUB_PATH -Value $glslcDir
}

Write-Host "glslc available at $($glslc.Source)"
& glslc --version
