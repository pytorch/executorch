# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

$ErrorActionPreference = "Stop"

conda create --yes --quiet -n et python=3.12
conda activate et

# Install CI requirements
pip install -r .ci/docker/requirements-ci.txt

# Provision the QNN SDK
if ($env:QNN_SDK_ROOT -and (Test-Path -Path $env:QNN_SDK_ROOT)) {
    Write-Host "Using existing QNN SDK at $env:QNN_SDK_ROOT"
} else {
    # Resolve the QNN_VERSION and QNN_ZIP_URL
    $qnnInfo = python -c "import sys; sys.path.insert(0, r'backends\qualcomm\scripts'); import download_qnn_sdk as d; print(d.QNN_VERSION); print(d.QNN_ZIP_URL)"
    if ($LASTEXITCODE -ne 0 -or $qnnInfo.Count -lt 2) {
        Write-Error "Failed to read QNN_VERSION and QNN_ZIP_URL from download_qnn_sdk.py."
        exit 1
    }
    $qnnVersion = $qnnInfo[0].Trim()
    $qnnZipUrl  = $qnnInfo[1].Trim()

    $qnnInstallDir = Join-Path $env:TEMP "qnn"
    New-Item -Path $qnnInstallDir -ItemType Directory -Force | Out-Null
    $qnnZip = Join-Path $env:TEMP "qnn_sdk.zip"
    Write-Host "Downloading QNN SDK v$qnnVersion ..."
    $ProgressPreference = "SilentlyContinue"
    Invoke-WebRequest -Uri $qnnZipUrl -OutFile $qnnZip
    Write-Host "Extracting QNN SDK ..."
    Expand-Archive -Path $qnnZip -DestinationPath $qnnInstallDir -Force
    Remove-Item -Path $qnnZip -Force

    $env:QNN_SDK_ROOT = Join-Path $qnnInstallDir "qairt\$qnnVersion"
    Write-Host "Set QNN_SDK_ROOT=$env:QNN_SDK_ROOT"
}

if (-not (Test-Path -Path (Join-Path $env:QNN_SDK_ROOT "include\QNN"))) {
    Write-Error "QNN SDK layout unexpected: missing include\QNN under $env:QNN_SDK_ROOT"
    exit 1
}

# Test x86_64 Windows host build
.\backends\qualcomm\scripts\build.ps1 -SkipArm64Windows -Release

$x86Artifacts = @(
    "build-x86_64-windows\backends\qualcomm\Release\PyQnnManagerAdaptor*.pyd",
    "build-x86_64-windows\backends\qualcomm\Release\qnn_executorch_backend.dll",
    "build-x86_64-windows\examples\qualcomm\executor_runner\Release\qnn_executor_runner.exe"
)
foreach ($artifact in $x86Artifacts) {
    if (-not (Get-ChildItem -Path $artifact -ErrorAction SilentlyContinue)) {
        Write-Error "ERROR: x86_64 artifact not found: $artifact"
        exit 1
    }
}

# The ARM64 MSVC toolchain is currently not installed in the Windows CI
# environment. Enabling this build configuration results in build failures
# due to the missing ARM64 platform definition.
# `.\backends\qualcomm\scripts\build.ps1 -SkipX86Windows -Release`
#
# Temporarily disable this build option until ARM64 MSVC support is available
# in CI. The configuration can be re-enabled in a future update.

Write-Host "PASSED: QNN backend Windows MSVC build completed"
