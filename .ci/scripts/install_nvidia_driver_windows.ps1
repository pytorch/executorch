# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Installs the NVIDIA display driver on a Windows GPU runner. The runner images carry the
# GPU but no driver, so without this nvcuda.dll is missing and every CUDA call fails with
# cudaErrorInsufficientDriver. The same driver, from the same place, that PyTorch's Windows
# CUDA smoke tests install first (.ci/pytorch/windows/internal/driver_update.bat in
# pytorch/pytorch), so both projects test against one driver.
$ErrorActionPreference = "Stop"
$version = "580.88"
$installer = Join-Path $env:RUNNER_TEMP "$version-data-center-tesla-desktop-win10-win11-64bit-dch-international.exe"
$url = "https://ossci-windows.s3.amazonaws.com/$(Split-Path $installer -Leaf)"

$nvcuda = Join-Path $env:SystemRoot "System32\nvcuda.dll"
if (Test-Path $nvcuda) {
    Write-Host "NVIDIA driver already present: $((Get-Item $nvcuda).VersionInfo.FileVersion)"
    exit 0
}
curl.exe --retry 3 -fsSL $url --output $installer
if ($LASTEXITCODE -ne 0) { throw "downloading the NVIDIA driver from $url failed" }
$process = Start-Process -FilePath $installer -ArgumentList "-s", "-noreboot" -Wait -PassThru
Remove-Item $installer -ErrorAction SilentlyContinue
if ($process.ExitCode -ne 0) { throw "the NVIDIA driver installer exited with $($process.ExitCode)" }
if (-not (Test-Path $nvcuda)) { throw "the NVIDIA driver installed but $nvcuda is missing" }
Write-Host "NVIDIA driver $version installed: $((Get-Item $nvcuda).VersionInfo.FileVersion)"
