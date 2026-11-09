# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Installs the published Windows CUDA wheel for one CUDA train and runs its smoke test
# against a program a Linux job exported for that train. This is the route a Windows user
# has: lowering for CUDA cannot run on Windows, so the program comes from Linux and only
# the wheel's C++ SDK runs it. The artifact carries both, so CI runs a CUDA program through
# the wheel the build job made, not one rebuilt here.
param(
    [Parameter(Mandatory = $true)][string]$Train,
    [Parameter(Mandatory = $true)][string]$Artifacts
)
$ErrorActionPreference = "Stop"

# The runner's powershell.exe is Windows PowerShell 5.1, where a native command that
# fails does not stop the script ($PSNativeCommandUseErrorActionPreference is 7.3+ only).
# Without this a failing test printed its traceback and the job still passed.
function Invoke-Native {
    param([Parameter(Mandatory = $true)][scriptblock]$Command)
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "exit code ${LASTEXITCODE}: $Command"
    }
}

if (-not (Test-Path (Join-Path $Artifacts "model.pte"))) {
    throw "no model.pte in $Artifacts; the Linux export job did not hand over a program"
}
# The wheel for this train, by its +cuXYZ tag, so a wheel from another train cannot stand in.
$cu = "cu" + $Train.Replace(".", "")
$wheels = @(Get-ChildItem (Join-Path $Artifacts "*.whl") | Where-Object { $_.Name -like "*+$cu-*" })
if ($wheels.Count -ne 1) {
    throw "expected one $cu wheel in $Artifacts, found $($wheels.Count): $((Get-ChildItem (Join-Path $Artifacts '*.whl')).Name -join ', ')"
}
$wheel = $wheels[0]
# The interpreter the wheel was built for, from its cpXY tag.
$tag = [regex]::Match($wheel.Name, '-cp(\d)(\d+)-')
if (-not $tag.Success) { throw "no cpXY tag in $($wheel.Name)" }
$python = "$($tag.Groups[1].Value).$($tag.Groups[2].Value)"

# The GPU runners come without a driver; the same step PyTorch's Windows CUDA tests run.
& "$PSScriptRoot\..\install_nvidia_driver_windows.ps1"

# The smoke test reads device code with the toolkit's cuobjdump, and the model's own
# library loads the toolkit's CUDA runtime, so the train's toolkit is still installed:
# the runner image's if it has it, else NVIDIA's redistributable archives.
$installed = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v$Train"
if (Test-Path (Join-Path $installed "bin\nvcc.exe")) {
    $cudaHome = $installed
} else {
    $cudaHome = Join-Path $env:RUNNER_TEMP "cuda-$Train"
    Invoke-Native { python .ci/scripts/wheel/install_cuda_redist.py --train $Train --platform windows-x86_64 --dest $cudaHome }
}
$env:CUDA_HOME = $cudaHome
$env:CUDA_PATH = $cudaHome
$env:PATH = "$cudaHome\bin\x64;$cudaHome\bin;$env:PATH"
Invoke-Native { cuobjdump --version }

Invoke-Native { conda create --yes --quiet -n et python=$python }
conda activate et
# For the smoke test's C++ consumer builds.
& "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64

# The torch, torchao nightly and torchvision the smoke test needs: CPU torch, since the
# Python side of this wheel carries no CUDA; the torchao the wheel declares, which PyPI does
# not carry; and with --example the torchvision the model run needs, as the release build's
# pre-build hook does.
Invoke-Native { python install_requirements.py --example }
# With its declared dependencies, as a user installs it: the smoke test's clean-install
# check fails on any module the wheel needs and does not declare.
Invoke-Native { pip install $wheel.FullName }

$env:EXECUTORCH_CUDA_WINDOWS_ARTIFACTS = (Resolve-Path $Artifacts).Path
# The published wheel this job installed, for the checks that read the wheel file itself
# (its platform tag); there is no local build to find it in.
$env:WHEEL_DIR = $wheel.DirectoryName
$env:PYTHONIOENCODING = "utf-8"
# From outside the checkout, so the checks inspect the installed wheel, not the sources.
$scripts = (Resolve-Path .ci/scripts/wheel).Path
Push-Location $env:RUNNER_TEMP
try {
    Invoke-Native { python (Join-Path $scripts "test_cuda_windows.py") }
} finally {
    Pop-Location
}
