param (
    [string]$editable = "false",
    [string]$cpuOnly = "false"
)

conda create --yes --quiet -n et python=3.12
conda activate et

# Activate the VS environment - this is required for Dynamo to work, as it uses MSVC.
# There are a bunch of environment variables that it requires.
# See https://learn.microsoft.com/en-us/cpp/build/building-on-the-command-line.
& "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64

# Install test dependencies
pip install -r .ci/docker/requirements-ci.txt

# The Windows CI image ships CUDA toolkits on PATH, so install_executorch
# (setup.py) auto-enables EXECUTORCH_BUILD_CUDA whenever the detected nvcc
# version is in SUPPORTED_CUDA_VERSIONS. CPU-only jobs install CPU torch, so a
# CUDA build of _portable_lib then fails to load its CUDA DLLs at import time
# ("DLL load failed while importing _portable_lib"). Force a CPU-only build
# when the caller asks for it.
if ($cpuOnly -eq 'true') {
    $env:CMAKE_ARGS = "$env:CMAKE_ARGS -DEXECUTORCH_BUILD_CUDA=OFF"
}

if ($editable -eq 'true') {
    install_executorch.bat --editable
} else {
    install_executorch.bat
}
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installation was unsuccessful. Exit code: $LASTEXITCODE."
    exit $LASTEXITCODE
}
