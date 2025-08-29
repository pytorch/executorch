param (
    [string]$editable
)

Set-PSDebug -Trace 1
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

conda create --yes --quiet -n et python=3.12
conda activate et

# Activate the VS environment - this is required for Dynamo to work, as it uses MSVC.
# There are a bunch of environment variables that it requires.
# See https://learn.microsoft.com/en-us/cpp/build/building-on-the-command-line.
& "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64

if ($editable -eq 'true') {
    install_executorch.bat --editable
} else {
    install_executorch.bat
}
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installation was unsuccessful. Exit code: $LASTEXITCODE."
    exit $LASTEXITCODE
}

# Run pytest with coverage
# pytest -n auto --cov=./ --cov-report=xml
pytest -n auto --continue-on-collection-errors -vv --timeout=600 --full-trace
if ($LASTEXITCODE -ne 0) {
    Write-Host "Pytest invocation was unsuccessful. Exit code: $LASTEXITCODE."
    exit $LASTEXITCODE
}
