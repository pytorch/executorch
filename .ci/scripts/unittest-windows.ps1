Set-PSDebug -Trace 1
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

conda create --yes --quiet -n et python=3.12
conda activate et

install_executorch.bat
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installation was unsuccessful. Exit code: $LASTEXITCODE."
    exit $LASTEXITCODE
}

# Run pytest with coverage
pytest -n auto --cov=./ --cov-report=xml
