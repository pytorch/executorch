param (
    [string]$editable = $false
)

Set-PSDebug -Trace 1
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

# Run pytest with coverage
# pytest -n auto --cov=./ --cov-report=xml
pytest -v --full-trace -c pytest-windows.ini
if ($LASTEXITCODE -ne 0) {
    Write-Host "Pytest invocation was unsuccessful. Exit code: $LASTEXITCODE."
    exit $LASTEXITCODE
}
