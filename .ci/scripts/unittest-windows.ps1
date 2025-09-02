param (
    [string]$editable = $false
)

Set-PSDebug -Trace 1
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

.ci/scripts/setup-windows.ps1 -editable $editable

# Run pytest with coverage
# pytest -n auto --cov=./ --cov-report=xml
pytest -v --full-trace -c pytest-windows.ini -n auto
if ($LASTEXITCODE -ne 0) {
    Write-Host "Pytest invocation was unsuccessful. Exit code: $LASTEXITCODE."
    exit $LASTEXITCODE
}
