Set-PSDebug -Trace 1
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

conda create --yes --quiet -n et python=3.12
conda activate et

#ls -Path "C:\Program Files"
#ls -Path "C:\Program Files (x86)"
#ls -Path "C:\Program Files\Microsoft Visual Studio\2022\"
#ls -Path "C:\Program Files (x86)\Microsoft Visual Studio\2022\"

# Activate the VS environment - this is required for Dynamo to work, as
# it uses the MSVC compiler.
#$vsInstallPath = "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\"
#Import-Module (Join-Path $vsInstallPath "Common7\Tools\vsdevshell\Microsoft.VisualStudio.DevShell.dll")
#Enter-VsDevShell -VsInstallPath $vsInstallPath -DevCmdArguments "-arch=x64" -SkipAutomaticLocation

install_executorch.bat
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installation was unsuccessful. Exit code: $LASTEXITCODE."
    exit $LASTEXITCODE
}

# Run pytest with coverage
# pytest -n auto --cov=./ --cov-report=xml
pytest -n auto --continue-on-collection-errors
if ($LASTEXITCODE -ne 0) {
    Write-Host "Pytest invocation was unsuccessful. Exit code: $LASTEXITCODE."
    exit $LASTEXITCODE
}
