param (
    [string]$editable = $false
)

conda create --yes --quiet -n et python=3.12
conda activate et

# Activate the VS environment - this is required for Dynamo to work, as it uses MSVC.
# There are a bunch of environment variables that it requires.
# See https://learn.microsoft.com/en-us/cpp/build/building-on-the-command-line.
& "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64

# Install test dependencies
pip install -r .ci/docker/requirements-ci.txt

# Create a symlink to work around path length issues when building submodules (tokenizers).
Push-Location
New-Item -ItemType SymbolicLink -Path "C:\_et" -Target "$CWD"
cd C:\_et

if ($editable -eq 'true') {
    install_executorch.bat --editable
} else {
    install_executorch.bat
}
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installation was unsuccessful. Exit code: $LASTEXITCODE."
    exit $LASTEXITCODE
}

Pop-Location
