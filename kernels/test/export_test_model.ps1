param (
    [string]$Modules,
    [string]$OutDir,
    [string]$CondaEnv
)

Set-PSDebug -Trace 1

# Activate the VS dev environment - needed for dynamo. Try to use vswhere to locate the install. If not,
# fall back to a reasonable guess for the build tools, which also happens to match the CLI setup.
$vswherePath = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vswherePath) {
    $vsInstallPath = & $vswherePath -latest -property installationPath
} else {
    $vsInstallPath = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\"
}

& "$vsInstallPath\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -SkipAutomaticLocation

conda activate $CondaEnv

$Modules = $Modules.Replace(" ", ",")
echo "Modules: $Modules"
python -m test.models.export_program --modules "$Modules" --outdir "$OutDir"
