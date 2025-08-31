param (
    [string]$modelName,
    [string]$backend,
    [string]$buildDir = "cmake-out",
    [bool]$strict = $false
)

Set-PSDebug -Trace 1
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

.ci/scripts/setup-windows.ps1

# Build the runner
if (Test-Path -Path $buildDir) {
    Remove-Item -Path $buildDir -Recurse -Force
}
New-Item -Path $buildDir -ItemType Directory
Push-Location $buildDir
cmake .. --preset windows
cmake --build . -t executor_runner -j16 --config Release
if ($LASTEXITCODE -ne 0) {
    Write-Host "Runner build failed. Exit code: $LASTEXITCODE."
    exit $LASTEXITCODE
}
$executorBinaryPath = Join-Path -Path $buildDir -ChildPath "Release\executor_runner.exe"
Pop-Location

# Export the model
$exportParams = "--model_name", "$modelName"
if ($strict) {
    $exportParams += "--strict"
}
python -m examples.portable.scripts.export @exportParams
if ($LASTEXITCODE -ne 0) {
    Write-Host "Model export failed. Exit code: $LASTEXITCODE."
    exit $LASTEXITCODE
}

# Run the runner
& "$executorBinaryPath" --model_path="$modelName.pte"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Model execution failed. Exit code: $LASTEXITCODE."
    exit $LASTEXITCODE
}