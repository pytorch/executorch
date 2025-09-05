param (
    [string]$modelName,
    [string]$backend,
    [string]$buildDir = "cmake-out",
    [bool]$strict = $false
)

Set-PSDebug -Trace 1
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

function ExportModel-Portable {
    param (
        [string]$model_name,
        [bool]$strict
    )

    $exportParams = "--model_name", "$modelName"
    if ($strict) {
        $exportParams += "--strict"
    }
    python -m examples.portable.scripts.export @exportParams | Write-Host
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Model export failed. Exit code: $LASTEXITCODE."
        exit $LASTEXITCODE
    }

    "$modelName.pte"
}

function ExportModel-Xnnpack {
    param (
        [string]$model_name,
        [bool]$quantize
    )

    if $(quantize) {
        python -m examples.xnnpack.aot_compiler --model_name="${MODEL_NAME}" --delegate --quantize | Write-Host
        $modelFile = "$($modelName)_xnnpack_q8.pte"
    } else {
        python -m examples.xnnpack.aot_compiler --model_name="${MODEL_NAME}" --delegate | Write-Host
        $modelFile = "$($modelName)_xnnpack_fp32.pte"
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Model export failed. Exit code: $LASTEXITCODE."
        exit $LASTEXITCODE
    }

    $modelFile
}

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
switch ($backend) {
    "portable" {
        $model_path = ExportModel-Portable -model_name $modelName -strict $strict
    }
    "xnnpack-f32" {
        $model_path = ExportModel-Xnnpack -model_name $modelName -quantize $false
    }
    "xnnpack-q8" {
        $model_path = ExportModel-Xnnpack -model_name $modelName -quantize $true
    }
    default {
        Write-Host "Unknown backend $backend."
        exit 1
    }
}

# Run the runner
& "$executorBinaryPath" --model_path="$model_path"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Model execution failed. Exit code: $LASTEXITCODE."
    exit $LASTEXITCODE
}
