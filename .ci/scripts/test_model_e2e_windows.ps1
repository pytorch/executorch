#!/usr/bin/env pwsh
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

param(
    [Parameter(Mandatory = $true)]
    [string]$Device,
    [Parameter(Mandatory = $true)]
    [string]$HfModel,
    [Parameter(Mandatory = $true)]
    [string]$QuantName,
    [string]$ModelDir = ".",
    [string]$ExpectedCudaVersion = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$ProgressPreference = "SilentlyContinue"

if ($Device -ne "cuda-windows") {
    throw "Unsupported device '$Device'. Expected 'cuda-windows'."
}

Write-Host "Testing model: $HfModel (quantization: $QuantName)"

$resolvedModelDir = (Resolve-Path -Path $ModelDir).Path
$modelPte = Join-Path -Path $resolvedModelDir -ChildPath "model.pte"
$cudaBlob = Join-Path -Path $resolvedModelDir -ChildPath "aoti_cuda_blob.ptd"

if (-not (Test-Path -Path $modelPte -PathType Leaf)) {
    throw "model.pte not found in '$resolvedModelDir'"
}
if (-not (Test-Path -Path $cudaBlob -PathType Leaf)) {
    throw "aoti_cuda_blob.ptd not found in '$resolvedModelDir'"
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$executorchRoot = (Resolve-Path -Path (Join-Path -Path $scriptDir -ChildPath "..\..")).Path

switch ($HfModel) {
    "mistralai/Voxtral-Mini-3B-2507" {
        $runnerTarget = "voxtral_runner"
        $runnerPath = "voxtral"
        $runnerPreset = "voxtral-cuda"
        $expectedOutput = "identity"
        $preprocessor = "voxtral_preprocessor.pte"
        $tokenizerUrl = "https://huggingface.co/mistralai/Voxtral-Mini-3B-2507/resolve/main" # @lint-ignore
        $tokenizerFile = "tekken.json"
        $audioUrl = "https://github.com/voxserv/audio_quality_testing_samples/raw/refs/heads/master/testaudio/16000/test01_20s.wav"
        $audioFile = "poem.wav"
    }
    "nvidia/parakeet-tdt" {
        $runnerTarget = "parakeet_runner"
        $runnerPath = "parakeet"
        $runnerPreset = "parakeet-cuda"
        $expectedOutput = "Phoebe"
        $preprocessor = ""
        $tokenizerUrl = ""
        $tokenizerFile = "tokenizer.model"
        $audioUrl = "https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav"
        $audioFile = "test_audio.wav"
    }
    default {
        throw "Unsupported model '$HfModel'. Supported: mistralai/Voxtral-Mini-3B-2507, nvidia/parakeet-tdt"
    }
}

function Download-IfNeeded {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Url,
        [Parameter(Mandatory = $true)]
        [string]$OutFile
    )

    if (Test-Path -Path $OutFile -PathType Leaf) {
        Write-Host "Using existing file: $OutFile"
        return
    }
    Write-Host "Downloading $Url -> $OutFile"
    Invoke-WebRequest -Uri $Url -OutFile $OutFile
}

Push-Location $executorchRoot
try {
    Write-Host "::group::Check CUDA toolchain"
    $nvccOutput = nvcc --version | Out-String
    Write-Host $nvccOutput
    nvidia-smi
    if (-not [string]::IsNullOrWhiteSpace($ExpectedCudaVersion)) {
        $versionMatch = [Regex]::Match($nvccOutput, "release\s+(\d+\.\d+)")
        if (-not $versionMatch.Success) {
            throw "Failed to parse CUDA version from nvcc output."
        }
        $actualCudaVersion = $versionMatch.Groups[1].Value
        if ($actualCudaVersion -ne $ExpectedCudaVersion) {
            throw "CUDA version mismatch. Expected: $ExpectedCudaVersion, Actual: $actualCudaVersion"
        }
        Write-Host "CUDA version check passed: $actualCudaVersion"
    }
    Write-Host "::endgroup::"

    Write-Host "::group::Build ExecuTorch (CUDA)"
    $numCores = [Math]::Max([Environment]::ProcessorCount - 1, 1)
    cmake --preset llm-release-cuda
    cmake --build cmake-out --target install --config Release -j $numCores
    Write-Host "::endgroup::"

    Write-Host "::group::Build $runnerTarget"
    Push-Location (Join-Path -Path $executorchRoot -ChildPath "examples\models\$runnerPath")
    try {
        cmake --preset $runnerPreset
        cmake --build (Join-Path -Path $executorchRoot -ChildPath "cmake-out\examples\models\$runnerPath") --target $runnerTarget --config Release -j $numCores
    }
    finally {
        Pop-Location
    }
    Write-Host "::endgroup::"

    Write-Host "::group::Prepare Artifacts"
    if ($preprocessor -ne "") {
        $preprocessorPath = Join-Path -Path $resolvedModelDir -ChildPath $preprocessor
        if (-not (Test-Path -Path $preprocessorPath -PathType Leaf)) {
            throw "Required preprocessor artifact not found: $preprocessorPath"
        }
    }
    if ($tokenizerFile -ne "") {
        $tokenizerPath = Join-Path -Path $resolvedModelDir -ChildPath $tokenizerFile
        if (-not (Test-Path -Path $tokenizerPath -PathType Leaf) -and $tokenizerUrl -eq "") {
            throw "Required tokenizer artifact not found: $tokenizerPath"
        }
    }
    if ($tokenizerUrl -ne "") {
        Download-IfNeeded -Url "$tokenizerUrl/$tokenizerFile" -OutFile (Join-Path -Path $resolvedModelDir -ChildPath $tokenizerFile)
    }
    if ($audioUrl -ne "") {
        Download-IfNeeded -Url $audioUrl -OutFile (Join-Path -Path $resolvedModelDir -ChildPath $audioFile)
    }
    Get-ChildItem -Path $resolvedModelDir
    Write-Host "::endgroup::"

    Write-Host "::group::Run $runnerTarget"
    $runnerExeCandidates = @(
        (Join-Path -Path $executorchRoot -ChildPath "cmake-out\examples\models\$runnerPath\Release\$runnerTarget.exe"),
        (Join-Path -Path $executorchRoot -ChildPath "cmake-out\examples\models\$runnerPath\$runnerTarget.exe")
    )
    $runnerExe = $runnerExeCandidates | Where-Object { Test-Path -Path $_ -PathType Leaf } | Select-Object -First 1
    if (-not $runnerExe) {
        throw "Runner executable not found. Checked: $($runnerExeCandidates -join ', ')"
    }

    $runnerArgs = @("--model_path", $modelPte, "--data_path", $cudaBlob)
    switch ($HfModel) {
        "mistralai/Voxtral-Mini-3B-2507" {
            $runnerArgs += @(
                "--temperature", "0",
                "--tokenizer_path", (Join-Path -Path $resolvedModelDir -ChildPath $tokenizerFile),
                "--audio_path", (Join-Path -Path $resolvedModelDir -ChildPath $audioFile),
                "--processor_path", (Join-Path -Path $resolvedModelDir -ChildPath $preprocessor)
            )
        }
        "nvidia/parakeet-tdt" {
            $runnerArgs = @(
                "--model_path", $modelPte,
                "--audio_path", (Join-Path -Path $resolvedModelDir -ChildPath $audioFile),
                "--tokenizer_path", (Join-Path -Path $resolvedModelDir -ChildPath $tokenizerFile),
                "--data_path", $cudaBlob
            )
        }
    }

    $stdoutFile = Join-Path -Path $env:TEMP -ChildPath ("et_runner_stdout_{0}.log" -f ([Guid]::NewGuid().ToString("N")))
    $stderrFile = Join-Path -Path $env:TEMP -ChildPath ("et_runner_stderr_{0}.log" -f ([Guid]::NewGuid().ToString("N")))
    try {
        $proc = Start-Process `
            -FilePath $runnerExe `
            -ArgumentList $runnerArgs `
            -NoNewWindow `
            -Wait `
            -PassThru `
            -RedirectStandardOutput $stdoutFile `
            -RedirectStandardError $stderrFile

        $stdout = if (Test-Path -Path $stdoutFile -PathType Leaf) { Get-Content -Path $stdoutFile -Raw } else { "" }
        $stderr = if (Test-Path -Path $stderrFile -PathType Leaf) { Get-Content -Path $stderrFile -Raw } else { "" }
        $output = @($stdout, $stderr) -join [Environment]::NewLine
        $exitCode = $proc.ExitCode
    }
    finally {
        Remove-Item -Path $stdoutFile -ErrorAction SilentlyContinue
        Remove-Item -Path $stderrFile -ErrorAction SilentlyContinue
    }
    Write-Host "Runner output:"
    Write-Host $output

    if ($exitCode -ne 0) {
        throw "Runner exited with code $exitCode`n$output"
    }

    if ($expectedOutput -ne "" -and $output -notmatch [Regex]::Escape($expectedOutput)) {
        throw "Expected output '$expectedOutput' not found in runner output"
    }
    Write-Host "Success: '$expectedOutput' found in output"
    Write-Host "::endgroup::"
}
finally {
    Pop-Location
}
