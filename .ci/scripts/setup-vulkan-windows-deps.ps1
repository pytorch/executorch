# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Install glslc (the Vulkan shader compiler) on Windows from a pinned shaderc
# build, and make sure it is on PATH. glslc is the only build-time Vulkan
# dependency -- the Vulkan headers and the volk loader come from the in-tree
# submodules -- so this avoids both the heavyweight LunarG SDK installer and a
# conda dependency. glslc.exe in these builds is a single static executable (it
# only imports kernel32.dll), so extracting it from the archive is enough.
#
# The binary comes from Google's shaderc CI artifacts. They are not QA'd, so the
# build is pinned by URL and SHA-256. To use a mirror, set GLSLC_ZIP_URL (and
# GLSLC_ZIP_SHA256 if the archive differs).

param(
    [string]$InstallDir = ""
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# shaderc v2026.5-dev, Windows VS 2022 amd64 release, build 53.
$zipUrl = if ($env:GLSLC_ZIP_URL) { $env:GLSLC_ZIP_URL } else { "https://storage.googleapis.com/shaderc/artifacts/prod/graphics_shader_compiler/shaderc/windows-vs2022-amd64-release/continuous/53/20261028-151553/install.zip" }
$zipSha256 = if ($env:GLSLC_ZIP_SHA256) { $env:GLSLC_ZIP_SHA256 } else { "2d5358873b2726a15ed9ab6fe45143a76913f4d62a8434a6f6a1a471bdfe1270" }
$zipEntry = "install/bin/glslc.exe"

if (-not $InstallDir) {
    $baseDir = if ($env:RUNNER_TEMP) { $env:RUNNER_TEMP } else { $env:TEMP }
    $InstallDir = Join-Path $baseDir "glslc"
}
$binDir = Join-Path $InstallDir "bin"
$glslcExe = Join-Path $binDir "glslc.exe"

if (-not (Test-Path $glslcExe)) {
    New-Item -ItemType Directory -Force -Path $binDir | Out-Null
    $zipPath = Join-Path $InstallDir "shaderc-install.zip"

    Write-Host "Downloading shaderc from $zipUrl..."
    for ($attempt = 1; ; $attempt++) {
        try {
            Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath -UseBasicParsing
            break
        } catch {
            if ($attempt -ge 3) {
                Write-Error "Failed to download shaderc after $attempt attempts: $($_.Exception.Message)"
                exit 1
            }
            Write-Host "Download failed (attempt $attempt): $($_.Exception.Message); retrying..."
            Start-Sleep -Seconds (5 * $attempt)
        }
    }

    $actualSha256 = (Get-FileHash -Algorithm SHA256 -Path $zipPath).Hash
    if ($actualSha256 -ne $zipSha256) {
        Write-Error "SHA-256 mismatch for $zipUrl (expected $zipSha256, got $actualSha256)"
        exit 1
    }

    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [System.IO.Compression.ZipFile]::OpenRead($zipPath)
    try {
        $entry = $zip.Entries | Where-Object { $_.FullName -eq $zipEntry }
        if (-not $entry) {
            Write-Error "$zipEntry not found in $zipUrl"
            exit 1
        }
        [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $glslcExe, $true)
    } finally {
        $zip.Dispose()
    }
    Remove-Item -Path $zipPath -Force
}

# Expose glslc to the current process and, when running as a GitHub Actions step,
# to subsequent steps.
$env:PATH = "${binDir};${env:PATH}"
if ($env:GITHUB_PATH) {
    Add-Content -Path $env:GITHUB_PATH -Value $binDir
}

$glslc = Get-Command glslc -ErrorAction SilentlyContinue
if (-not $glslc) {
    Write-Error "glslc not found on PATH after installing it to $binDir"
    exit 1
}

Write-Host "glslc available at $($glslc.Source)"
& glslc --version
