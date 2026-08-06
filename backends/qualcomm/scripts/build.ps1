# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Windows PowerShell build script for the Qualcomm AI Engine Direct backend.
# Mirrors backends/qualcomm/scripts/build.sh but targets Windows x86-64 (host)
# and Windows-on-ARM64 (cross-compiled via LLVM/Clang for arm64-windows).
#
# Usage:
#   .\backends\qualcomm\scripts\build.ps1 [options]
#
# Options:
#   -SkipX86Windows      Skip the x86-64 Windows host build (AOT + pybind)
#   -SkipArm64Windows    Skip the ARM64 Windows cross-compiled build
#   -EnableHexagon       Enable Hexagon DSP direct-mode skel library build
#   -NoClean             Incremental build (skip rm of build dir)
#   -Release             Use Release build type (default: RelWithDebInfo)
#   -JobNumber <N>       Parallel jobs for cmake --build (default: 16)
#   -DspType <N>         DSP domain for Hexagon direct-mode (default: 3 = CDSP)

param(
    [switch]$SkipX86Windows,
    [switch]$SkipArm64Windows,
    [switch]$EnableHexagon,
    [switch]$NoClean,
    [switch]$Release,
    [int]$JobNumber = 16,
    [int]$DspType = 3
)

# Stop on any error, mirroring bash's set -e.
$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------
# Validate required environment variables
# ---------------------------------------------------------------------------

if (-not $env:QNN_SDK_ROOT) {
    Write-Error "Please set `$env:QNN_SDK_ROOT to the QNN SDK root directory."
    exit 1
}

# ARM64 cross-compile requires the LLVM toolchain that ships with VS.
# The caller is expected to have run vcvarsall.bat or Setup-BuildEnv.ps1 so
# that clang-cl and lld-link are on PATH.  We validate lazily inside the
# build-arm64-windows block rather than up front, because the user may only
# want the x86 host build.

if ($EnableHexagon) {
    foreach ($var in @('ANDROID_NDK_ROOT', 'HEXAGON_SDK_ROOT', 'HEXAGON_TOOLS_ROOT', 'DSP_VERSION')) {
        if (-not (Get-Item "env:$var" -ErrorAction SilentlyContinue)) {
            Write-Error "Hexagon build requires `$env:$var to be set."
            exit 1
        }
    }
}

# ---------------------------------------------------------------------------
# Derived settings
# ---------------------------------------------------------------------------

$BuildType     = if ($Release) { 'Release' } else { 'RelWithDebInfo' }
$Clean         = -not $NoClean

# Resolve the repo root as the directory three levels above this script
# (backends/qualcomm/scripts/ -> backends/qualcomm/ -> backends/ -> repo root).
$PrjRoot = (Resolve-Path "$PSScriptRoot\..\..\..").Path

$CmakeX86    = 'build-x86_64-windows'
$CmakeArm64  = 'build-arm64-windows'
$CmakeHexagon = 'build-hexagon'

# Preserve the current Python executable path before launching the Visual
# Studio developer shell (Launch-VsDevShell.ps1)
#
# When launching, it rewrites the $env:PATH to inject the MSVC toolchain.
# Depending on the Visual Studio version, these changes can override or remove
# the Python entry from $env:PATH
#
# To ensure the Python interpreter remains accessible, capture and store its
# absolute path before invoking Launch-VsDevShell.ps1.
$PyCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $PyCmd) {
    Write-Error "No 'python' found on PATH. Requires 'python' to continue the QNN Windows MSVC build."
    exit 1
}
$PythonExe = $PyCmd.Source
Write-Host "Using Python: $PythonExe" -ForegroundColor Cyan

# ---------------------------------------------------------------------------
# Helper: clean or prepare a build directory
# ---------------------------------------------------------------------------
function Prepare-BuildDir([string]$BuildRoot) {
    if ($Clean) {
        if (Test-Path $BuildRoot) {
            Write-Host "Removing $BuildRoot ..."
            Remove-Item -Recurse -Force $BuildRoot
        }
        New-Item -ItemType Directory -Path $BuildRoot | Out-Null
    } else {
        # Incremental: flatcc must be rebuilt for the host platform.
        # On Windows flatcc is a CMake ExternalProject; there is no Makefile
        # to run 'make clean' against, so we remove its stamp files so CMake
        # re-runs it on the next configure.
        $FlatccStamp = Join-Path $BuildRoot 'third-party\flatcc\src\flatcc_ep-stamp'
        if (Test-Path $FlatccStamp) {
            Write-Host "Removing flatcc stamp files for incremental rebuild ..."
            Remove-Item -Recurse -Force $FlatccStamp
        }
    }
}

# ---------------------------------------------------------------------------
# Helper: run cmake configure + build, aborting on failure
# ---------------------------------------------------------------------------
function Run-CMake([string[]]$ConfigArgs, [string]$BuildDir, [string]$Target = 'install') {
    Write-Host "`n=== cmake configure ===" -ForegroundColor Cyan
    & cmake @ConfigArgs
    if ($LASTEXITCODE -ne 0) { throw "cmake configure failed (exit $LASTEXITCODE)" }

    Write-Host "`n=== cmake build (target: $Target) ===" -ForegroundColor Cyan
    if ($Target -eq 'install') {
        & cmake --build $BuildDir --config $BuildType -j $JobNumber --target install
    } else {
        & cmake --build $BuildDir --config $BuildType -j $JobNumber
    }
    if ($LASTEXITCODE -ne 0) { throw "cmake build failed (exit $LASTEXITCODE)" }
}

# ---------------------------------------------------------------------------
# Helper: locate an installed Visual Studio and its matching CMake generator
# ---------------------------------------------------------------------------
function Get-InstalledVsGenerator() {
    $ProgramRoots = @(${env:ProgramFiles(x86)}, $env:ProgramFiles) | Where-Object { $_ }
    $VsInstalls = @(
        @{ Dir = "18"; Generator = "Visual Studio 18 2026" },
        @{ Dir = "2022"; Generator = "Visual Studio 17 2022" }
    )
    $Editions = @("Enterprise", "Professional", "Community", "Preview", "BuildTools")
    foreach ($root in $ProgramRoots) {
        foreach ($vs in $VsInstalls) {
            foreach ($edition in $Editions) {
                $DevShell = "$root\Microsoft Visual Studio\$($vs.Dir)\$edition\Common7\Tools\Launch-VsDevShell.ps1"
                if (Test-Path $DevShell) {
                    Write-Host "Using VS dev shell: $DevShell" -ForegroundColor Cyan
                    return [PSCustomObject]@{ Generator = $vs.Generator; DevShell = $DevShell }
                }
            }
        }
    }
    throw "No supported Visual Studio installation found (2026 or 2022)."
}

# ---------------------------------------------------------------------------
# Helper: validate that the active CMake supports the selected generator
#
# Must be executed after activating the Visual Studio Developer Shell so the
# `cmake` resolved from PATH matches the build environment.
# ---------------------------------------------------------------------------
function Assert-GeneratorSupported([string]$Generator) {
    if ($Generator -eq "Visual Studio 18 2026") {
        $cmakeVersion = [version](& cmake --version |
            Select-String -Pattern '(\d+\.\d+\.\d+)').Matches[0].Value
        if ($cmakeVersion.Major -lt 4) {
            throw "'Visual Studio 18 2026' requires CMake >= 4, but found CMake version $cmakeVersion"
        }
    }
}

# ---------------------------------------------------------------------------
# Block 1: ARM64 Windows cross-compiled build  (build-arm64-windows/)
#
# Cross-compiles the ExecuTorch runtime + QNN backend for arm64-windows using
# the LLVM/Clang toolchain bundled with Visual Studio.  This produces the
# on-device libraries and example runners for Windows-on-ARM devices.
#
# Differences from the Linux Android block:
#   - No Android NDK toolchain file; instead we set CMAKE_SYSTEM_NAME=Windows
#     and CMAKE_SYSTEM_PROCESSOR=ARM64 so CMake selects the MSVC/Clang-CL
#     cross-compiler for arm64.
#   - No ANDROID_ABI / ANDROID_PLATFORM flags.
#   - Example runners are built for Windows (no adb push needed).
# ---------------------------------------------------------------------------
if (-not $SkipArm64Windows) {
    $BuildRoot = Join-Path $PrjRoot $CmakeArm64

    # Activate the VS environment for ARM64 MSVC toolchain
    $VsGenerator = Get-InstalledVsGenerator
    & $VsGenerator.DevShell -Arch arm64
    Assert-GeneratorSupported $VsGenerator.Generator

    Prepare-BuildDir $BuildRoot

    $ConfigArgs = @(
        $PrjRoot,
        "-G", $VsGenerator.Generator,
        "-A", "ARM64",
        "-DCMAKE_SYSTEM_NAME=Windows",
        "-DCMAKE_SYSTEM_PROCESSOR=ARM64",
        "-DCMAKE_BUILD_TYPE=$BuildType",
        "-DCMAKE_INSTALL_PREFIX=$BuildRoot",
        "-DPYTHON_EXECUTABLE=$PythonExe",
        "-DQNN_SDK_ROOT=$env:QNN_SDK_ROOT",
        "-DEXECUTORCH_BUILD_QNN=ON",
        "-DEXECUTORCH_BUILD_DEVTOOLS=ON",
        "-DEXECUTORCH_BUILD_EXTENSION_LLM=ON",
        "-DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=ON",
        "-DEXECUTORCH_BUILD_EXTENSION_MODULE=ON",
        "-DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON",
        "-DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON",
        "-DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON",
        "-DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON",
        "-DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON",
        "-DEXECUTORCH_ENABLE_EVENT_TRACER=ON",
        "-DEXECUTORCH_ENABLE_LOGGING=ON",
        "-B$BuildRoot"
    )
    Run-CMake -ConfigArgs $ConfigArgs -BuildDir $BuildRoot

    # Build QNN example runners for arm64-windows
    $ExampleRoot  = Join-Path $PrjRoot 'examples\qualcomm'
    $ExampleBuild = Join-Path $BuildRoot 'examples\qualcomm'
    $CmakePrefixPath = "$BuildRoot;$BuildRoot\third-party\gflags"

    $DirectModeFlag = if ($EnableHexagon) { '-DBUILD_DIRECT_MODE=ON' } else { '-DBUILD_DIRECT_MODE=OFF' }

    $ExampleArgs = @(
        $ExampleRoot,
        "-G", $VsGenerator.Generator,
        "-A", "ARM64",
        "-DCMAKE_SYSTEM_NAME=Windows",
        "-DCMAKE_SYSTEM_PROCESSOR=ARM64",
        "-DCMAKE_BUILD_TYPE=$BuildType",
        "-DCMAKE_PREFIX_PATH=$CmakePrefixPath",
        "-DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH",
        "-DPYTHON_EXECUTABLE=$PythonExe",
        "-DEXECUTORCH_ENABLE_LOGGING=ON",
        "-DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON",
        "-DSUPPORT_REGEX_LOOKAHEAD=ON",
        "-DBUILD_TESTING=OFF",
        "-DDSP_TYPE=$DspType",
        $DirectModeFlag,
        "-B$ExampleBuild"
    )
    Write-Host "`n=== cmake configure (examples/qualcomm arm64-windows) ===" -ForegroundColor Cyan
    & cmake @ExampleArgs
    if ($LASTEXITCODE -ne 0) { throw "cmake configure (examples/qualcomm) failed" }
    & cmake --build $ExampleBuild --config $BuildType -j $JobNumber
    if ($LASTEXITCODE -ne 0) { throw "cmake build (examples/qualcomm) failed" }

    # Build Llama runner for arm64-windows
    $LlamaRoot  = Join-Path $PrjRoot 'examples\models\llama'
    $LlamaBuild = Join-Path $BuildRoot 'examples\models\llama'

    $LlamaArgs = @(
        $LlamaRoot,
        "-G", $VsGenerator.Generator,
        "-A", "ARM64",
        "-DCMAKE_SYSTEM_NAME=Windows",
        "-DCMAKE_SYSTEM_PROCESSOR=ARM64",
        "-DCMAKE_BUILD_TYPE=$BuildType",
        "-DCMAKE_PREFIX_PATH=$CmakePrefixPath",
        "-DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH",
        "-DPYTHON_EXECUTABLE=$PythonExe",
        "-DEXECUTORCH_ENABLE_LOGGING=ON",
        "-DBUILD_TESTING=OFF",
        "-B$LlamaBuild"
    )
    Write-Host "`n=== cmake configure (examples/models/llama arm64-windows) ===" -ForegroundColor Cyan
    & cmake @LlamaArgs
    if ($LASTEXITCODE -ne 0) { throw "cmake configure (llama) failed" }
    & cmake --build $LlamaBuild --config $BuildType -j $JobNumber
    if ($LASTEXITCODE -ne 0) { throw "cmake build (llama) failed" }
}

# ---------------------------------------------------------------------------
# Block 2: Hexagon DSP direct-mode skel library  (build-hexagon/)
#
# Identical in purpose to the Linux script's Hexagon block.  Builds the
# DSP-side skel library that runs directly on the Hexagon processor.
# Requires HEXAGON_SDK_ROOT, HEXAGON_TOOLS_ROOT, DSP_VERSION in the
# environment (validated above).
# ---------------------------------------------------------------------------
if ($EnableHexagon) {
    $BuildRoot = Join-Path $PrjRoot $CmakeHexagon
    Prepare-BuildDir $BuildRoot

    $ConfigArgs = @(
        $PrjRoot,
        "-DCMAKE_TOOLCHAIN_FILE=$env:HEXAGON_SDK_ROOT\build\cmake\hexagon_toolchain.cmake",
        "-DCMAKE_BUILD_TYPE=$BuildType",
        "-DCMAKE_INSTALL_PREFIX=$BuildRoot",
        "-DPYTHON_EXECUTABLE=$PythonExe",
        "-DQNN_SDK_ROOT=$env:QNN_SDK_ROOT",
        "-DHEXAGON_SDK_ROOT=$env:HEXAGON_SDK_ROOT",
        "-DHEXAGON_TOOLS_ROOT=$env:HEXAGON_TOOLS_ROOT",
        "-DDSP_VERSION=$env:DSP_VERSION",
        "-DDSP_TYPE=$DspType",
        "-DEXECUTORCH_BUILD_QNN=ON",
        "-DEXECUTORCH_BUILD_XNNPACK=OFF",
        "-DEXECUTORCH_BUILD_DEVTOOLS=ON",
        "-DEXECUTORCH_BUILD_EXTENSION_MODULE=ON",
        "-DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON",
        "-DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON",
        "-DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON",
        "-DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON",
        "-DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON",
        "-DEXECUTORCH_BUILD_PTHREADPOOL=OFF",
        "-DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF",
        "-DEXECUTORCH_ENABLE_EVENT_TRACER=ON",
        "-DEXECUTORCH_ENABLE_LOGGING=ON",
        "-DFLATCC_ALLOW_WERROR=OFF",
        "-B$BuildRoot"
    )
    Run-CMake -ConfigArgs $ConfigArgs -BuildDir $BuildRoot
}

# ---------------------------------------------------------------------------
# Block 3: x86-64 Windows host build  (build-x86_64-windows/)
#
# Builds the AOT host libraries and the PyQnnManagerAdaptor pybind module.
# This block always runs last (same as the Linux script) because its
# post-build file copies make the Python AOT environment functional.
#
# Differences from the Linux x86_64 block:
#   - No -DANDROID_ABI / -DANDROID_PLATFORM on the Llama example cmake
#     (the Linux script has a copy-paste bug there; we omit those flags).
#   - File copies use PowerShell Copy-Item instead of cp.
#   - The pybind .pyd is named PyQnnManagerAdaptor*.pyd on Windows.
# ---------------------------------------------------------------------------
if (-not $SkipX86Windows) {
    $BuildRoot = Join-Path $PrjRoot $CmakeX86

    # Activate the VS environment for AMD64 MSVC toolchain
    $VsGenerator = Get-InstalledVsGenerator
    & $VsGenerator.DevShell -Arch amd64
    Assert-GeneratorSupported $VsGenerator.Generator

    Prepare-BuildDir $BuildRoot

    $ConfigArgs = @(
        $PrjRoot,
        "-G", $VsGenerator.Generator,
        "-A", "x64",
        "-DCMAKE_SYSTEM_NAME=Windows",
        "-DCMAKE_SYSTEM_PROCESSOR=AMD64",
        "-DCMAKE_BUILD_TYPE=$BuildType",
        "-DCMAKE_INSTALL_PREFIX=$BuildRoot",
        "-DPYTHON_EXECUTABLE=$PythonExe",
        "-DQNN_SDK_ROOT=$env:QNN_SDK_ROOT",
        "-DEXECUTORCH_BUILD_QNN=ON",
        "-DEXECUTORCH_BUILD_DEVTOOLS=ON",
        "-DEXECUTORCH_BUILD_EXTENSION_LLM=ON",
        "-DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=ON",
        "-DEXECUTORCH_BUILD_EXTENSION_MODULE=ON",
        "-DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON",
        "-DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON",
        "-DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON",
        "-DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON",
        "-DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON",
        "-DEXECUTORCH_ENABLE_EVENT_TRACER=ON",
        "-DEXECUTORCH_ENABLE_LOGGING=ON",
        "-B$BuildRoot"
    )
    Run-CMake -ConfigArgs $ConfigArgs -BuildDir $BuildRoot

    # --- Post-build: copy pybind module into the Python-importable location ---
    # On Windows the pybind module has a Python ABI tag in its name, e.g.
    # PyQnnManagerAdaptor.cp310-win_amd64.pyd.  We copy all matching files.
    $PyDst = Join-Path $PrjRoot 'backends\qualcomm\python'
    Write-Host "`nCopying pybind module to $PyDst ..." -ForegroundColor Cyan
    # Remove stale pybind module files first; preserve other files such as .gitignore.
    Get-ChildItem $PyDst -ErrorAction SilentlyContinue |
        Where-Object { $_.Extension -in '.pyd', '.dll' } |
        Remove-Item -Force

    # The multi-config generator (Visual Studio / Ninja Multi-Config) places
    # outputs under <build>/<BuildType>/.  Single-config generators place them
    # directly under <build>/.  Try both locations.
    $PySrcs = @(
        Get-ChildItem (Join-Path $BuildRoot "backends\qualcomm\$BuildType\PyQnnManagerAdaptor*") -ErrorAction SilentlyContinue
        Get-ChildItem (Join-Path $BuildRoot "backends\qualcomm\PyQnnManagerAdaptor*")            -ErrorAction SilentlyContinue
    ) | Where-Object { $_.Extension -in '.pyd', '.dll' } | Select-Object -Unique

    if (-not $PySrcs) {
        throw "Could not find PyQnnManagerAdaptor.pyd under $BuildRoot\backends\qualcomm\"
    }
    foreach ($f in $PySrcs) {
        Write-Host "`nCopying $($f.FullName) -> $PyDst"
        Copy-Item $f.FullName $PyDst -Force
    }

    # --- Post-build: copy FlatBuffers schemas for the AOT serialization pipeline ---
    Write-Host "`nCopying FlatBuffers schemas ..." -ForegroundColor Cyan
    Copy-Item (Join-Path $PrjRoot 'schema\program.fbs')     (Join-Path $PrjRoot 'exir\_serialize\program.fbs')     -Force
    Copy-Item (Join-Path $PrjRoot 'schema\scalar_type.fbs') (Join-Path $PrjRoot 'exir\_serialize\scalar_type.fbs') -Force

    # --- Post-build: initialise tokenizers submodule (needed for LLM runner) ---
    # Note: extension/llm/tokenizers is intentionally skipped on Windows in the
    # install pipeline (see CLAUDE.md), but the submodule init is still useful
    # if the user wants to build the runner manually later.
    Write-Host "`nInitialising tokenizers submodule ..." -ForegroundColor Cyan
    Push-Location (Join-Path $PrjRoot 'extension\llm\tokenizers')
    & git submodule update --init
    if ($LASTEXITCODE -ne 0) { Write-Warning "git submodule update --init failed (non-fatal)" }
    Pop-Location

    # --- Build QNN example runners for x86-64 Windows ---
    $ExampleRoot  = Join-Path $PrjRoot 'examples\qualcomm'
    $ExampleBuild = Join-Path $BuildRoot 'examples\qualcomm'
    $CmakePrefixPath = "$BuildRoot;$BuildRoot\third-party\gflags"

    $ExampleArgs = @(
        $ExampleRoot,
        "-G", $VsGenerator.Generator,
        "-A", "x64",
        "-DCMAKE_SYSTEM_NAME=Windows",
        "-DCMAKE_SYSTEM_PROCESSOR=AMD64",
        "-DCMAKE_BUILD_TYPE=$BuildType",
        "-DCMAKE_PREFIX_PATH=$CmakePrefixPath",
        "-DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH",
        "-DPYTHON_EXECUTABLE=$PythonExe",
        "-DEXECUTORCH_ENABLE_LOGGING=ON",
        "-DSUPPORT_REGEX_LOOKAHEAD=ON",
        "-DBUILD_TESTING=OFF",
        "-B$ExampleBuild"
    )
    Write-Host "`n=== cmake configure (examples/qualcomm x86-64 Windows) ===" -ForegroundColor Cyan
    & cmake @ExampleArgs
    if ($LASTEXITCODE -ne 0) { throw "cmake configure (examples/qualcomm) failed" }
    & cmake --build $ExampleBuild --config $BuildType -j $JobNumber
    if ($LASTEXITCODE -ne 0) { throw "cmake build (examples/qualcomm) failed" }

    # --- Build Llama runner for x86-64 Windows ---
    # Note: the Linux script incorrectly passes -DANDROID_ABI and
    # -DANDROID_PLATFORM here (copy-paste from the Android block).  Those
    # flags are omitted here since no Android toolchain is active.
    $LlamaRoot  = Join-Path $PrjRoot 'examples\models\llama'
    $LlamaBuild = Join-Path $BuildRoot 'examples\models\llama'

    $LlamaArgs = @(
        $LlamaRoot,
        "-G", $VsGenerator.Generator,
        "-A", "x64",
        "-DCMAKE_SYSTEM_NAME=Windows",
        "-DCMAKE_SYSTEM_PROCESSOR=AMD64",
        "-DCMAKE_BUILD_TYPE=$BuildType",
        "-DCMAKE_PREFIX_PATH=$CmakePrefixPath",
        "-DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH",
        "-DPYTHON_EXECUTABLE=$PythonExe",
        "-DEXECUTORCH_ENABLE_LOGGING=ON",
        "-DBUILD_TESTING=OFF",
        "-B$LlamaBuild"
    )
    Write-Host "`n=== cmake configure (examples/models/llama x86-64 Windows) ===" -ForegroundColor Cyan
    & cmake @LlamaArgs
    if ($LASTEXITCODE -ne 0) { throw "cmake configure (llama) failed" }
    & cmake --build $LlamaBuild --config $BuildType -j $JobNumber
    if ($LASTEXITCODE -ne 0) { throw "cmake build (llama) failed" }
}

Write-Host "`nBuild complete." -ForegroundColor Green
