Set-PSDebug -Trace 1
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

$vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vsInstallPath = & $vsWhere -latest -products * `
    -requires Microsoft.VisualStudio.Component.VC.Tools.ARM64 `
    -property installationPath
if (-not $vsInstallPath) {
    throw "Visual Studio with the ARM64 C++ toolchain was not found."
}
$vsDevShell = Join-Path $vsInstallPath "Common7\Tools\Launch-VsDevShell.ps1"
& $vsDevShell -Arch arm64 -HostArch amd64

$buildDir = "cmake-out-windows-arm64"
if (Test-Path -Path $buildDir) {
    Remove-Item -Path $buildDir -Recurse -Force
}

# XNNPACK's optional ARM ISA and assembly microkernels do not build with MSVC.
# Baseline ARM64 NEON kernels remain enabled.
cmake -S . -B $buildDir `
    -G "Visual Studio 17 2022" `
    -A ARM64 `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_CXX_STANDARD=20 `
    -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=ON `
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON `
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON `
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON `
    -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON `
    -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON `
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON `
    -DEXECUTORCH_BUILD_KERNELS_CUSTOM=OFF `
    -DEXECUTORCH_BUILD_KERNELS_CUSTOM_AOT=OFF `
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=OFF `
    -DEXECUTORCH_BUILD_XNNPACK=ON `
    -DEXECUTORCH_BUILD_EXTENSION_LLM=ON `
    -DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=ON `
    -DXNNPACK_ENABLE_ASSEMBLY=OFF `
    -DXNNPACK_ENABLE_ARM_BF16=OFF `
    -DXNNPACK_ENABLE_ARM_DOTPROD=OFF `
    -DXNNPACK_ENABLE_ARM_FP16_SCALAR=OFF `
    -DXNNPACK_ENABLE_ARM_FP16_VECTOR=OFF

cmake --build $buildDir --config Release -j $env:NUMBER_OF_PROCESSORS

Write-Host "Windows ARM64 build completed successfully!"
