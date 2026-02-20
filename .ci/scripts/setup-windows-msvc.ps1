conda create --yes --quiet -n et python=3.12
conda activate et

# Install cmake
conda install -y cmake

# Activate the VS environment - this is required for MSVC to work
# There are a bunch of environment variables that it requires.
# See https://learn.microsoft.com/en-us/cpp/build/building-on-the-command-line.
& "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64

# Install CI requirements
pip install -r .ci/docker/requirements-ci.txt

# Create build directory
$buildDir = "cmake-out-msvc"
if (Test-Path -Path $buildDir) {
    Remove-Item -Path $buildDir -Recurse -Force
}
New-Item -Path $buildDir -ItemType Directory

# Configure CMake with MSVC (not ClangCL) and disable custom/quantized ops
cmake -S . -B $buildDir `
    -DCMAKE_BUILD_TYPE=Release `
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
    -DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=ON

if ($LASTEXITCODE -ne 0) {
    Write-Host "CMake configuration failed. Exit code: $LASTEXITCODE."
    exit $LASTEXITCODE
}

# Build with MSVC
cmake --build $buildDir --config Release -j16

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed. Exit code: $LASTEXITCODE."
    exit $LASTEXITCODE
}

Write-Host "MSVC build completed successfully!"
