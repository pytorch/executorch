param (
    [string]$buildMode = "Release",
    [bool]$runPythonTests = $true,
    [bool]$runNativeTests = $true
)

Set-PSDebug -Trace 1
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

if ($runNativeTests) {
# Run native unit tests (via ctest)
    New-Item -Path "test-build" -ItemType Directory
    cd "test-build"

    cmake .. --preset windows -B . -DEXECUTORCH_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=$buildMode
    if ($LASTEXITCODE -ne 0) {
        Write-Host "CMake configuration was unsuccessful. Exit code: $LASTEXITCODE."
        exit $LASTEXITCODE
    }

    cmake --build . -j8 --config $buildMode --verbose
    if ($LASTEXITCODE -ne 0) {
        Write-Host "CMake build was unsuccessful. Exit code: $LASTEXITCODE."
        exit $LASTEXITCODE
    }

    ctest -j8 . --build-config $buildMode --output-on-failure -E "method_test|tensor_parser_test"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "CTest run was unsuccessful. Exit code: $LASTEXITCODE."
        exit $LASTEXITCODE
    }

    cd ..
}

if ($runPythonTests) {
# Run pytest
    pytest -v -c pytest-windows.ini
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Pytest invocation was unsuccessful. Exit code: $LASTEXITCODE."
        exit $LASTEXITCODE
    }
}
