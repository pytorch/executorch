param (
    [string]$editable = $false
)

Set-PSDebug -Trace 1
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

# Run pytest
pytest -v -c pytest-windows.ini
if ($LASTEXITCODE -ne 0) {
    Write-Host "Pytest invocation was unsuccessful. Exit code: $LASTEXITCODE."
    exit $LASTEXITCODE
}

# Run native unit tests (via ctest)
New-Item -Path "test-build" -ItemType Directory
cd "test-build"

cmake .. --preset windows -B . -DCMAKE_BUILD_TESTS=ON
if ($LASTEXITCODE -ne 0) {
    Write-Host "CMake configuration was unsuccessful. Exit code: $LASTEXITCODE."
    exit $LASTEXITCODE
}

cmake --build . -j8 --config Release
if ($LASTEXITCODE -ne 0) {
    Write-Host "CMake build was unsuccessful. Exit code: $LASTEXITCODE."
    exit $LASTEXITCODE
}

ctest -j8
if ($LASTEXITCODE -ne 0) {
    Write-Host "CTest run was unsuccessful. Exit code: $LASTEXITCODE."
    exit $LASTEXITCODE
}
