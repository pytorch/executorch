@ECHO OFF
setlocal EnableDelayedExpansion

rem Copyright (c) Meta Platforms, Inc. and affiliates.
rem All rights reserved.

rem This batch file provides a basic functionality similar to the bash script.

cd /d "%~dp0"

rem Verify that Git checked out symlinks correctly. Without this the Python install
rem will fail when attempting to copy files from src\executorch.
where git >NUL 2>&1
if not errorlevel 1 (
    set "GIT_SYMLINKS="
    for /f "usebackq delims=" %%i in (`git config --get core.symlinks 2^>nul`) do set "GIT_SYMLINKS=%%i"
    if /I not "!GIT_SYMLINKS!"=="true" (
        echo ExecuTorch requires Git symlink support on Windows.
        echo Enable Developer Mode and run: git config --global core.symlinks true
        echo Re-clone the repository after enabling symlinks, then rerun install_executorch.bat.
        exit /b 1
    )
)

rem Under windows, it's always python
set PYTHON_EXECUTABLE=python

"%PYTHON_EXECUTABLE%" install_executorch.py %*

set "EXIT_CODE=%ERRORLEVEL%"
endlocal & exit /b %EXIT_CODE%
