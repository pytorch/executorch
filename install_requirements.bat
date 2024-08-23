@ECHO OFF

rem Copyright (c) Meta Platforms, Inc. and affiliates.
rem All rights reserved.

rem This batch file provides a basic functionality similar to the bash script.

cd /d "%~dp0"

rem Find the names of the python tools to use (replace with your actual python installation)
if "%PYTHON_EXECUTABLE%"=="" (
  if "%CONDA_DEFAULT_ENV%"=="" OR "%CONDA_DEFAULT_ENV%"=="base" OR NOT EXIST "python" (
    set PYTHON_EXECUTABLE=python3
  ) else (
    set PYTHON_EXECUTABLE=python
  )
)

"%PYTHON_EXECUTABLE%" install_requirements.py %*

exit /b %ERRORLEVEL%