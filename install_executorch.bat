@ECHO OFF

rem Copyright (c) Meta Platforms, Inc. and affiliates.
rem All rights reserved.

rem This batch file provides a basic functionality similar to the bash script.

cd /d "%~dp0"

rem Under windows it's always python
set PYTHON_EXECUTABLE=python

"%PYTHON_EXECUTABLE%" install_executorch.py %*

exit /b %ERRORLEVEL%
