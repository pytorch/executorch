REM This is lightly modified from the torchvision Windows build logic.
REM See https://github.com/pytorch/vision/blob/main/packaging/windows/internal/vc_env_helper.bat

@echo on

set VC_VERSION_LOWER=17
set VC_VERSION_UPPER=18

for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -legacy -products * -version [%VC_VERSION_LOWER%^,%VC_VERSION_UPPER%^) -property installationPath`) do (
    if exist "%%i" if exist "%%i\VC\Auxiliary\Build\vcvarsall.bat" (
        set "VS15INSTALLDIR=%%i"
        set "VS15VCVARSALL=%%i\VC\Auxiliary\Build\vcvarsall.bat"
        goto vswhere
    )
)

:vswhere
if "%VSDEVCMD_ARGS%" == "" (
    call "%VS15VCVARSALL%" x64 || exit /b 1
) else (
    call "%VS15VCVARSALL%" x64 %VSDEVCMD_ARGS% || exit /b 1
)

@echo on

if "%CU_VERSION%" == "xpu" call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"

set DISTUTILS_USE_SDK=1

set args=%1
shift
:start
if [%1] == [] goto done
set args=%args% %1
shift
goto start

:done
if "%args%" == "" (
    echo Usage: vc_env_helper.bat [command] [args]
    echo e.g. vc_env_helper.bat cl /c test.cpp
)

echo "Evaluating symlink status. CWD: %CD%"
set work_dir=%CD%
if exist setup.py (
    echo "Creating symlink..."
    REM Setup a symlink to shorten the path length.
    REM Note that the ET directory has to be named "executorch".
    cd %GITHUB_WORKSPACE%
    if not exist et\ (
        mkdir et
    )
    cd et
    if not exist executorch\ (
        mklink /d executorch !work_dir!
    )
    cd executorch
)
echo "Post symlink CWD: %CD%"

%args% || exit /b 1
