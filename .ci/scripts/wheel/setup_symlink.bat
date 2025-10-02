set work_dir=%CD%
echo "Evaluting symlink for %work_dir%"
if exist setup.py (
    REM Relocate the repo to a shorter path. Setup a symlink to preserve the original usage.
    REM Note that the ET directory has to be named "executorch".
    if not exist C:\_et\executorch\ (
        echo "Relocating executorch repo..."
        cd C:\
        if not exist _et\ (
            mkdir _et
        )
        cd _et
        move $work_dir% .
        cd executorch
        mklink /d %work_dir% C:\_et\executorch\
    ) else (
        cd C:\_et\executorch\
    )
)
