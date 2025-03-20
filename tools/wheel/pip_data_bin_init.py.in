# This file should be written to the wheel package as
# `executorch/data/bin/__init__.py`.
#
# Setuptools will expect to be able to say something like `from
# executorch.data.bin import mybin; mybin()` for each entry listed in the
# [project.scripts] section of pyproject.toml. This file makes the `mybin()`
# function execute the binary at `executorch/data/bin/mybin` and exit with that
# binary's exit status.

import subprocess
import os
import sys
import types

# This file should live in the target `bin` directory.
_bin_dir = os.path.join(os.path.dirname(__file__))

def _find_executable_files_under(dir):
    """Lists all executable files in the given directory."""
    bin_names = []
    for filename in os.listdir(dir):
        filepath = os.path.join(dir, filename)
        if os.path.isfile(filepath) and os.access(filepath, os.X_OK):
            # Remove .exe suffix on windows.
            filename_without_ext = os.path.splitext(filename)[0]
            bin_names.append(filename_without_ext)
    return bin_names

# The list of binaries to create wrapper functions for.
_bin_names = _find_executable_files_under(_bin_dir)

# We'll define functions named after each binary. Make them importable.
__all__ = _bin_names

def _run(name):
    """Runs the named binary, which should live under _bin_dir.

    Exits the current process with the return code of the subprocess.
    """
    raise SystemExit(subprocess.call([os.path.join(_bin_dir, name)] + sys.argv[1:], close_fds=False))

# Define a function named after each of the binaries.
for bin_name in _bin_names:
    exec(f"def {bin_name}(): _run('{bin_name}')")
