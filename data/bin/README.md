## PLEASE DO NOT REMOVE THIS DIRECTORY!

This directory is used to host binaries installed during pip wheel build time.

## How to add a binary into pip wheel

1. Update `[project.scripts]` section of `pyproject.toml` file. Add the new binary name and it's corresponding module name similar to:

```
flatc = "executorch.data.bin:flatc"
```

For example, `flatc` is built during wheel packaging, we first build `flatc` through CMake and copy the file to `<executorch root>/data/bin/flatc` and ask `setuptools` to generate a commandline wrapper for `flatc`, then route it to `<executorch root>/data/bin/flatc`.

This way after installing `executorch`, a user will be able to call `flatc` directly in commandline and it points to `<executorch root>/data/bin/flatc`

2. Update `setup.py` to include the logic of building the new binary and copying the binary to this directory.

```python
BuiltFile(
    src_dir="%CMAKE_CACHE_DIR%/third-party/flatbuffers/%BUILD_TYPE%/",
    src_name="flatc",
    dst="executorch/data/bin/",
    is_executable=True,
),
```
This means find `flatc` in `CMAKE_CACHE_DIR` and copy it to `<executorch root>/data/bin`. Notice that this works for both pip wheel packaging as well as editable mode install.

## Why we can't create this directory at wheel build time?

The reason is without `data/bin/` present in source file, we can't tell `setuptools` to generate a module `executorch.data.bin` in editable mode, partially because we don't have a good top level module `executorch` and have to enumerate all the second level modules, including `executorch.data.bin`.
