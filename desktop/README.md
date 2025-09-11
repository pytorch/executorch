# Experimental: PyTorch Unified Python-less Solution

This folder contains the experimental PyTorch Unified Python-less Solution, for both compiler and runtime. Proceed with caution.


## torch dependency
We use the pinned pytorch version from `install_requirements.py` and CI should be using `.ci/docker/ci_commit_pins/pytorch.txt` which should be consistent with `install_requirements.py`.


## Compiler
All code should live in `compiler/` folder. Code uses `torch` nightly as mentioned in torch dependency section.

## Runtime
All code should live in `runtime/` folder. CMake build system should leverage `libtorch` in the pip install of `torch` nightly. To build runtime, we need to point `CMAKE_PREFIX_PATH` to the pip install location of `torch` nightly. This way we can do:

```cmake
find_package(torch REQUIRED)
```
