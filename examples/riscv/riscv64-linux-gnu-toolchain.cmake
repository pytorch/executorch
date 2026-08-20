# Copyright 2026 The ExecuTorch Authors.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# CMake toolchain file for cross-compiling to riscv64 Linux glibc using the
# Ubuntu / Debian gcc-riscv64-linux-gnu and g++-riscv64-linux-gnu packages.
# Resulting binaries can be executed under qemu-user-static (qemu-riscv64) or
# directly on a riscv64 Linux host.

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

set(CMAKE_C_COMPILER
    "riscv64-linux-gnu-gcc"
    CACHE FILEPATH "RISC-V cross C compiler"
)
set(CMAKE_CXX_COMPILER
    "riscv64-linux-gnu-g++"
    CACHE FILEPATH "RISC-V cross C++ compiler"
)
set(CMAKE_AR
    "riscv64-linux-gnu-ar"
    CACHE FILEPATH "RISC-V archiver"
)
set(CMAKE_RANLIB
    "riscv64-linux-gnu-ranlib"
    CACHE FILEPATH "RISC-V ranlib"
)
set(CMAKE_STRIP
    "riscv64-linux-gnu-strip"
    CACHE FILEPATH "RISC-V strip"
)

# Sysroot installed by the apt package gcc-riscv64-linux-gnu.
set(CMAKE_FIND_ROOT_PATH "/usr/riscv64-linux-gnu")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
