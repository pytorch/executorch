# Copyright 2026 The ExecuTorch Authors.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# CMake toolchain file for cross-compiling to riscv64 Linux glibc using the
# Ubuntu / Debian gcc-riscv64-linux-gnu and g++-riscv64-linux-gnu packages.
# Resulting binaries can be executed under qemu-user-static (qemu-riscv64) or
# directly on a riscv64 Linux host.

if(CMAKE_VERSION VERSION_LESS 3.20)
  message(FATAL_ERROR "This toolchain file requires at least CMake 3.20")
endif()

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

set(_RISCV_TRIPLE "riscv64-linux-gnu")

set(CMAKE_C_COMPILER
    "${_RISCV_TRIPLE}-gcc"
    CACHE FILEPATH "RISC-V cross C compiler"
)
set(CMAKE_CXX_COMPILER
    "${_RISCV_TRIPLE}-g++"
    CACHE FILEPATH "RISC-V cross C++ compiler"
)
set(CMAKE_AR
    "${_RISCV_TRIPLE}-ar"
    CACHE FILEPATH "RISC-V archiver"
)
set(CMAKE_RANLIB
    "${_RISCV_TRIPLE}-ranlib"
    CACHE FILEPATH "RISC-V ranlib"
)
set(CMAKE_STRIP
    "${_RISCV_TRIPLE}-strip"
    CACHE FILEPATH "RISC-V strip"
)

# Sysroot installed by the apt package gcc-riscv64-linux-gnu.
set(CMAKE_SYSROOT "/usr/${_RISCV_TRIPLE}")
set(CMAKE_FIND_ROOT_PATH "${CMAKE_SYSROOT}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
