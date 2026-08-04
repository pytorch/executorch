# Copyright 2026 The ExecuTorch Authors.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# rv64 baremetal cross-toolchain (Ubuntu 26.04+ packages:
# gcc-riscv64-unknown-elf, picolibc-riscv64-unknown-elf,
# libstdc++-riscv64-unknown-elf-picolibc). The resulting ELF runs under
# qemu-system-riscv64 -machine virt with semihosting.

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

set(CMAKE_C_COMPILER
    "riscv64-unknown-elf-gcc"
    CACHE FILEPATH ""
)
set(CMAKE_CXX_COMPILER
    "riscv64-unknown-elf-g++"
    CACHE FILEPATH ""
)
set(CMAKE_ASM_COMPILER
    "riscv64-unknown-elf-gcc"
    CACHE FILEPATH ""
)
set(CMAKE_AR
    "riscv64-unknown-elf-ar"
    CACHE FILEPATH ""
)
set(CMAKE_RANLIB
    "riscv64-unknown-elf-ranlib"
    CACHE FILEPATH ""
)
set(CMAKE_STRIP
    "riscv64-unknown-elf-strip"
    CACHE FILEPATH ""
)

set(CMAKE_EXECUTABLE_SUFFIX ".elf")
# try_compile() can't link without crt0/specs; archive-only sidesteps that.
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# Picked baseline: rv64iafd / lp64d. Ubuntu's picolibc + libstdc++ packages
# don't ship the rv64gc (= rv64imafdc) multilib, so this drops M (integer mul)
# and C (compressed) but keeps double-float. -mcmodel=medany because medlow's
# signed-32-bit-around-0 reach can't address our 0x80000000 base.
# --specs=picolibc.specs has to appear at *compile* time too: libstdc++'s
# <cstring>/<cassert>/<sys/types.h> need picolibc's C headers via the spec's
# sysroot.
add_compile_options(
  --specs=picolibc.specs
  -march=rv64iafd
  -mabi=lp64d
  -mcmodel=medany
  -fdata-sections
  -ffunction-sections
  "$<$<COMPILE_LANGUAGE:CXX>:-fno-rtti;-fno-exceptions;-fno-unwind-tables>"
)
# -nostdlib++ drops g++'s implicit libstdc++.a (medlow-built, won't relocate at
# 0x80000000); we only use its templates, no runtime calls. -nostartfiles drops
# picolibc's crt0 in favour of our start.S.
add_link_options(
  --specs=picolibc.specs
  -march=rv64iafd
  -mabi=lp64d
  -mcmodel=medany
  -nostdlib++
  -nostartfiles
  "LINKER:--gc-sections"
)
