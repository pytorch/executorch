# Copyright 2026 The ExecuTorch Authors.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# rv32 baremetal cross-toolchain. Uses the multilib-aware riscv64-unknown-elf
# gcc (one package, both XLENs); `-march=rv32...` + `-mabi=ilp32d` selects the
# 32-bit picolibc + libstdc++ variant. ELF runs under qemu-system-riscv32
# -machine virt with semihosting.

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR riscv32)

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

# Baseline rv32imafdc / ilp32d — the rv32gc-equivalent multilib Ubuntu's
# picolibc + libstdc++ ship. (Unlike rv64, the full rv32gc multilib *is*
# packaged, so we don't have to drop M / C here.) -mcmodel=medany because medlow
# can't reach our 0x80000000 base. picolibc.specs must be on the compile line
# too so libstdc++ headers find picolibc's C headers via the spec's sysroot.
add_compile_options(
  --specs=picolibc.specs
  -march=rv32imafdc
  -mabi=ilp32d
  -mcmodel=medany
  -fdata-sections
  -ffunction-sections
  "$<$<COMPILE_LANGUAGE:CXX>:-fno-rtti;-fno-exceptions;-fno-unwind-tables>"
)
# -nostdlib++ drops g++'s implicit libstdc++.a (medlow-built, won't relocate).
# -nostartfiles drops picolibc's crt0 in favour of our start.S.
add_link_options(
  --specs=picolibc.specs
  -march=rv32imafdc
  -mabi=ilp32d
  -mcmodel=medany
  -nostdlib++
  -nostartfiles
  "LINKER:--gc-sections"
)
