# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set(XTENSA_TOOLCHAIN_PATH $ENV{XTENSA_TOOLCHAIN})

if(NOT EXISTS ${XTENSA_TOOLCHAIN_PATH})
  message(
    FATAL_ERROR
      "Nothing found at XTENSA_TOOLCHAIN_PATH: '${XTENSA_TOOLCHAIN_PATH}'"
  )
endif()

set(TOOLCHAIN_HOME ${XTENSA_TOOLCHAIN_PATH}/$ENV{TOOLCHAIN_VER}/XtensaTools)

set(LINKER ld)
set(BINTOOLS gnu)

set(CROSS_COMPILE_TARGET xt)
set(SYSROOT_TARGET xtensa-elf)

set(CROSS_COMPILE ${TOOLCHAIN_HOME}/bin/${CROSS_COMPILE_TARGET}-)
set(SYSROOT_DIR ${TOOLCHAIN_HOME}/${SYSROOT_TARGET})

set(NOSYSDEF_CFLAG "")

list(APPEND TOOLCHAIN_C_FLAGS -fms-extensions)

set(TOOLCHAIN_HAS_NEWLIB
    OFF
    CACHE BOOL "True if toolchain supports newlib"
)

set(COMPILER xt-clang)
# set(CC clang) set(C++ clang++)
set(LINKER xt-ld)

set(CMAKE_CROSSCOMPILING TRUE)
set(CMAKE_C_COMPILER ${TOOLCHAIN_HOME}/bin/${CROSS_COMPILE_TARGET}-clang)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_HOME}/bin/${CROSS_COMPILE_TARGET}-clang++)

set(CMAKE_C_FLAGS_INIT "-stdlib=libc++ -mtext-section-literals -mlongcalls")
set(CMAKE_CXX_FLAGS_INIT "-stdlib=libc++ -mtext-section-literals -mlongcalls")
set(CMAKE_SYSROOT ${TOOLCHAIN_HOME}/${SYSROOT_TARGET})
set(CMAKE_LINKER ${TOOLCHAIN_HOME}/bin/xt-ld)
add_link_options(-lm -stdlib=libc++ -Wl,--no-as-needed -static)
message(STATUS "Found toolchain: xt-clang (${XTENSA_TOOLCHAIN_PATH})")
