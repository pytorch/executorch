# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set(TARGET_CPU
    "cortex-m55"
    CACHE STRING "Target CPU"
)
string(TOLOWER "${TARGET_CPU}" CMAKE_SYSTEM_PROCESSOR)

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

if(NOT DEFINED ENV{ATFE_TOOLCHAIN_ROOT})
  message(
    FATAL_ERROR
      "ATFE_TOOLCHAIN_ROOT is not set. Run examples/arm/setup.sh --target-toolchain clang and source examples/arm/arm-scratch/setup_path.sh."
  )
endif()
file(TO_CMAKE_PATH "$ENV{ATFE_TOOLCHAIN_ROOT}" ATFE_TOOLCHAIN_ROOT)
set(_atfe_bin "${ATFE_TOOLCHAIN_ROOT}/bin")

set(CMAKE_C_COMPILER "${_atfe_bin}/clang")
set(CMAKE_CXX_COMPILER "${_atfe_bin}/clang++")
set(CMAKE_ASM_COMPILER "${_atfe_bin}/clang")
set(CMAKE_C_COMPILER_TARGET arm-none-eabi)
set(CMAKE_CXX_COMPILER_TARGET arm-none-eabi)
set(CMAKE_ASM_COMPILER_TARGET arm-none-eabi)
set(CMAKE_AR "${_atfe_bin}/llvm-ar")
set(CMAKE_RANLIB "${_atfe_bin}/llvm-ranlib")
set(CMAKE_LINKER "${_atfe_bin}/ld.lld")
set(CMAKE_OBJCOPY "${_atfe_bin}/llvm-objcopy")
set(CMAKE_SIZE "${_atfe_bin}/llvm-size")
set(CMAKE_STRIP "${_atfe_bin}/llvm-strip")

set(CMAKE_EXECUTABLE_SUFFIX ".elf")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

add_compile_options(
  -mcpu=${CMAKE_SYSTEM_PROCESSOR}
  -mthumb
  "$<$<CONFIG:DEBUG>:-gdwarf-3>"
  "$<$<COMPILE_LANGUAGE:CXX>:-fno-unwind-tables;-fno-rtti;-fno-exceptions;-include;${CMAKE_CURRENT_LIST_DIR}/arm-none-eabi-clang-random-device.h>"
  -fdata-sections
  -ffunction-sections
  -ftls-model=local-exec
  -Wno-error=deprecated-declarations
  -Wno-error=shift-count-overflow
  -Wno-error=unused-template
  -Wno-ignored-attributes
  -Wno-psabi
)
if(EXECUTORCH_OPTIMIZE_SIZE)
  # TODO: Remove this override after the LLVM register coalescer crash is fixed:
  # https://github.com/llvm/llvm-project/issues/229400
  add_compile_options("$<$<AND:$<CONFIG:RELEASE>,$<COMPILE_LANGUAGE:CXX>>:-Oz>")
endif()
add_compile_definitions("$<$<NOT:$<CONFIG:DEBUG>>:NDEBUG>")
add_link_options(
  -mcpu=${CMAKE_SYSTEM_PROCESSOR} -mthumb -fno-exceptions -fno-rtti
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "\\+fp")
  set(FLOAT hard)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "\\+nofp")
  set(FLOAT soft)
elseif(
  CMAKE_SYSTEM_PROCESSOR MATCHES "cortex-m33(\\+|$)"
  OR CMAKE_SYSTEM_PROCESSOR MATCHES "cortex-m55(\\+|$)"
  OR CMAKE_SYSTEM_PROCESSOR MATCHES "cortex-m85(\\+|$)"
)
  set(FLOAT hard)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "cortex-m4(\\+|$)"
       OR CMAKE_SYSTEM_PROCESSOR MATCHES "cortex-m7(\\+|$)"
)
  set(FLOAT hard)
  set(FPU_CONFIG "fpv4-sp-d16")
  add_compile_options(-mfpu=${FPU_CONFIG})
  add_link_options(-mfpu=${FPU_CONFIG})
else()
  set(FLOAT soft)
endif()

if(FLOAT)
  add_compile_options(-mfloat-abi=${FLOAT})
  add_link_options(-mfloat-abi=${FLOAT})
endif()

if(SEMIHOSTING)
  add_link_options(-nostartfiles LINKER:-l:crt0-semihost.o)
  set(CMAKE_C_STANDARD_LIBRARIES "${CMAKE_C_STANDARD_LIBRARIES} -lsemihost")
  set(CMAKE_CXX_STANDARD_LIBRARIES "${CMAKE_CXX_STANDARD_LIBRARIES} -lsemihost")
elseif(NOT EXECUTORCH_SIZE_TEST_NO_OS_LINK)
  set(CMAKE_C_STANDARD_LIBRARIES "${CMAKE_C_STANDARD_LIBRARIES} -ldummyhost")
  set(CMAKE_CXX_STANDARD_LIBRARIES
      "${CMAKE_CXX_STANDARD_LIBRARIES} -ldummyhost"
  )
endif()

add_link_options(LINKER:--nmagic,--gc-sections)

if(NOT DEFINED EXECUTORCH_PAL_DEFAULT)
  set(EXECUTORCH_PAL_DEFAULT
      minimal
      CACHE STRING "Bare-metal PAL implementation"
  )
endif()
if(NOT DEFINED EXECUTORCH_PAL_DEFAULT_FILE_PATH)
  get_filename_component(
    _executorch_root "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE
  )
  set(EXECUTORCH_PAL_DEFAULT_FILE_PATH
      "${_executorch_root}/runtime/platform/default/minimal.cpp"
      CACHE FILEPATH "Bare-metal PAL implementation file"
  )
endif()

unset(_atfe_bin)
unset(_executorch_root)
