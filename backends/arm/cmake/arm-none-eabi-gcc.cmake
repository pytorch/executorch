# Copyright 2023 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set(TARGET_CPU "cortex-m55" CACHE STRING "Target CPU")
string(TOLOWER ${TARGET_CPU} CMAKE_SYSTEM_PROCESSOR)

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_C_COMPILER "arm-none-eabi-gcc")
set(CMAKE_CXX_COMPILER "arm-none-eabi-g++")
set(CMAKE_ASM_COMPILER "arm-none-eabi-gcc")
set(CMAKE_LINKER "arm-none-eabi-ld")

set(CMAKE_EXECUTABLE_SUFFIX ".elf")
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# Select C/C++ version
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)

set(GCC_CPU ${CMAKE_SYSTEM_PROCESSOR})
string(REPLACE "cortex-m85" "cortex-m55" GCC_CPU ${GCC_CPU})

# Compile options
add_compile_options(
    -mcpu=${GCC_CPU}
    -mthumb
    "$<$<CONFIG:DEBUG>:-gdwarf-3>"
    "$<$<COMPILE_LANGUAGE:CXX>:-fno-unwind-tables;-fno-rtti;-fno-exceptions>"
    -fdata-sections
    -ffunction-sections)

# Compile defines
add_compile_definitions(
    "$<$<NOT:$<CONFIG:DEBUG>>:NDEBUG>")

# Link options
add_link_options(
    -mcpu=${GCC_CPU}
    -mthumb
    --specs=nosys.specs)

# Set floating point unit
if(CMAKE_SYSTEM_PROCESSOR MATCHES "\\+fp")
    set(FLOAT hard)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "\\+nofp")
    set(FLOAT soft)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "cortex-m33(\\+|$)" OR
       CMAKE_SYSTEM_PROCESSOR MATCHES "cortex-m55(\\+|$)" OR
       CMAKE_SYSTEM_PROCESSOR MATCHES "cortex-m85(\\+|$)")
    set(FLOAT hard)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "cortex-m4(\\+|$)" OR
       CMAKE_SYSTEM_PROCESSOR MATCHES "cortex-m7(\\+|$)")
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

add_link_options(LINKER:--nmagic,--gc-sections)

# Compilation warnings
add_compile_options(
#    -Wall
#    -Wextra

#    -Wcast-align
#    -Wdouble-promotion
#    -Wformat
#    -Wmissing-field-initializers
#    -Wnull-dereference
#    -Wredundant-decls
#    -Wshadow
#    -Wswitch
#    -Wswitch-default
#    -Wunused
    -Wno-redundant-decls
    -Wno-psabi
)
