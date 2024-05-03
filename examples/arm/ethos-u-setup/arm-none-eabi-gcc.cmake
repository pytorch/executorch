#
# Copyright (c) 2020-2022 Arm Limited. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Copied this file from core_platform/cmake/toolchain/arm-non-eabi-gcc.cmake And
# modified to align better with cs300 platform

set(TARGET_CPU
    "cortex-m55"
    CACHE STRING "Target CPU"
)
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
set(CMAKE_CXX_STANDARD 17)

set(GCC_CPU ${CMAKE_SYSTEM_PROCESSOR})
string(REPLACE "cortex-m85" "cortex-m55" GCC_CPU ${GCC_CPU})

# Compile options
add_compile_options(
  -mcpu=${GCC_CPU} -mthumb "$<$<CONFIG:DEBUG>:-gdwarf-3>"
  "$<$<COMPILE_LANGUAGE:CXX>:-fno-unwind-tables;-fno-rtti;-fno-exceptions>"
  -fdata-sections -ffunction-sections
)

# Compile defines
add_compile_definitions("$<$<NOT:$<CONFIG:DEBUG>>:NDEBUG>")

# Link options
add_link_options(-mcpu=${GCC_CPU} -mthumb)

if(SEMIHOSTING)
  add_link_options(--specs=rdimon.specs)
else()
  add_link_options(--specs=nosys.specs)
endif()

# Set floating point unit
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

add_link_options(LINKER:--nmagic,--gc-sections)

# Compilation warnings
add_compile_options(
  # -Wall -Wextra -Wcast-align -Wdouble-promotion -Wformat
  # -Wmissing-field-initializers -Wnull-dereference -Wredundant-decls -Wshadow
  # -Wswitch -Wswitch-default -Wunused -Wno-redundant-decls
  -Wno-psabi
)
