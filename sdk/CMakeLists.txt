# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Please this file formatted by running:
# ~~~
# cmake-format --first-comment-is-literal=True CMakeLists.txt
# ~~~

cmake_minimum_required(VERSION 3.19)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(NOT PYTHON_EXECUTABLE)
  set(PYTHON_EXECUTABLE python3)
endif()

if(NOT FLATCC_EXECUTABLE)
  set(FLATCC_EXECUTABLE flatcc)
endif()

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
endif()

# Paths to headers generated from the .fbs files. set(_etdump_schemas
# etdump_schema_flatcc.fbs scalar_type.fbs)

set(_etdump_schema__srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/etdump/etdump_schema_flatcc.fbs
    ${CMAKE_CURRENT_SOURCE_DIR}/etdump/scalar_type.fbs)

set(_bundled_program_schema__srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/bundled_program/schema/bundled_program_schema.fbs
    ${CMAKE_CURRENT_SOURCE_DIR}/bundled_program/schema/scalar_type.fbs)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../third-party/flatcc
                 ${CMAKE_BINARY_DIR}/third-party/flatcc)

# Assume we are cross-compiling and the CMAKE_TOOLCHAIN_FILE is set
include(ExternalProject)

# The include directory that will contain the generated schema headers.
set(_program_schema__include_dir "${CMAKE_BINARY_DIR}/sdk/include")

# Add the host project
# lint_cmake: -readability/wonkycase
ExternalProject_Add(
  flatcc_project
  PREFIX ${CMAKE_BINARY_DIR}/_host_build
  SOURCE_DIR ${CMAKE_SOURCE_DIR}/third-party/flatcc
  BINARY_DIR ${CMAKE_BINARY_DIR}/_host_build
  INSTALL_COMMAND "" # Prevent the install step, modify as needed
)

set(_etdump_schema__outputs)
foreach(fbs_file ${_etdump_schema__srcs})
  string(REGEX REPLACE "[.]fbs$" "_generated.h" generated "${fbs_file}")
  list(APPEND _etdump_schema__outputs
       "${_program_schema__include_dir}/${generated}")
endforeach()

set(_bundled_program_schema__outputs)
foreach(fbs_file ${_bundled_program_schema__srcs})
  string(REGEX REPLACE "[.]fbs$" "_generated.h" generated "${fbs_file}")
  list(APPEND _bundled_program_schema__outputs
       "${_program_schema__include_dir}/${generated}")
endforeach()

add_library(etdump_schema INTERFACE ${_etdump_schema__outputs})
add_library(bundled_program_schema INTERFACE ${_bundled_program_schema__outputs})

# Ensure the host tool is built before the main project
add_dependencies(etdump_schema flatcc_cli)

message("Debug _etdump_schema__outputs: ${_etdump_schema__outputs}")
message("Debug _bundled_program_schema__outputs: ${_bundled_program_schema__outputs}")

file(MAKE_DIRECTORY ${_program_schema__include_dir}/executorch/sdk/etdump)
file(MAKE_DIRECTORY ${_program_schema__include_dir}/executorch/sdk/bundled_program)

add_custom_command(
  OUTPUT ${_etdump_schema__outputs}
  COMMAND
    ${CMAKE_SOURCE_DIR}/third-party/flatcc/bin/flatcc -cwr -o
    ${_program_schema__include_dir}/executorch/sdk/etdump
    ${_etdump_schema__srcs}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/sdk
  DEPENDS flatcc_project
  COMMENT "Generating etdump headers"
  VERBATIM)

add_custom_command(
  OUTPUT ${_bundled_program_schema__outputs}
  COMMAND
    ${CMAKE_SOURCE_DIR}/third-party/flatcc/bin/flatcc -cwr -o
    ${_program_schema__include_dir}/executorch/sdk/bundled_program
    ${_bundled_program_schema__srcs}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/sdk
  DEPENDS flatcc_project
  COMMENT "Generating bundled_program headers"
  VERBATIM)

target_include_directories(
  etdump PUBLIC ${_program_schema__include_dir}
                ${CMAKE_SOURCE_DIR}/third-party/flatcc/include)

set_target_properties(bundled_program PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(
  bundled_program
  INTERFACE ${_program_schema__include_dir}
            ${EXECUTORCH_ROOT}/third-party/flatbuffers/include)
