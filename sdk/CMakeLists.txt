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

if(NOT FLATCC_EXECUTABLE)
  set(FLATCC_EXECUTABLE flatcc)
endif()

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
endif()

include(${EXECUTORCH_ROOT}/build/Utils.cmake)

if(NOT PYTHON_EXECUTABLE)
  resolve_python_executable()
endif()

if(NOT FLATC_EXECUTABLE)
  set(FLATC_EXECUTABLE flatc)
endif()

# Paths to headers generated from the .fbs files. set(_etdump_schemas
# etdump_schema_flatcc.fbs scalar_type.fbs)

set(_etdump_schema_names "etdump_schema_flatcc.fbs" "scalar_type.fbs")
set(_bundled_input_schema_names "bundled_program_schema.fbs" "scalar_type.fbs")

foreach(schema_file ${_etdump_schema_names})
  list(APPEND _etdump_schema__srcs
       "${CMAKE_CURRENT_SOURCE_DIR}/etdump/${schema_file}")
endforeach()

foreach(schema_file ${_bundled_input_schema_names})
  list(APPEND _bundled_program_schema__srcs
       "${CMAKE_CURRENT_SOURCE_DIR}/bundled_program/schema/${schema_file}")
endforeach()

set(FLATCC_TEST OFF CACHE BOOL "")
set(FLATCC_REFLECTION OFF CACHE BOOL "")
set(_flatcc_source_dir ${CMAKE_CURRENT_SOURCE_DIR}/../third-party/flatcc)
add_subdirectory(${_flatcc_source_dir} ${CMAKE_BINARY_DIR}/third-party/flatcc)

# Fix for "relocation R_X86_64_32 against `.rodata' can not be used when making
# a shared object; recompile with -fPIC" when building on some x86 linux
# systems.
set_property(TARGET flatccrt PROPERTY POSITION_INDEPENDENT_CODE ON)

# Assume we are cross-compiling and the CMAKE_TOOLCHAIN_FILE is set
include(ExternalProject)

# The include directory that will contain the generated schema headers.
set(_program_schema__include_dir "${CMAKE_BINARY_DIR}/sdk/include")
set(_bundled_schema__include_dir "${CMAKE_BINARY_DIR}/sdk/bundled_program")

# Add the host project
# lint_cmake: -readability/wonkycase
ExternalProject_Add(
  flatcc_project
  PREFIX ${CMAKE_BINARY_DIR}/_host_build
  SOURCE_DIR ${_flatcc_source_dir}
  BINARY_DIR ${CMAKE_BINARY_DIR}/_host_build
  CMAKE_CACHE_ARGS -DFLATCC_TEST:BOOL=OFF -DFLATCC_REFLECTION:BOOL=OFF
      # See above comment about POSITION_INDEPENDENT_CODE.
      -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
  INSTALL_COMMAND "" # Prevent the install step, modify as needed
)

set(_etdump_schema__outputs)
foreach(fbs_file ${_etdump_schema_names})
  string(REGEX REPLACE "[.]fbs$" "_reader.h" generated "${fbs_file}")
  list(APPEND _etdump_schema__outputs
       "${_program_schema__include_dir}/executorch/sdk/etdump/${generated}")
  string(REGEX REPLACE "[.]fbs$" "_builder.h" generated "${fbs_file}")
  list(APPEND _etdump_schema__outputs
       "${_program_schema__include_dir}/executorch/sdk/etdump/${generated}")
endforeach()

# lint_cmake: -linelength
set(_bundled_program_schema__outputs)
foreach(fbs_file ${_bundled_input_schema_names})
  string(REGEX REPLACE "[.]fbs$" "_generated.h" generated "${fbs_file}")
  list(
    APPEND
    _bundled_program_schema__outputs
    "${_bundled_schema__include_dir}/executorch/sdk/bundled_program/schema/${generated}"
  )
endforeach()

add_library(etdump_schema INTERFACE ${_etdump_schema__outputs})
add_library(bundled_program_schema INTERFACE
            ${_bundled_program_schema__outputs})

# Ensure the host tool is built before the main project
add_dependencies(etdump_schema flatcc_cli)

file(MAKE_DIRECTORY ${_program_schema__include_dir}/executorch/sdk/etdump)
file(MAKE_DIRECTORY
     ${_program_schema__include_dir}/executorch/sdk/bundled_program)

add_custom_command(
  OUTPUT ${_etdump_schema__outputs}
  COMMAND
    # Note that the flatcc project actually writes its outputs into the source
    # tree instead of under the binary directory, and there's no way to change
    # that behavior.
    ${_flatcc_source_dir}/bin/flatcc -cwr -o
    ${_program_schema__include_dir}/executorch/sdk/etdump
    ${_etdump_schema__srcs}
  # TODO(dbort): flatcc installs its files directly in its source directory
  # instead of under CMAKE_BINARY_DIR, and it has no options to avoid
  # doing this. We build flatcc twice in the executorch build: once to get
  # the `flatcc` host commandline tool, and once to get the (potentially
  # cross-compiled) target runtime library. The host build will put its outputs
  # in the source tree, making the cross-compiling target build think that
  # the outputs have already been built. It will then try to link against the
  # host-architecture libraries, failing when cross-compiling. To work around
  # this, delete the host outputs after running this command (which only runs
  # when setting up the cmake files, not when actually building). This leaves
  # room for the target build to put its own files in the source tree. We should
  # try to remove this hack, ideally by submitting an upstream PR that adds an
  # option to change the installation location.
  COMMAND
    rm -f ${_flatcc_source_dir}/bin/*
    ${_flatcc_source_dir}/lib/*
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/sdk
  DEPENDS flatcc_project
  COMMENT "Generating etdump headers"
  VERBATIM)

add_library(etdump ${CMAKE_CURRENT_SOURCE_DIR}/etdump/etdump_flatcc.cpp
                   ${CMAKE_CURRENT_SOURCE_DIR}/etdump/emitter.cpp)

target_link_libraries(
  etdump
  PUBLIC etdump_schema flatccrt
  PRIVATE executorch)

add_custom_command(
  OUTPUT ${_bundled_program_schema__outputs}
  COMMAND
    ${FLATC_EXECUTABLE} --cpp --cpp-std c++11 --gen-mutable --scoped-enums -o
    "${_bundled_schema__include_dir}/executorch/sdk/bundled_program/schema"
    ${_bundled_program_schema__srcs}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/sdk
  DEPENDS ${FLATC_EXECUTABLE} ${_bundled_program_schema__srcs}
  COMMENT "Generating bundled_program headers"
  VERBATIM)

# add_library(bundled_program INTERFACE ${_bundled_program_schema__outputs})
add_library(bundled_program
            ${CMAKE_CURRENT_SOURCE_DIR}/bundled_program/bundled_program.cpp)
target_link_libraries(bundled_program executorch bundled_program_schema)

set_target_properties(bundled_program PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(
  bundled_program PUBLIC ${_bundled_schema__include_dir}
                         ${EXECUTORCH_ROOT}/third-party/flatbuffers/include)

target_include_directories(
  etdump PUBLIC ${_program_schema__include_dir}
                ${_flatcc_source_dir}/include)

# Install libraries
install(
  TARGETS bundled_program etdump flatccrt
  DESTINATION ${CMAKE_BINARY_DIR}/lib
  INCLUDES
  DESTINATION ${_common_include_directories})
