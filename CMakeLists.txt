# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Simple CMake build system for runtime components.
#
# ### One-time setup ###
#
# Configure the CMake build system. It's good practice to do this whenever
# cloning or pulling the upstream repo. Once this is done, you don't need to do
# it again until you pull from the upstream repo again.
#
# NOTE: If your `buck2` binary is not on the PATH, you can change this line to
# say something like `-DBUCK2=/tmp/buck2` to point directly to the tool.
#[[
  (rm -rf cmake-out \
    && mkdir cmake-out \
    && cd cmake-out \
    && cmake -DBUCK2=buck2 ..)
]]
#
# ### Build ###
#
# NOTE: The `-j` argument specifies how many jobs/processes to use when
# building, and tends to speed up the build significantly. It's typical to use
# "core count + 1" as the `-j` value.
# ~~~
# cmake --build cmake-out -j9
# ~~~
#
# ### Editing this file ###
#
# This file should be formatted with
# ~~~
# cmake-format --first-comment-is-literal=True CMakeLists.txt
# ~~~
# It should also be cmake-lint clean.
#

cmake_minimum_required(VERSION 3.19)
project(executorch)
include(build/Utils.cmake)

# option to register custom operator `my_ops::mul3` in
# `examples/custom_ops/custom_ops_1.py`
option(REGISTER_EXAMPLE_CUSTOM_OP_1
       "Register custom operator defined in examples/custom_ops/custom_ops_1.py"
       OFF)
# option to register custom operator `my_ops::mul4` in
# `examples/custom_ops/custom_ops_2.py`
option(REGISTER_EXAMPLE_CUSTOM_OP_2
       "Register custom operator defined in examples/custom_ops/custom_ops_2.py"
       OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(NOT BUCK2)
  set(BUCK2 buck2)
endif()

# TODO(dbort): Fix these warnings and remove this flag.
set(_common_compile_options -Wno-deprecated-declarations)

if(REGISTER_EXAMPLE_CUSTOM_OP_2)
  # Need to be linked to a shared library
  list(APPEND _common_compile_options -fPIC)
endif()

# Let files say "include <executorch/path/to/header.h>".
set(_common_include_directories ${CMAKE_CURRENT_SOURCE_DIR}/..)

#
# The `_<target>_srcs` lists are defined by including ${EXECUTORCH_SRCS_FILE}.
#

if(NOT EXECUTORCH_SRCS_FILE)
  # A file wasn't provided. Run a script to extract the source lists from the
  # buck2 build system and write them to a file we can include.
  #
  # NOTE: This will only happen once during cmake setup, so it will not re-run
  # if the buck2 targets change.
  message(STATUS "executorch: Generating source lists")
  set(EXECUTORCH_SRCS_FILE "${CMAKE_CURRENT_BINARY_DIR}/executorch_srcs.cmake")
  execute_process(
    COMMAND python3 build/extract_sources.py --buck2=${BUCK2}
            --config=build/cmake_deps.toml --out=${EXECUTORCH_SRCS_FILE}
    OUTPUT_VARIABLE gen_srcs_output
    ERROR_VARIABLE gen_srcs_error
    RESULT_VARIABLE gen_srcs_exit_code
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  if(NOT gen_srcs_exit_code EQUAL 0)
    message("Error while generating ${EXECUTORCH_SRCS_FILE}. "
            "Exit code: ${gen_srcs_exit_code}")
    message("Output:\n${gen_srcs_output}")
    message("Error:\n${gen_srcs_error}")
    message(FATAL_ERROR "executorch: source list generation failed")
  endif()
endif()

# This file defines the `_<target>__srcs` variables used below.
message(STATUS "executorch: Using sources file ${EXECUTORCH_SRCS_FILE}")
include(${EXECUTORCH_SRCS_FILE})

#
# flatc: Flatbuffer commandline tool to generate .h files from .fbs files
#

option(FLATBUFFERS_BUILD_FLATC "" ON)
option(FLATBUFFERS_BUILD_FLATHASH "" OFF)
option(FLATBUFFERS_BUILD_FLATLIB "" OFF)
option(FLATBUFFERS_BUILD_TESTS "" OFF)
option(FLATBUFFERS_INSTALL "" OFF)
add_subdirectory(third-party/flatbuffers)

#
# gflags: Commandline flag libgrary
#

add_subdirectory(third-party/gflags)

#
# program_schema: Generated .h files from schema/*.fbs inputs
#

# The include directory that will contain the generated schema headers.
set(_program_schema__include_dir "${CMAKE_CURRENT_BINARY_DIR}/schema/include")

# Paths to headers generated from the .fbs files.
set(_program_schema__outputs)
foreach(fbs_file ${_program_schema__srcs})
  string(REGEX REPLACE "[.]fbs$" "_generated.h" generated "${fbs_file}")
  list(APPEND _program_schema__outputs
       "${_program_schema__include_dir}/executorch/${generated}")
endforeach()

# Generate the headers from the .fbs files.
add_custom_command(
  OUTPUT ${_program_schema__outputs}
  COMMAND
    flatc --cpp --cpp-std c++11 --gen-mutable --scoped-enums
    # Add a subdirectory to the include dir so the files can be included as
    # <executorch/schema/x_generated.h>
    -o "${_program_schema__include_dir}/executorch/schema"
    ${_program_schema__srcs}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS flatc ${_program_schema__srcs}
  COMMENT "Generating program_schema headers"
  VERBATIM)

add_library(program_schema INTERFACE ${_program_schema__outputs})
set_target_properties(program_schema PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(
  program_schema
  INTERFACE ${_program_schema__include_dir}
            ${CMAKE_CURRENT_SOURCE_DIR}/third-party/flatbuffers/include)

#
# executorch: Core runtime library
#
# Only contains primitive operators; does not contain portable kernels or other
# full operators. Does not contain any backends.
#

add_library(executorch ${_executorch__srcs})
target_link_libraries(executorch PRIVATE program_schema)
target_link_libraries(executorch PRIVATE dl) # For dladdr()
target_include_directories(executorch PUBLIC ${_common_include_directories})
target_compile_options(executorch PUBLIC ${_common_compile_options})

#
# portable_kernels: Pure-C++ kernel library for ATen ops
#
# Focused on portability and understandability rather than speed.
#

add_library(portable_kernels ${_portable_kernels__srcs})
target_link_libraries(portable_kernels PRIVATE executorch)
target_compile_options(portable_kernels PUBLIC ${_common_compile_options})

#
# portable_kernels_bindings: Bindings and registration for all ops defined in
# kernels/portable/functions.yaml
#
# Real integrations should supply their own YAML file that only lists the
# operators necessary for the models that will run.
#
# TODO(dbort): Make it possible to provide a custom YAML file. It will be easier
# once we stop using buck2 for this step.
#

set(_portable_kernels_bindings__generated_files
    # Although the codegen tool generates more files, these are the only ones we
    # need for non-custom kernels.
    NativeFunctions.h RegisterCodegenUnboxedKernelsEverything.cpp)

set(_portable_kernels_bindings__output_dir
    "${CMAKE_CURRENT_BINARY_DIR}/portable_kernels_bindings")

# Paths to files generated by the codegen step.
set(_portable_kernels_bindings__outputs)
foreach(gen ${_portable_kernels_bindings__generated_files})
  list(APPEND _portable_kernels_bindings__outputs
       "${_portable_kernels_bindings__output_dir}/${gen}")
endforeach()

set(_portable_kernels_bindings__cpp_files
    ${_portable_kernels_bindings__outputs})
list(FILTER _portable_kernels_bindings__cpp_files INCLUDE REGEX "[.]cpp$")

# Build the generated files.
#
# NOTE: This will only happen once during cmake setup, so it will not re-run if
# the functions.yaml file changes. TODO(dbort): Stop using buck2 to do this. Use
# add_custom_command() to run the codegen tool directly.
message(STATUS "portable_kernels_bindings: Generating bindings")
execute_process(
  COMMAND ${BUCK2} build //kernels/portable:generated_lib_combined --show-output
  OUTPUT_VARIABLE buck_output
  ERROR_VARIABLE buck_error
  RESULT_VARIABLE buck_exit_code
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
if(buck_exit_code EQUAL 0)
  # The output will look like
  # ~~~
  # root//kernels/portable:generated_lib_combined buck-out/<path-we-want>
  # ~~~
  # Extract the second field while avoiding trailing whitespace.
  string(REGEX MATCH "buck-out/[^ \t\r\n]*" srcdir ${buck_output})

  # Assemble the list of source files, which live under the buck output dir.
  set(_srcfiles)
  foreach(gen ${_portable_kernels_bindings__generated_files})
    list(APPEND _srcfiles "${CMAKE_CURRENT_SOURCE_DIR}/${srcdir}/${gen}")
  endforeach()

  file(MAKE_DIRECTORY ${_portable_kernels_bindings__output_dir})
  file(COPY ${_srcfiles} DESTINATION ${_portable_kernels_bindings__output_dir})
  message(STATUS "portable_kernels_bindings: "
                 "Copied files to ${_portable_kernels_bindings__output_dir}")
else()
  message(
    "Error occurred while executing buck2 command. Exit code: ${buck_exit_code}"
  )
  message("Buck2 Output:\n${buck_output}")
  message("Buck2 Error:\n${buck_error}")
  message(FATAL_ERROR "portable_kernels_bindings: codegen failed")
endif()

add_library(portable_kernels_bindings)
target_sources(portable_kernels_bindings
               PRIVATE ${_portable_kernels_bindings__cpp_files})
target_link_libraries(portable_kernels_bindings PRIVATE executorch)
target_link_libraries(portable_kernels_bindings INTERFACE portable_kernels)

# Ensure that the load-time constructor functions run. By default, the linker
# would remove them since there are no other references to them.
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL
                                             "AppleClang")
  if(CMAKE_TOOLCHAIN_FILE MATCHES ".*android\.toolchain\.cmake$")
    # For Android tool chain
    gcc_kernel_link_options(portable_kernels_bindings)
  else()
    # For host tool chain
    clang_kernel_link_options(portable_kernels_bindings)
  endif()
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # Using gcc
  gcc_kernel_link_options(portable_kernels_bindings)
endif()

#
# executor_runner: A simple commandline tool that loads and runs a program file.
#

# ios can only build library but not binary
if(NOT CMAKE_TOOLCHAIN_FILE MATCHES ".*ios\.toolchain\.cmake$")
  add_executable(executor_runner ${_executor_runner__srcs})
  target_link_libraries(executor_runner executorch portable_kernels_bindings
                        gflags)
  target_compile_options(executor_runner PUBLIC ${_common_compile_options})
endif()

# Generate custom_ops_lib based on REGISTER_EXAMPLE_CUSTOM_OPS
if(REGISTER_EXAMPLE_CUSTOM_OP_1 OR REGISTER_EXAMPLE_CUSTOM_OP_2)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/examples/custom_ops)
  target_link_libraries(executor_runner custom_ops_lib)
endif()

# Print all summary
executorch_print_configuration_summary()
