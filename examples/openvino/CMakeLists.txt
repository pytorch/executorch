# Copyright (c) Intel Corporation
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file in the root
# directory of this source tree for more details.

set(CMAKE_CXX_STANDARD 17)

cmake_minimum_required(VERSION 3.19)
project(openvino_runner_example)

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

include(${EXECUTORCH_ROOT}/build/Utils.cmake)
include(${EXECUTORCH_ROOT}/build/Codegen.cmake)

if(NOT PYTHON_EXECUTABLE)
  resolve_python_executable()
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

set(_common_compile_options -Wno-deprecated-declarations -fPIC)

# Let files say "include <executorch/path/to/header.h>".
set(_common_include_directories ${EXECUTORCH_ROOT}/..)

#
# The `_<target>_srcs` lists are defined by including ${EXECUTORCH_SRCS_FILE}.
#
set(EXECUTORCH_SRCS_FILE
    "${CMAKE_CURRENT_BINARY_DIR}/../../../build/executorch_srcs.cmake"
)
extract_sources(${EXECUTORCH_SRCS_FILE})
include(${EXECUTORCH_SRCS_FILE})

set(_openvino_executor_runner__srcs ${CMAKE_CURRENT_LIST_DIR}/../openvino/executor_runner/openvino_executor_runner.cpp) 

# preprocess executor runner src files
list(PREPEND _openvino_executor_runner__srcs
     ${CMAKE_CURRENT_LIST_DIR}/../openvino/executor_runner/openvino_executor_runner.cpp
)

find_package(executorch CONFIG REQUIRED)
target_include_directories(executorch INTERFACE ${_common_include_directories})
target_compile_options(executorch INTERFACE ${_common_compile_options})

# portable_ops_lib
gen_selected_ops(LIB_NAME "openvino_portable_ops_lib" INCLUDE_ALL_OPS "ON")
generate_bindings_for_kernels(
  LIB_NAME "openvino_portable_ops_lib" FUNCTIONS_YAML
  ${EXECUTORCH_ROOT}/backends/openvino/openvino_functions.yaml
)
gen_operators_lib(
  LIB_NAME "openvino_portable_ops_lib" KERNEL_LIBS portable_kernels DEPS executorch
)
target_compile_options(
  openvino_portable_ops_lib INTERFACE -DET_EVENT_TRACER_ENABLED
)
target_include_directories(
  openvino_portable_ops_lib PUBLIC ${_common_include_directories}
)


# build executor runner
add_executable(openvino_executor_runner ${_openvino_executor_runner__srcs})
target_include_directories(
  openvino_executor_runner PUBLIC ${_common_include_directories}
)

# Set the path to the library directory
set(LIBRARY_DIR "${CMAKE_CURRENT_LIST_DIR}/../../cmake-openvino-out/lib/")

# List the libraries you want to link (without the 'lib' prefix and file extension)
set(LIBRARIES_TO_LINK ${LIBRARY_DIR}/libopenvino_backend.so 
                      ${LIBRARY_DIR}/libexecutorch.a 
                      ${LIBRARY_DIR}/libexecutorch_core.a 
                      ${EXECUTORCH_ROOT}/third-party/gflags/build/lib/libgflags_nothreads.a 
                      ${LIBRARY_DIR}/libpthreadpool.a 
                      ${LIBRARY_DIR}/libextension_data_loader.a 
                      ${LIBRARY_DIR}/libextension_runner_util.a 
)

# Add the library directory to the link search path
link_directories(${LIBRARY_DIR})

# Link all libraries at once
target_link_libraries(openvino_executor_runner PRIVATE ${LIBRARIES_TO_LINK} openvino_portable_ops_lib)

set_target_properties(
  openvino_executor_runner PROPERTIES LINK_FLAGS "-Wl,-rpath='$ORIGIN'"
)


get_filename_component(
  EXECUTORCH_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE
)


