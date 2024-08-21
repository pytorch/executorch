#
# Copyright (c) 2023 Apple Inc. All rights reserved. Provided subject to the
# LICENSE file in the top level directory.
#

cmake_minimum_required(VERSION 3.19)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
endif()

include(${EXECUTORCH_ROOT}/build/Utils.cmake)

if(NOT PYTHON_EXECUTABLE)
  resolve_python_executable()
endif()

if(NOT FLATC_EXECUTABLE)
  set(FLATC_EXECUTABLE flatc)
endif()

set(_common_compile_options -Wno-deprecated-declarations)
set(_common_include_directories ${EXECUTORCH_ROOT}/..)

set(_mps_schema__include_dir "${CMAKE_BINARY_DIR}/schema/include")

# Paths to headers generated from the .fbs files.
set(_mps_schema__outputs)
foreach(fbs_file ${_mps_schema__srcs})
  string(REGEX REPLACE "serialization/([^/]+)[.]fbs$" "\\1_generated.h"
                       generated "${fbs_file}"
  )
  list(APPEND _mps_schema__outputs
       "${_mps_schema__include_dir}/executorch/${generated}"
  )
endforeach()

# Generate the headers from the .fbs files.
add_custom_command(
  OUTPUT ${_mps_schema__outputs}
  COMMAND
    ${FLATC_EXECUTABLE} --cpp --cpp-std c++11 --scoped-enums -o
    "${_mps_schema__include_dir}/executorch/backends/apple/mps"
    ${_mps_schema__srcs}
  WORKING_DIRECTORY ${EXECUTORCH_ROOT}
  COMMENT "Generating mps_schema headers"
  VERBATIM
)

add_library(mps_schema INTERFACE ${_mps_schema__outputs})
set_target_properties(mps_schema PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(
  mps_schema
  INTERFACE ${_mps_schema__include_dir}
            ${EXECUTORCH_ROOT}/third-party/flatbuffers/include
            ${_common_include_directories}
)

list(TRANSFORM _mps_backend__srcs PREPEND "${EXECUTORCH_ROOT}/")
add_library(mpsdelegate ${_mps_backend__srcs})

find_library(FOUNDATION_FRAMEWORK Foundation)
find_library(METAL_FRAMEWORK Metal)
find_library(MPS_FRAMEWORK MetalPerformanceShaders)
find_library(MPS_GRAPH_FRAMEWORK MetalPerformanceShadersGraph)

target_link_libraries(
  mpsdelegate
  PRIVATE bundled_program
          mps_schema
          executorch_no_prim_ops
          ${FOUNDATION_FRAMEWORK}
          ${METAL_FRAMEWORK}
          ${MPS_FRAMEWORK}
          ${MPS_GRAPH_FRAMEWORK}
)

target_link_options_shared_lib(mpsdelegate)
target_compile_options(mpsdelegate PUBLIC ${_common_compile_options})
target_compile_options(mpsdelegate PRIVATE "-fno-objc-arc")

install(
  TARGETS mpsdelegate
  DESTINATION lib
  INCLUDES
  DESTINATION ${_common_include_directories}
)
