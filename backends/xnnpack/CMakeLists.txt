# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024-2025 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Please this file formatted by running:
# ~~~
# cmake-format -i CMakeLists.txt
# ~~~

cmake_minimum_required(VERSION 3.24)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

if(EXECUTORCH_XNNPACK_ENABLE_WEIGHT_CACHE)
  add_definitions(-DENABLE_XNNPACK_WEIGHTS_CACHE)
endif()
if(EXECUTORCH_XNNPACK_SHARED_WORKSPACE)
  add_definitions(-DENABLE_XNNPACK_SHARED_WORKSPACE)
endif()
if(EXECUTORCH_XNNPACK_ENABLE_KLEIDI)
  add_definitions(-DENABLE_XNNPACK_KLEIDI)
endif()

set(_common_include_directories ${EXECUTORCH_ROOT}/..)
set(_common_compile_options -Wno-deprecated-declarations -fPIC)

set(_xnnpack_schema__include_dir "${CMAKE_BINARY_DIR}/schema/include")
# Paths to headers generated from the .fbs files.
set(_xnnpack_flatbuffer__outputs)
foreach(fbs_file ${_xnnpack_schema__srcs})
  string(REGEX REPLACE "([^/]+)[.]fbs$" "\\1_generated.h" generated
                       "${fbs_file}"
  )
  list(APPEND _xnnpack_flatbuffer__outputs
       "${_xnnpack_schema__include_dir}/executorch/${generated}"
  )
endforeach()

set(_xnnpack_schema__outputs)
foreach(fbs_file ${_xnnpack_schema__srcs})
  string(REGEX REPLACE "runtime_([^/]+)[.]fbs$" "\\1_generated.h" generated
                       "${fbs_file}"
  )
  list(APPEND _xnnpack_schema__outputs
       "${_xnnpack_schema__include_dir}/executorch/${generated}"
  )
endforeach()

if(WIN32)
  set(MV_COMMAND
      powershell -Command
      "Move-Item -Path ${_xnnpack_flatbuffer__outputs} -Destination ${_xnnpack_schema__outputs}"
  )
else()
  set(MV_COMMAND mv ${_xnnpack_flatbuffer__outputs} ${_xnnpack_schema__outputs})
endif()

# Generate the headers from the .fbs files.
add_custom_command(
  OUTPUT ${_xnnpack_schema__outputs}
  COMMAND
    flatc --cpp --cpp-std c++11 --scoped-enums -o
    "${_xnnpack_schema__include_dir}/executorch/backends/xnnpack/serialization"
    ${_xnnpack_schema__srcs}
  COMMAND ${MV_COMMAND}
  WORKING_DIRECTORY ${EXECUTORCH_ROOT}
  DEPENDS flatc
  COMMENT "Generating xnnpack_schema headers"
  VERBATIM
)

unset(MV_COMMAND)

add_library(xnnpack_schema INTERFACE ${_xnnpack_schema__outputs})
set_target_properties(xnnpack_schema PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(
  xnnpack_schema INTERFACE ${_xnnpack_schema__include_dir}
                           ${EXECUTORCH_ROOT}/third-party/flatbuffers/include
)

set(xnnpack_third_party pthreadpool extension_threadpool cpuinfo)

include(cmake/Dependencies.cmake)

list(TRANSFORM _xnnpack_backend__srcs PREPEND "${EXECUTORCH_ROOT}/")
add_library(xnnpack_backend ${_xnnpack_backend__srcs})
target_link_libraries(
  xnnpack_backend PUBLIC ${xnnpack_third_party} executorch_core xnnpack_schema
                         extension_threadpool
)

target_include_directories(
  xnnpack_backend PUBLIC ${_common_include_directories}
)
target_include_directories(xnnpack_backend PUBLIC ${XNNPACK_INCLUDE_DIR})
target_include_directories(
  xnnpack_backend
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/third-party/pthreadpool/include
)
target_include_directories(
  xnnpack_backend
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/third-party/cpuinfo/include
)
target_compile_options(xnnpack_backend PUBLIC ${_common_compile_options})
target_link_options_shared_lib(xnnpack_backend)

install(
  TARGETS xnnpack_backend
  INCLUDES
  DESTINATION ${_common_include_directories}
)

if(BUILD_TESTING)
  add_subdirectory(test)
endif()
