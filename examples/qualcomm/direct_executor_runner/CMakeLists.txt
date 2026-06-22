# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

if(NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES aarch64)
  message(
    FATAL_ERROR
      "This direct mode runner requires aarch64, but found: ${CMAKE_SYSTEM_PROCESSOR}"
  )
endif()

# Avoid CMAKE cache env variable when non-clean build
set(HEXAGON_SDK_ROOT $ENV{HEXAGON_SDK_ROOT})

# set qnn_executorch_stub as build_idl dependent so will perform build_idl first
add_library(
  qnn_executorch_stub SHARED ${CMAKE_CURRENT_BINARY_DIR}/qnn_executorch_stub.c
                             ${CMAKE_CURRENT_BINARY_DIR}/qnn_executorch.h
)

# Ignore this for now, this is for unused feature in hexagon_fun.cmake.
set(PREBUILT_LIB_DIR "android_aarch64")
# Include Hexagon SDK helper functions.
include(${HEXAGON_SDK_ROOT}/build/cmake/hexagon_fun.cmake)

include_directories(
  ${HEXAGON_SDK_ROOT}/incs ${HEXAGON_SDK_ROOT}/incs/stddef
  ${CMAKE_CURRENT_BINARY_DIR} ${HEXAGON_SDK_ROOT}/utils/examples
)

set(_qnn_executorch_idl_src
    ${CMAKE_SOURCE_DIR}/../../backends/qualcomm/runtime/backends/direct_mode/qnn_executorch.idl
)
message("CURRENT BINARY DIR ${CMAKE_CURRENT_BINARY_DIR}")
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/qnn_executorch_stub.c
         ${CMAKE_CURRENT_BINARY_DIR}/qnn_executorch_skel.c
         ${CMAKE_CURRENT_BINARY_DIR}/qnn_executorch.h
  COMMAND
    ${HEXAGON_SDK_ROOT}/ipc/fastrpc/qaic/Ubuntu/qaic
    ${_qnn_executorch_idl_src} # Path to .idl file
    -I ${HEXAGON_SDK_ROOT}/incs -I ${HEXAGON_SDK_ROOT}/incs/stddef -o
    ${CMAKE_CURRENT_BINARY_DIR}
  WORKING_DIRECTORY ${EXECUTORCH_SOURCE_DIR}
  DEPENDS ${_qnn_executorch_idl_src}
  COMMENT "Codegen for fastrpc files"
)

# Function in hexagon_fun.cmake. Find the DSP from provided domain value,
# default is CDSP
choose_dsprpc(${DSP_TYPE} dsprpc)

message(
  STATUS "Building runner for domain: ${DSP_TYPE}. Library name: ${dsprpc}"
)

# To link cdsprpc or adsprpc libraries
set(DSPRPC_LIB
    ${HEXAGON_SDK_ROOT}/ipc/fastrpc/remote/ship/android_aarch64/lib${dsprpc}.so
)
target_link_libraries(qnn_executorch_stub PRIVATE ${DSPRPC_LIB})

add_executable(
  qnn_executor_direct_runner
  qnn_executor_direct_runner.cpp
  ${HEXAGON_SDK_ROOT}/utils/examples/dsp_capabilities_utils.c
)

target_link_libraries(
  qnn_executor_direct_runner PRIVATE executorch_core gflags qnn_executorch_stub
                                     ${DSPRPC_LIB}
)
