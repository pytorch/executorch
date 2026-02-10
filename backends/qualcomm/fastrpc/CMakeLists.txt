# Copyright (c) Qualcomm Innovation Center, Inc.
# Copyright 2025 Arm Limited and/or its affiliates.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(_qnn_fastrpc__dir ${CMAKE_BINARY_DIR}/backends/qualcomm/fastrpc)
set(_qnn_fastrpc__srcs ${CMAKE_CURRENT_LIST_DIR}/qnn_executorch.idl)
set(_qnn_fastrpc__outputs
    ${_qnn_fastrpc__dir}/qnn_executorch.h
    ${_qnn_fastrpc__dir}/qnn_executorch_stub.c
    ${_qnn_fastrpc__dir}/qnn_executorch_skel.c
)

if(DEFINED ENV{HEXAGON_SDK_ROOT})
  add_custom_command(
    OUTPUT ${_qnn_fastrpc__outputs}
    COMMAND mkdir -p ${_qnn_fastrpc__dir}
    COMMAND
      $ENV{HEXAGON_SDK_ROOT}/ipc/fastrpc/qaic/bin/qaic -I
      $ENV{HEXAGON_SDK_ROOT}/incs -I $ENV{HEXAGON_SDK_ROOT}/incs/stddef -o
      ${_qnn_fastrpc__dir} ${_qnn_fastrpc__srcs}
    WORKING_DIRECTORY ${EXECUTORCH_SOURCE_DIR}
    DEPENDS qnn_executorch_backend
    COMMENT "Codegen for fastrpc files"
  )
  add_custom_target(
    fastrpc_codegen
    DEPENDS ${_qnn_fastrpc__outputs}
    COMMENT "Codegen for fastrpc files"
  )

endif()

if(${CMAKE_SYSTEM_PROCESSOR} MATCHES Hexagon)
  add_library(
    qnn_executorch_skel SHARED
    ${_qnn_fastrpc__dir}/qnn_executorch.h
    ${_qnn_fastrpc__dir}/qnn_executorch_skel.c qnn_executorch_impl.cpp
  )
  target_include_directories(qnn_executorch_skel PRIVATE ${_qnn_fastrpc__dir})
  target_link_libraries(
    qnn_executorch_skel PRIVATE extension_data_loader qnn_executorch_backend
                                c++ c
  )
  add_dependencies(qnn_executorch_skel fastrpc_codegen)
endif()

if(${CMAKE_SYSTEM_PROCESSOR} MATCHES aarch64)
  include_directories(
    $ENV{HEXAGON_SDK_ROOT}/incs $ENV{HEXAGON_SDK_ROOT}/incs/stddef
    ${_qnn_fastrpc__dir}
  )
  link_directories(
    $ENV{HEXAGON_SDK_ROOT}/ipc/fastrpc/remote/ship/android_aarch64
  )
  add_library(
    qnn_executorch_stub SHARED ${_qnn_fastrpc__dir}/qnn_executorch.h
                               ${_qnn_fastrpc__dir}/qnn_executorch_stub.c
  )
  # TODO: support cdsp if necessary
  target_link_libraries(qnn_executorch_stub PRIVATE adsprpc)
  add_dependencies(qnn_executorch_stub fastrpc_codegen)

  # build minimum example app
  add_executable(qnn_executor_runner qnn_executor_runner.cpp)
  target_link_libraries(
    qnn_executor_runner PRIVATE executorch_core gflags qnn_executorch_stub
                                adsprpc
  )
  # TODO: support cdsp if necessary
  target_link_libraries(qnn_executor_runner PRIVATE adsprpc)
endif()
