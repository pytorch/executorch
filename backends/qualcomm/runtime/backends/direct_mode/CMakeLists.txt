# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

if(NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES Hexagon)
  message(
    FATAL_ERROR
      "This requires Hexagon tool chain, but found: ${CMAKE_SYSTEM_PROCESSOR}"
  )
endif()

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/qnn_executorch_stub.c
         ${CMAKE_CURRENT_BINARY_DIR}/qnn_executorch_skel.c
         ${CMAKE_CURRENT_BINARY_DIR}/qnn_executorch.h
  COMMAND
    ${HEXAGON_SDK_ROOT}/ipc/fastrpc/qaic/Ubuntu/qaic
    ${CMAKE_CURRENT_SOURCE_DIR}/qnn_executorch.idl # Path to .idl file
    -I ${HEXAGON_SDK_ROOT}/incs -I ${HEXAGON_SDK_ROOT}/incs/stddef -o
    ${CMAKE_CURRENT_BINARY_DIR}
  WORKING_DIRECTORY ${EXECUTORCH_SOURCE_DIR}
  DEPENDS qnn_executorch_backend
  COMMENT "Codegen for fastrpc files"
)

add_library(
  qnn_executorch_skel SHARED
  ${CMAKE_CURRENT_BINARY_DIR}/qnn_executorch.h
  ${CMAKE_CURRENT_BINARY_DIR}/qnn_executorch_skel.c qnn_executorch_imp.cpp
  QnnExecuTorchIdlWrapper.h QnnExecuTorchIdlWrapper.cpp
)

target_include_directories(
  qnn_executorch_skel PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
)
target_link_libraries(
  qnn_executorch_skel
  PRIVATE
    extension_data_loader
    qnn_executorch_backend
    etdump
    ${HEXAGON_TOOLS_ROOT}/Tools/target/hexagon/lib/${DSP_VERSION}/G0/pic/libc.so
    ${HEXAGON_TOOLS_ROOT}/Tools/target/hexagon/lib/${DSP_VERSION}/G0/pic/libc++.so.1
    ${HEXAGON_TOOLS_ROOT}/Tools/target/hexagon/lib/${DSP_VERSION}/G0/pic/libc++abi.so.1
)
