# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# executorch_backend
target_sources(
  executorch_backend INTERFACE ${EXECUTORCH_SOURCE_DIR}/backends/backend.h
)

# qnn_executorch_header
target_sources(
  qnn_executorch_header INTERFACE ${CMAKE_CURRENT_LIST_DIR}/QnnExecuTorch.h
)

# qnn_executorch_backend
target_sources(
  qnn_executorch_backend
  INTERFACE ${CMAKE_CURRENT_LIST_DIR}/QnnExecuTorchBackend.h
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/QnnExecuTorchBackend.cpp
)

# qnn_manager
target_sources(
  qnn_manager
  INTERFACE ${CMAKE_CURRENT_LIST_DIR}/QnnManager.h
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/QnnManager.cpp
)

# logging
target_sources(
  qnn_executorch_logging
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/Logging.h
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/Logging.cpp
)

# utils
target_sources(
  utils
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/Utils.h
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/Utils.cpp
)

# shared_buffer
target_sources(
  shared_buffer PRIVATE ${CMAKE_CURRENT_LIST_DIR}/SharedBuffer.h
                        ${CMAKE_CURRENT_LIST_DIR}/SharedBuffer.cpp
)
