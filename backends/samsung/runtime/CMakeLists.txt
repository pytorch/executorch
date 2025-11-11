# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# logging
target_sources(
  enn_logging
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/logging.h
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/logging.cpp
)

if(${ANDROID})
  # backend
  target_sources(
    enn_backend
    PRIVATE ${CMAKE_CURRENT_LIST_DIR}/enn_backend.cpp
            ${CMAKE_CURRENT_LIST_DIR}/enn_executor.cpp
            ${CMAKE_CURRENT_LIST_DIR}/enn_api_implementation.cpp
  )
endif()
