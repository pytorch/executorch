# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# wrappers
target_sources(
  wrappers
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/TensorWrapper.cpp
         ${CMAKE_CURRENT_LIST_DIR}/TensorWrapper.h
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/QuantizeParamsWrapper.cpp
          ${CMAKE_CURRENT_LIST_DIR}/QuantizeParamsWrapper.h
          ${CMAKE_CURRENT_LIST_DIR}/OpWrapper.cpp
          ${CMAKE_CURRENT_LIST_DIR}/OpWrapper.h
          ${CMAKE_CURRENT_LIST_DIR}/ParamWrapper.h
          ${CMAKE_CURRENT_LIST_DIR}/ScalarParamWrapper.h
          ${CMAKE_CURRENT_LIST_DIR}/TensorParamWrapper.h
)
