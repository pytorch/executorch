# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# PyQnnManagerAdaptor
target_sources(
  PyQnnManagerAdaptor PUBLIC ${CMAKE_CURRENT_LIST_DIR}/PyQnnManagerAdaptor.cpp
                             ${CMAKE_CURRENT_LIST_DIR}/PyQnnManagerAdaptor.h
)

# PyQnnWrapperAdaptor
target_sources(
  PyQnnWrapperAdaptor PUBLIC ${CMAKE_CURRENT_LIST_DIR}/PyQnnWrapperAdaptor.cpp
                             ${CMAKE_CURRENT_LIST_DIR}/PyQnnWrapperAdaptor.h
)
