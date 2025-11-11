# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Please this file formatted by running:
# ~~~
# cmake-format -i CMakeLists.txt
# ~~~

cmake_minimum_required(VERSION 3.19)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../)
endif()

include(${EXECUTORCH_ROOT}/tools/cmake/Test.cmake)
include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

set(_common_include_directories ${EXECUTORCH_ROOT}/..)
set(_common_compile_options -Wno-deprecated-declarations -fPIC)

add_library(
  executor_backend STATIC ExecutorBackendRegister.cpp ExecutorBackend.cpp
)
target_link_libraries(executor_backend PRIVATE executorch_core)

target_include_directories(
  executor_backend PUBLIC ${_common_include_directories}
)
install(
  TARGETS executor_backend
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES
  DESTINATION ${_common_include_directories}
)
