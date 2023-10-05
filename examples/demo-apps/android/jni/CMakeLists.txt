# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


cmake_minimum_required(VERSION 3.19)

add_subdirectory(third-party/fbjni)

if(NOT CMAKE_TOOLCHAIN_FILE MATCHES ".*ios\.toolchain\.cmake$")
  add_library(executorchdemo SHARED jni_layer.cpp)
  target_link_libraries(executorchdemo xnn_executor_runner_lib fbjni)
  if(EXECUTORCH_BUILD_QNN)
    target_link_libraries(executorchdemo qnn_executorch_backend)
  endif()
  target_compile_options(executorchdemo PUBLIC ${_common_compile_options})
endif()
