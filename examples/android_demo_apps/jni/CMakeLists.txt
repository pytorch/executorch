# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# Please this file formatted by running:
# ~~~
# cmake-format --first-comment-is-literal=True CMakeLists.txt
# ~~~

cmake_minimum_required(VERSION 3.19)

if(NOT CMAKE_TOOLCHAIN_FILE MATCHES ".*ios\.toolchain\.cmake$")
  add_library(executorchdemo SHARED jni_layer.cpp)
  target_link_libraries(executorchdemo xnn_executor_runner_lib fbjni)
  target_compile_options(executorchdemo PUBLIC ${_common_compile_options})
endif()
