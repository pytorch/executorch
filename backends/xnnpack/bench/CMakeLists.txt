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

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../../..)
endif()

set(XNNPACK_INCLUDE_DIR
"${EXECUTORCH_ROOT}/backends/xnnpack/third-party/XNNPACK/include")
set(PTHREADPOOL_INCLUDE_DIR
"${EXECUTORCH_ROOT}/backends/xnnpack/third-party/pthreadpool/include")
add_executable(model_bench bench_model.cpp)
target_link_libraries(model_bench executorch xnn_executor_runner_lib)
target_include_directories(model_bench
                           PUBLIC ${_common_include_directories})
target_include_directories(model_bench PUBLIC ${XNNPACK_INCLUDE_DIR})
target_include_directories(model_bench PUBLIC ${PTHREADPOOL_INCLUDE_DIR})
target_compile_options(model_bench PUBLIC ${_common_compile_options})
