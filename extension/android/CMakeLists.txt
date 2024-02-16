# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


cmake_minimum_required(VERSION 3.19)

project(executorch_jni)

set(EXECUTORCH_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
include(${EXECUTORCH_ROOT}/build/Utils.cmake)

set(_common_include_directories ${EXECUTORCH_ROOT}/..)

add_subdirectory(
        ${EXECUTORCH_ROOT}/examples/third-party/fbjni
        ${CMAKE_CURRENT_BINARY_DIR}/third-party/fbjni)

if(NOT CMAKE_TOOLCHAIN_FILE MATCHES ".*ios\.toolchain\.cmake$")
    add_library(executorch_jni SHARED jni/jni_layer.cpp)
    target_link_libraries(executorch_jni extension_data_loader
        extension_module xnn_executor_runner_lib fbjni)
    if(EXECUTORCH_BUILD_QNN)
        target_link_libraries(executorch_jni qnn_executorch_backend)
    endif()
    target_compile_options(executorch_jni PUBLIC ${_common_compile_options})
endif()
