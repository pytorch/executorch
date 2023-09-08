# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# The logic is copied from
# https://github.com/pytorch/pytorch/blob/main/cmake/Dependencies.cmake
set(THIRD_PARTY_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third-party")

# --- cpuinfo
set(CPUINFO_SOURCE_DIR "${THIRD_PARTY_ROOT}/cpuinfo")
set(CPUINFO_BUILD_TOOLS OFF CACHE BOOL "")
set(CPUINFO_BUILD_UNIT_TESTS OFF CACHE BOOL "")
set(CPUINFO_BUILD_MOCK_TESTS OFF CACHE BOOL "")
set(CPUINFO_BUILD_BENCHMARKS OFF CACHE BOOL "")
set(CPUINFO_LIBRARY_TYPE "static" CACHE STRING "")
set(CPUINFO_LOG_LEVEL "error" CACHE STRING "")
set(CLOG_SOURCE_DIR "${CPUINFO_SOURCE_DIR}/deps/clog")
add_subdirectory("${CPUINFO_SOURCE_DIR}")
list(APPEND xnn_executor_runner_libs cpuinfo)

# --- pthreadpool
set(PTHREADPOOL_SOURCE_DIR "${THIRD_PARTY_ROOT}/pthreadpool")
set(PTHREADPOOL_BUILD_TESTS OFF CACHE BOOL "")
set(PTHREADPOOL_BUILD_BENCHMARKS OFF CACHE BOOL "")
set(PTHREADPOOL_LIBRARY_TYPE "static" CACHE STRING "")
set(PTHREADPOOL_ALLOW_DEPRECATED_API ON CACHE BOOL "")
add_subdirectory("${PTHREADPOOL_SOURCE_DIR}")
list(APPEND xnn_executor_runner_libs pthreadpool)

# --- XNNPACK
set(XNNPACK_SOURCE_DIR "${THIRD_PARTY_ROOT}/XNNPACK")
set(XNNPACK_INCLUDE_DIR "${XNNPACK_SOURCE_DIR}/include")
set(XNNPACK_LIBRARY_TYPE "static" CACHE STRING "")
set(XNNPACK_BUILD_BENCHMARKS OFF CACHE BOOL "")
set(XNNPACK_BUILD_TESTS OFF CACHE BOOL "")
add_subdirectory("${XNNPACK_SOURCE_DIR}")
include_directories(SYSTEM ${XNNPACK_INCLUDE_DIR})
list(APPEND xnn_executor_runner_libs XNNPACK)
