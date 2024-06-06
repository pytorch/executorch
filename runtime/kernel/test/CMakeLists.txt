# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# This file should be formatted with
# ~~~
# cmake-format -i CMakeLists.txt
# ~~~
# It should also be cmake-lint clean.
#

cmake_minimum_required(VERSION 3.19)
project(runtime_kernel_test)

# Use C++17 for test.
set(CMAKE_CXX_STANDARD 17)

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

include(${EXECUTORCH_ROOT}/build/Test.cmake)

set(_test_srcs
    kernel_runtime_context_test.cpp
    operator_registry_test
)

et_cxx_test(runtime_kernel_test SOURCES ${_test_srcs} EXTRA_LIBS)

et_cxx_test(runtime_max_kernel_num_test SOURCES operator_registry_max_kernel_num_test)

et_cxx_test(runtime_double_registration_test SOURCES kernel_double_registration_test)
