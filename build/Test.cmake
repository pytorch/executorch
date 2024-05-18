# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# This file is intended to have helper functions for test-related
# CMakeLists.txt files.
#
# ### Editing this file ###
#
# This file should be formatted with
# ~~~
# cmake-format -i Utils.cmake
# ~~~
# It should also be cmake-lint clean.
#

function(et_cxx_test target_name)
set(multi_arg_names SOURCES EXTRA_LIBS)
cmake_parse_arguments(ET_CXX_TEST "" "" "${multi_arg_names}" ${ARGN})

message(${ET_CXX_TEST_SOURCES})

# Find prebuilt executorch library
find_package(executorch CONFIG REQUIRED)

enable_testing()
find_package(GTest CONFIG REQUIRED)

# Let files say "include <executorch/path/to/header.h>".
set(_common_include_directories ${EXECUTORCH_ROOT}/..)
target_include_directories(executorch INTERFACE ${_common_include_directories})

add_executable(${target_name} ${ET_CXX_TEST_SOURCES})
# Includes gtest, gmock, executorch by default
target_link_libraries(
  ${target_name} GTest::gtest GTest::gtest_main GTest::gmock executorch
  ${ET_CXX_TEST_EXTRA_LIBS}
)
add_test(ExecuTorchTest ${target_name})

endfunction()
