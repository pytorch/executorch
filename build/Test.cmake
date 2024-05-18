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

# A helper function to generate a gtest cxx executable target
# @param target_name: name for the executable
# @param SOURCES <list_of_sources>: test sources to be compiled. Sometimes
# util sources are used as well
# @param EXTRA LIBS <list_of_libs>: additional libraries to be linked against
# the target. gtest, gmock, executorch are linked by default, but Sometimes
# user may need additional libraries like kernels.
# We use CMake package executorch in this helper, so user can easily add
# installed libraries.
#
# Example:
# et_cxx_test(my_test SOURCES my_test.cpp EXTRA_LIBS portable_kernels)
#
# This defines a gtest executable my_test, compiling my_test.cpp, and linking
# against libportable_kernels.a.
#
function(et_cxx_test target_name)

set(multi_arg_names SOURCES EXTRA_LIBS)
cmake_parse_arguments(ET_CXX_TEST "" "" "${multi_arg_names}" ${ARGN})

# Find prebuilt executorch library
find_package(executorch CONFIG REQUIRED)

enable_testing()
find_package(GTest CONFIG REQUIRED)

# Let files say "include <executorch/path/to/header.h>".
target_include_directories(executorch INTERFACE ${EXECUTORCH_ROOT}/..)

add_executable(${target_name} ${ET_CXX_TEST_SOURCES})
# Includes gtest, gmock, executorch by default
target_link_libraries(
  ${target_name} GTest::gtest GTest::gtest_main GTest::gmock executorch
  ${ET_CXX_TEST_EXTRA_LIBS}
)

# add_test adds a test target to be used by ctest.
# We use `ExecuTorchTest` as the ctest target name for the test executable
# Usage: cd cmake-out/path/to/test/; ctest
# Note: currently we directly invoke the test target, without using ctest
add_test(ExecuTorchTest ${target_name})

endfunction()
