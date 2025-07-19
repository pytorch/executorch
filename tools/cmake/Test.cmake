# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# This file is intended to have helper functions for test-related CMakeLists.txt
# files.
#
# ### Editing this file ###
#
# This file should be formatted with
# ~~~
# cmake-format -i Test.cmake
# ~~~
# It should also be cmake-lint clean.
#

# A helper function to generate a gtest cxx executable target @param
# target_name: name for the executable @param SOURCES <list_of_sources>: test
# sources to be compiled. Sometimes util sources are used as well @param EXTRA
# LIBS <list_of_libs>: additional libraries to be linked against the target.
# gtest, gmock, executorch are linked by default, but Sometimes user may need
# additional libraries like kernels. We use CMake package executorch in this
# helper, so user can easily add installed libraries.
#
# Example: et_cxx_test(my_test SOURCES my_test.cpp EXTRA_LIBS portable_kernels)
#
# This defines a gtest executable my_test, compiling my_test.cpp, and linking
# against libportable_kernels.a.
#
function(et_cxx_test target_name)

  set(multi_arg_names SOURCES EXTRA_LIBS)
  cmake_parse_arguments(ET_CXX_TEST "" "" "${multi_arg_names}" ${ARGN})

  add_executable(
    ${target_name}
    ${ET_CXX_TEST_SOURCES}
    ${EXECUTORCH_ROOT}/runtime/core/exec_aten/testing_util/tensor_util.cpp
  )
  if(NOT TARGET GTest::gtest)
    find_package(GTest REQUIRED)
  endif()
  # Includes gtest, gmock, executorch_core by default
  target_link_libraries(
    ${target_name} GTest::gtest GTest::gtest_main GTest::gmock executorch_core
    ${ET_CXX_TEST_EXTRA_LIBS}
  )

  # add_test adds a test target to be used by ctest
  add_test(NAME ${target_name} COMMAND ${target_name})

endfunction()
