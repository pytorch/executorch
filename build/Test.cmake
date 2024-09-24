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

include(${EXECUTORCH_ROOT}/build/Utils.cmake)

# Find prebuilt executorch library
find_package(executorch CONFIG REQUIRED)

enable_testing()
find_package(GTest CONFIG REQUIRED)

target_link_options_shared_lib(cpuinfo)
target_link_options_shared_lib(extension_data_loader)
target_link_options_shared_lib(portable_kernels)
target_link_options_shared_lib(portable_ops_lib)
target_link_options_shared_lib(pthreadpool)
target_link_options_shared_lib(quantized_ops_lib)

# Add code coverage flags to supported compilers
if(EXECUTORCH_USE_CPP_CODE_COVERAGE)
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    string(APPEND CMAKE_C_FLAGS " --coverage -fprofile-abs-path")
    string(APPEND CMAKE_CXX_FLAGS " --coverage -fprofile-abs-path")
  elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    string(APPEND CMAKE_C_FLAGS " -fprofile-instr-generate -fcoverage-mapping")
    string(APPEND CMAKE_CXX_FLAGS
           " -fprofile-instr-generate -fcoverage-mapping"
    )
  else()
    message(ERROR
            "Code coverage for compiler ${CMAKE_CXX_COMPILER_ID} is unsupported"
    )
  endif()
endif()

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

  # Let files say "include <executorch/path/to/header.h>".
  target_include_directories(executorch INTERFACE ${EXECUTORCH_ROOT}/..)

  set(ET_TEST_UTIL_SOURCES
      ${EXECUTORCH_ROOT}/runtime/core/exec_aten/testing_util/tensor_util.cpp
  )

  add_executable(${target_name} ${ET_CXX_TEST_SOURCES} ${ET_TEST_UTIL_SOURCES})
  # Includes gtest, gmock, executorch by default
  target_link_libraries(
    ${target_name} GTest::gtest GTest::gtest_main GTest::gmock executorch
    ${ET_CXX_TEST_EXTRA_LIBS}
  )

  # add_test adds a test target to be used by ctest. We use `ExecuTorchTest` as
  # the ctest target name for the test executable Usage: cd
  # cmake-out/path/to/test/; ctest Note: currently we directly invoke the test
  # target, without using ctest
  add_test(ExecuTorchTest ${target_name})

endfunction()
