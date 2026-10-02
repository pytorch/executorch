# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.24)
project(manual_registration_test LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
if(NOT CODEGEN_CMAKE)
  set(CODEGEN_CMAKE ${EXECUTORCH_ROOT}/tools/cmake/Codegen.cmake)
endif()
include(${CODEGEN_CMAKE})

if(TEST_CASE STREQUAL "integration")
  find_package(executorch CONFIG REQUIRED)
  include(
    ${EXECUTORCH_ROOT}/examples/portable/custom_ops/named_manual_registration.cmake
  )
  gen_selected_ops(LIB_NAME automatic_ops_lib ROOT_OPS "my_ops::mul3.out")
  generate_bindings_for_kernels(
    LIB_NAME automatic_ops_lib CUSTOM_OPS_YAML
    ${EXECUTORCH_ROOT}/examples/portable/custom_ops/custom_ops.yaml
  )
  gen_operators_lib(
    LIB_NAME automatic_ops_lib KERNEL_LIBS manual_ops_1_lib_kernels DEPS
    executorch
  )
  install(
    TARGETS automatic_ops_lib
    ARCHIVE DESTINATION lib
    PUBLIC_HEADER DESTINATION include/legacy
  )
  add_executable(automatic_consumer automatic_consumer.cpp)
  target_link_libraries(automatic_consumer PRIVATE automatic_ops_lib executorch)
elseif(TEST_CASE STREQUAL "consumer")
  find_package(executorch CONFIG REQUIRED)
  include(${MANUAL_TARGETS})
  add_executable(
    consumer
    ${EXECUTORCH_ROOT}/examples/portable/custom_ops/named_manual_registration_test.cpp
  )
  target_compile_definitions(consumer PRIVATE INSTALLED_CONSUMER)
  if(NOT MSVC)
    target_compile_options(consumer PRIVATE -Wall -Wextra -Werror)
  endif()
  target_link_libraries(consumer PRIVATE manual_ops_1_lib manual_ops_2_lib)
elseif(TEST_CASE STREQUAL "invalid")
  if(OMIT_LIB_NAME)
    cmake_language(CALL ${TEST_FUNCTION} MANUAL_REGISTRATION)
  else()
    cmake_language(
      CALL ${TEST_FUNCTION} LIB_NAME "${TEST_LIB_NAME}" MANUAL_REGISTRATION
    )
  endif()
else()
  if(BINDINGS_MANUAL)
    set(_bindings_option MANUAL_REGISTRATION)
  endif()
  if(OPERATORS_MANUAL)
    set(_operators_option MANUAL_REGISTRATION)
  endif()
  if(SHARED_LIBRARY)
    list(APPEND _operators_option SHARED)
  endif()
  generate_bindings_for_kernels(LIB_NAME test_ops ${_bindings_option})
  gen_operators_lib(LIB_NAME test_ops ${_operators_option})
  if(SHARED_LIBRARY)
    get_target_property(_type test_ops TYPE)
    if(NOT _type STREQUAL "SHARED_LIBRARY")
      message(FATAL_ERROR "SHARED option did not create a shared library")
    endif()
  endif()
  if(NOT OPERATORS_MANUAL)
    get_target_property(_headers test_ops PUBLIC_HEADER)
    if(NOT
       "${_headers}"
       STREQUAL
       "${CMAKE_CURRENT_BINARY_DIR}/test_ops/Functions.h;${CMAKE_CURRENT_BINARY_DIR}/test_ops/NativeFunctions.h"
    )
      message(FATAL_ERROR "Nonmanual PUBLIC_HEADER contract changed")
    endif()
  endif()
endif()
