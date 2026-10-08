# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.24)
project(manual_registration_test LANGUAGES CXX)
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
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
elseif(TEST_CASE STREQUAL "empty_installed_consumer")
  include(${MANUAL_TARGETS})
  add_executable(empty_shared_consumer empty_shared_consumer.cpp)
  target_include_directories(
    empty_shared_consumer PRIVATE ${SOURCE_INCLUDE_ROOT} ${TORCH_INCLUDE_ROOT}
  )
  target_link_libraries(empty_shared_consumer PRIVATE empty_ops)
elseif(TEST_CASE STREQUAL "empty_shared")
  gen_selected_ops(LIB_NAME empty_ops)
  generate_bindings_for_kernels(
    LIB_NAME empty_ops MANUAL_REGISTRATION REGISTRATION_NAME empty
  )
  gen_operators_lib(
    LIB_NAME empty_ops MANUAL_REGISTRATION SHARED
  )
  target_sources(empty_ops PRIVATE empty_runtime_stub.cpp)
  target_include_directories(empty_ops PRIVATE ${SOURCE_INCLUDE_ROOT} ${TORCH_INCLUDE_ROOT})
  target_compile_definitions(empty_ops PRIVATE ET_LOG_ENABLED=0)
  set_target_properties(empty_ops PROPERTIES CXX_VISIBILITY_PRESET hidden)
  add_executable(empty_shared_consumer empty_shared_consumer.cpp)
  target_include_directories(
    empty_shared_consumer PRIVATE ${SOURCE_INCLUDE_ROOT} ${TORCH_INCLUDE_ROOT}
  )
  target_link_libraries(empty_shared_consumer PRIVATE empty_ops)
  install(
    TARGETS empty_ops
    EXPORT EmptyRegistrationTargets
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    FILE_SET HEADERS DESTINATION include
  )
  install(EXPORT EmptyRegistrationTargets DESTINATION lib/cmake/EmptyRegistration)
elseif(TEST_CASE STREQUAL "invalid")
  if(TEST_REGISTRATION_NAME)
    list(APPEND _invalid_options REGISTRATION_NAME "${TEST_REGISTRATION_NAME}")
  endif()
  if(OMIT_LIB_NAME)
    cmake_language(CALL ${TEST_FUNCTION} MANUAL_REGISTRATION ${_invalid_options})
  else()
    cmake_language(
      CALL ${TEST_FUNCTION} LIB_NAME "${TEST_LIB_NAME}" MANUAL_REGISTRATION
      ${_invalid_options}
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
  if(TEST_LIB_NAME)
    set(_lib_name "${TEST_LIB_NAME}")
  else()
    set(_lib_name test_ops)
  endif()
  if(TEST_REGISTRATION_NAME)
    set(_registration_name_option REGISTRATION_NAME "${TEST_REGISTRATION_NAME}")
  endif()
  generate_bindings_for_kernels(
    LIB_NAME "${_lib_name}" ${_bindings_option} ${_registration_name_option}
  )
  gen_operators_lib(LIB_NAME "${_lib_name}" ${_operators_option})
  if(SHARED_LIBRARY)
    get_target_property(_type ${_lib_name} TYPE)
    if(NOT _type STREQUAL "SHARED_LIBRARY")
      message(FATAL_ERROR "SHARED option did not create a shared library")
    endif()
  endif()
  if(NOT OPERATORS_MANUAL)
    get_target_property(_headers ${_lib_name} PUBLIC_HEADER)
    if(NOT
       "${_headers}"
       STREQUAL
       "${CMAKE_CURRENT_BINARY_DIR}/test_ops/Functions.h;${CMAKE_CURRENT_BINARY_DIR}/test_ops/NativeFunctions.h"
    )
      message(FATAL_ERROR "Nonmanual PUBLIC_HEADER contract changed")
    endif()
  endif()
endif()
