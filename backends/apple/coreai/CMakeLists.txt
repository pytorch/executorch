# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

if(NOT APPLE)
  message(FATAL_ERROR "Core AI host tests require Apple platforms")
endif()

enable_language(OBJCXX)

set(_coreai_runtime_sources runtime/coreai_assets.mm runtime/coreai_storage.mm)

function(coreai_configure_objc_target target)
  target_include_directories(
    ${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/runtime
  )
  set_target_properties(
    ${target} PROPERTIES OBJCXX_STANDARD 17 OBJCXX_STANDARD_REQUIRED YES
  )
  target_compile_options(
    ${target}
    PRIVATE "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc;-fno-exceptions;-fno-rtti>"
  )
endfunction()

if(EXECUTORCH_BUILD_TESTS)
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    message(FATAL_ERROR "Core AI host tests require a macOS build")
  endif()
  add_executable(
    coreai_host_test
    runtime/test/coreai_host_test.mm runtime/test/coreai_manifest_test.mm
    runtime/test/coreai_storage_test.mm
    runtime/test/coreai_filesystem_fixture.mm ${_coreai_runtime_sources}
  )
  coreai_configure_objc_target(coreai_host_test)
  target_compile_definitions(coreai_host_test PRIVATE COREAI_ASSETS_TESTING=1)
  find_library(COREAI_FOUNDATION_FRAMEWORK Foundation REQUIRED)
  target_link_libraries(
    coreai_host_test PRIVATE executorch GTest::gtest
                             ${COREAI_FOUNDATION_FRAMEWORK}
  )
  add_test(NAME coreai_host_test COMMAND coreai_host_test)
  set_tests_properties(coreai_host_test PROPERTIES TIMEOUT 120)
endif()
