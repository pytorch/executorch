# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

if(NOT APPLE)
  message(FATAL_ERROR "Core AI runtime and host tests require Apple platforms")
endif()

if(CMAKE_VERSION VERSION_LESS 3.31)
  message(FATAL_ERROR "Core AI requires CMake 3.31 or newer for Swift support")
endif()

# ExecuTorch's numeric preprocessor definitions are not Swift conditions.
get_directory_property(_coreai_inherited_definitions COMPILE_DEFINITIONS)
set_property(DIRECTORY PROPERTY COMPILE_DEFINITIONS "")
foreach(definition IN LISTS _coreai_inherited_definitions)
  add_compile_definitions("$<$<NOT:$<COMPILE_LANGUAGE:Swift>>:${definition}>")
endforeach()

enable_language(OBJCXX)

set(_coreai_runtime_sources
    runtime/coreai_backend.mm
    runtime/coreai_assets.mm
    runtime/coreai_storage.mm
    runtime/coreai_bookmarks.mm
    runtime/coreai_load_coordinator.mm
    runtime/coreai_pte.cpp
    runtime/coreai_cache.mm
)

# Keep ARC and C++ exception settings away from the private Swift module.
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
    runtime/test/coreai_host_test.mm
    runtime/test/coreai_manifest_test.mm
    runtime/test/coreai_storage_test.mm
    runtime/test/coreai_filesystem_fixture.mm
    runtime/test/coreai_source_test.mm
    runtime/test/coreai_bookmark_test.mm
    runtime/test/coreai_source_fixture.mm
    runtime/test/coreai_bookmark_fixture.mm
    runtime/test/coreai_fake_loader.mm
    runtime/test/coreai_acquisition_test.mm
    runtime/test/coreai_acquisition_fixture.mm
    runtime/test/coreai_delegate_test.mm
    runtime/test/coreai_program_data_test.mm
    runtime/test/coreai_pte_fixture.mm
    runtime/test/coreai_cache_test.mm
    runtime/test/coreai_eviction_test.mm
    runtime/ETCoreAITensor.mm
    ${_coreai_runtime_sources}
  )
  coreai_configure_objc_target(coreai_host_test)
  target_compile_definitions(coreai_host_test PRIVATE COREAI_ASSETS_TESTING=1)
  find_library(COREAI_FOUNDATION_FRAMEWORK Foundation REQUIRED)
  # The consolidated runtime target is created after this directory.
  if(EXECUTORCH_BUILD_SHARED)
    set(_coreai_host_runtime executorch_shared)
    set(_coreai_host_data_loader "")
  else()
    set(_coreai_host_runtime executorch)
    set(_coreai_host_data_loader extension_data_loader)
  endif()
  target_link_libraries(
    coreai_host_test
    PRIVATE ${_coreai_host_runtime} ${_coreai_host_data_loader} program_schema
            GTest::gtest ${COREAI_FOUNDATION_FRAMEWORK}
  )
  add_test(NAME coreai_host_test COMMAND coreai_host_test)
  set_tests_properties(coreai_host_test PROPERTIES TIMEOUT 120)
endif()

if(NOT CMAKE_GENERATOR MATCHES "Ninja|Xcode")
  message(FATAL_ERROR "Core AI Swift compilation requires Ninja or Xcode")
endif()
if(NOT CMAKE_SYSTEM_NAME MATCHES "^(Darwin|iOS)$")
  message(FATAL_ERROR "Core AI currently supports macOS and iOS only")
endif()

include(${PROJECT_SOURCE_DIR}/tools/cmake/ExecuTorchCoreAIDependencies.cmake)
executorch_coreai_dependencies()
# CMake 3.31 does not infer an iOS target for its Swift compiler probe.
if(CMAKE_SYSTEM_NAME STREQUAL "iOS" AND NOT CMAKE_GENERATOR STREQUAL "Xcode")
  if(CMAKE_OSX_ARCHITECTURES)
    list(GET CMAKE_OSX_ARCHITECTURES 0 _coreai_probe_arch)
  else()
    set(_coreai_probe_arch "${CMAKE_SYSTEM_PROCESSOR}")
  endif()
  set(_coreai_probe_environment "")
  if(CMAKE_OSX_SYSROOT MATCHES "[Ss]imulator")
    set(_coreai_probe_environment "-simulator")
  endif()
  set(CMAKE_Swift_COMPILER_TARGET
      "${_coreai_probe_arch}-apple-ios27.0${_coreai_probe_environment}"
  )
endif()
enable_language(Swift)

# CMake has no per-target OSX_DEPLOYMENT_TARGET property for Ninja.
function(coreai_require_os27 target)
  set(deployment 27.0)
  if(CMAKE_OSX_DEPLOYMENT_TARGET VERSION_GREATER deployment)
    set(deployment "${CMAKE_OSX_DEPLOYMENT_TARGET}")
  endif()
  set(target_environment "")
  if(CMAKE_SYSTEM_NAME STREQUAL "iOS")
    set(platform ios)
    if(CMAKE_OSX_SYSROOT MATCHES "[Ss]imulator")
      set(minimum_flag "-mios-simulator-version-min=${deployment}")
      set(target_environment -simulator)
    else()
      set(minimum_flag "-miphoneos-version-min=${deployment}")
    endif()
  else()
    set(platform macosx)
    set(minimum_flag "-mmacosx-version-min=${deployment}")
  endif()
  set_target_properties(
    ${target}
    PROPERTIES XCODE_ATTRIBUTE_MACOSX_DEPLOYMENT_TARGET "${deployment}"
               XCODE_ATTRIBUTE_IPHONEOS_DEPLOYMENT_TARGET "${deployment}"
  )
  if(NOT CMAKE_GENERATOR STREQUAL "Xcode")
    set(arch "${CMAKE_OSX_ARCHITECTURES}")
    if(NOT arch)
      set(arch "${CMAKE_SYSTEM_PROCESSOR}")
    endif()
    list(LENGTH arch arch_count)
    if(NOT arch_count EQUAL 1)
      message(
        FATAL_ERROR
          "Core AI Ninja builds require one CMAKE_OSX_ARCHITECTURES value"
      )
    endif()
    set(triple "${arch}-apple-${platform}${deployment}${target_environment}")
    target_compile_options(
      ${target}
      PRIVATE "$<$<COMPILE_LANGUAGE:C,CXX,OBJC,OBJCXX>:${minimum_flag}>"
              "$<$<COMPILE_LANGUAGE:Swift>:-target;${triple}>"
    )
    target_link_options(
      ${target} PRIVATE "$<$<LINK_LANGUAGE:C,CXX,OBJC,OBJCXX>:${minimum_flag}>"
      "$<$<LINK_LANGUAGE:Swift>:-target;${triple}>"
    )
  endif()
endfunction()

set(_coreai_generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")
if(CMAKE_GENERATOR STREQUAL "Xcode")
  string(APPEND _coreai_generated_dir
         "/$(CONFIGURATION)$(EFFECTIVE_PLATFORM_NAME)"
  )
else()
  file(MAKE_DIRECTORY "${_coreai_generated_dir}")
endif()
set(_coreai_swift_header "${_coreai_generated_dir}/CoreAIBridge-Swift.h")
add_library(coreai_swift STATIC runtime/ETCoreAIModel.swift)
set_target_properties(
  coreai_swift
  PROPERTIES Swift_MODULE_NAME CoreAIBridge
             Swift_MODULE_DIRECTORY "${_coreai_generated_dir}"
             Swift_LANGUAGE_VERSION 6
             POSITION_INDEPENDENT_CODE ON
)
if(CMAKE_GENERATOR STREQUAL "Xcode")
  set_target_properties(
    coreai_swift
    PROPERTIES XCODE_ATTRIBUTE_PRODUCT_MODULE_NAME CoreAIBridge
               XCODE_ATTRIBUTE_SWIFT_MODULE_NAME CoreAIBridge
               XCODE_ATTRIBUTE_SWIFT_OBJC_BRIDGING_HEADER
               "${CMAKE_CURRENT_SOURCE_DIR}/runtime/ETCoreAIBridge.h"
               XCODE_ATTRIBUTE_SWIFT_OBJC_INTERFACE_HEADER_NAME
               CoreAIBridge-Swift.h
  )
  # Xcode merges architecture-specific headers into DERIVED_SOURCES_DIR.
  add_custom_command(
    TARGET coreai_swift
    POST_BUILD
    COMMAND
      ${CMAKE_COMMAND} "-DPUBLISH_ROOT=${CMAKE_CURRENT_BINARY_DIR}/generated" -P
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/PublishSwiftHeader.cmake"
    BYPRODUCTS "${_coreai_swift_header}"
    COMMENT "Publish the merged Swift Objective-C header"
    VERBATIM
  )
else()
  set(_coreai_swift_options
      -import-objc-header
      "${CMAKE_CURRENT_SOURCE_DIR}/runtime/ETCoreAIBridge.h" -emit-objc-header
      -emit-objc-header-path "${_coreai_swift_header}"
  )
  target_compile_options(
    coreai_swift
    PRIVATE "$<$<COMPILE_LANGUAGE:Swift>:${_coreai_swift_options}>"
  )
  add_custom_command(
    TARGET coreai_swift
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E true
    BYPRODUCTS "${_coreai_swift_header}"
    COMMENT "Track the generated Swift Objective-C header"
  )
endif()
# The bridge header is an input to Swift even when only its declarations change.
set_property(
  SOURCE runtime/ETCoreAIModel.swift
  APPEND
  PROPERTY OBJECT_DEPENDS
           "${CMAKE_CURRENT_SOURCE_DIR}/runtime/ETCoreAIBridge.h"
)
coreai_require_os27(coreai_swift)
target_link_libraries(coreai_swift PRIVATE executorch::coreai_dependencies)

# The Objective-C half of the bridge. Host tests replace coreai_bridge.mm with a
# fake, so these objects are only absorbed by production targets.
add_library(
  coreai_bridge_obj OBJECT runtime/ETCoreAITensor.mm runtime/coreai_bridge.mm
)
coreai_configure_objc_target(coreai_bridge_obj)
coreai_require_os27(coreai_bridge_obj)
set_target_properties(coreai_bridge_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(coreai_bridge_obj PRIVATE "${_coreai_generated_dir}")
add_dependencies(coreai_bridge_obj coreai_swift)
set_property(
  SOURCE runtime/coreai_bridge.mm
  APPEND
  PROPERTY OBJECT_DEPENDS "${_coreai_swift_header}"
)

# The Swift tests use the bridge's internal declarations, so they compile into
# one module with it and run under xctest. Ninja on macOS only.
if(EXECUTORCH_BUILD_TESTS
   AND CMAKE_SYSTEM_NAME STREQUAL "Darwin"
   AND NOT CMAKE_GENERATOR STREQUAL "Xcode"
)
  execute_process(
    COMMAND xcrun --sdk macosx --show-sdk-platform-path
    OUTPUT_VARIABLE _coreai_platform
    OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY
  )
  set(_coreai_xctest_frameworks
      "${_coreai_platform}/Developer/Library/Frameworks"
  )
  set(_coreai_xctest_libs "${_coreai_platform}/Developer/usr/lib")
  add_library(
    coreai_swift_bridge_test MODULE
    runtime/ETCoreAIModel.swift runtime/test/SwiftBridgeTests.swift
    runtime/ETCoreAITensor.mm
  )
  set_target_properties(
    coreai_swift_bridge_test
    PROPERTIES BUNDLE TRUE
               BUNDLE_EXTENSION xctest
               Swift_MODULE_NAME CoreAIBridgeTests
               Swift_LANGUAGE_VERSION 6
  )
  coreai_configure_objc_target(coreai_swift_bridge_test)
  coreai_require_os27(coreai_swift_bridge_test)
  target_compile_options(
    coreai_swift_bridge_test
    PRIVATE
      "$<$<COMPILE_LANGUAGE:Swift>:-import-objc-header;${CMAKE_CURRENT_SOURCE_DIR}/runtime/ETCoreAIBridge.h;-F;${_coreai_xctest_frameworks};-I;${_coreai_xctest_libs}>"
  )
  target_link_options(
    coreai_swift_bridge_test PRIVATE "-F${_coreai_xctest_frameworks}"
    "-L${_coreai_xctest_libs}" "LINKER:-rpath,${_coreai_xctest_frameworks}"
    "LINKER:-rpath,${_coreai_xctest_libs}"
  )
  target_link_libraries(
    coreai_swift_bridge_test PRIVATE "-framework XCTest" XCTestSwiftSupport
                                     executorch::coreai_dependencies
  )
  add_test(NAME coreai_swift_bridge_test
           COMMAND xcrun xctest "$<TARGET_BUNDLE_DIR:coreai_swift_bridge_test>"
  )
  set_tests_properties(coreai_swift_bridge_test PROPERTIES TIMEOUT 120)
endif()

if(EXECUTORCH_BUILD_SHARED)
  set(_coreai_library_type SHARED)
  set(_coreai_runtime executorch_shared)
  set(_coreai_data_loader "")
else()
  set(_coreai_library_type STATIC)
  set(_coreai_runtime executorch)
  set(_coreai_data_loader extension_data_loader)
endif()
add_library(
  coreaidelegate ${_coreai_library_type} ${_coreai_runtime_sources}
                 $<TARGET_OBJECTS:coreai_bridge_obj>
)
coreai_configure_objc_target(coreaidelegate)
# The delegate contains no Swift sources; its archive is assembled by C++.
set_target_properties(coreaidelegate PROPERTIES LINKER_LANGUAGE CXX)
coreai_require_os27(coreaidelegate)
target_include_directories(coreaidelegate PUBLIC ${_common_include_directories})
add_dependencies(coreaidelegate coreai_swift)
target_link_libraries(
  coreaidelegate PRIVATE coreai_swift ${_coreai_runtime} ${_coreai_data_loader}
                         program_schema executorch::coreai_dependencies
)
executorch_target_link_options_shared_lib(coreaidelegate)
if(EXECUTORCH_BUILD_SHARED)
  set_target_properties(
    coreaidelegate PROPERTIES OUTPUT_NAME executorch_backend_coreai
  )
  executorch_target_soname_policy(coreaidelegate)
  executorch_target_shipped_runtime_path(coreaidelegate)
endif()

# A consumer is necessary to check Swift linkage and registration retention.
add_executable(coreai_runtime_smoke runtime/test/runtime_smoke.cpp)
if(EXECUTORCH_BUILD_TESTS)
  add_test(NAME coreai_runtime_smoke COMMAND coreai_runtime_smoke)
else()
  set_target_properties(coreai_runtime_smoke PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()
coreai_require_os27(coreai_runtime_smoke)
set_target_properties(coreai_runtime_smoke PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(
  coreai_runtime_smoke PRIVATE coreaidelegate ${_coreai_runtime}
)

install(
  FILES runtime/coreai_cache.h
  DESTINATION
    ${CMAKE_INSTALL_INCLUDEDIR}/executorch/backends/apple/coreai/runtime
)

install(
  TARGETS coreaidelegate coreai_swift
  EXPORT ExecuTorchTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
