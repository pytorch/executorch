# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

if(NOT APPLE)
  message(FATAL_ERROR "Core AI requires Apple platforms")
endif()

if(CMAKE_VERSION VERSION_LESS 3.31)
  message(FATAL_ERROR "Core AI requires CMake 3.31 or newer for Swift support")
endif()

# Inherited Clang flags and numeric definitions are not valid Swift options.
foreach(property IN ITEMS COMPILE_DEFINITIONS COMPILE_OPTIONS)
  get_directory_property(_coreai_inherited ${property})
  set_property(DIRECTORY PROPERTY ${property} "")
  foreach(value IN LISTS _coreai_inherited)
    set_property(
      DIRECTORY
      APPEND
      PROPERTY ${property} "$<$<NOT:$<COMPILE_LANGUAGE:Swift>>:${value}>"
    )
  endforeach()
endforeach()

enable_language(OBJCXX)

# The consolidated shared runtime target is created after this directory.
if(EXECUTORCH_BUILD_SHARED)
  set(_coreai_library_type SHARED)
  set(_coreai_runtime executorch_shared)
  set(_coreai_data_loader "")
else()
  set(_coreai_library_type STATIC)
  set(_coreai_runtime executorch)
  set(_coreai_data_loader extension_data_loader)
endif()

set(_coreai_runtime_sources
    runtime/coreai_backend.mm
    runtime/coreai_cache_entry.mm
    runtime/coreai_cache.mm
    runtime/coreai_cache_key.mm
    runtime/coreai_cache_root.mm
    runtime/coreai_error.mm
    runtime/coreai_fault_injection.mm
    runtime/coreai_load_coordinator.mm
    runtime/coreai_manifest.mm
    runtime/coreai_pte.cpp
    runtime/coreai_source_bundle.mm
    runtime/coreai_tensor_io.mm
    runtime/coreai_validation.mm
    runtime/coreai_storage.mm
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

if(EXECUTORCH_BUILD_TESTS AND CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  add_executable(
    coreai_host_test
    runtime/test/coreai_host_test.mm
    runtime/test/coreai_manifest_test.mm
    runtime/test/coreai_cache_root_test.mm
    runtime/test/coreai_storage_test.mm
    runtime/test/coreai_filesystem_fixture.mm
    runtime/test/coreai_cache_key_test.mm
    runtime/test/coreai_cache_entry_test.mm
    runtime/test/coreai_cache_entry_contention_test.mm
    runtime/test/coreai_source_test.mm
    runtime/test/coreai_source_integrity_test.mm
    runtime/test/coreai_source_publication_test.mm
    runtime/test/coreai_cache_entry_fixture.mm
    runtime/test/coreai_source_fixture.mm
    runtime/test/coreai_fake_loader.mm
    runtime/test/coreai_acquisition_test.mm
    runtime/test/coreai_acquisition_fixture.mm
    runtime/test/coreai_delegate_fixture.mm
    runtime/test/coreai_delegate_test.mm
    runtime/test/coreai_pte_test.mm
    runtime/test/coreai_pte_fixture.mm
    runtime/test/coreai_cache_test.mm
    runtime/test/coreai_eviction_test.mm
    runtime/ETCoreAITensor.mm
    ${_coreai_runtime_sources}
  )
  coreai_configure_objc_target(coreai_host_test)
  target_compile_definitions(coreai_host_test PRIVATE COREAI_FAULT_INJECTION=1)
  find_library(COREAI_FOUNDATION_FRAMEWORK Foundation REQUIRED)
  target_link_libraries(
    coreai_host_test
    PRIVATE ${_coreai_runtime} ${_coreai_data_loader} program_schema
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

include(${CMAKE_CURRENT_LIST_DIR}/coreai-dependencies.cmake)
executorch_coreai_dependencies()
set(_coreai_deployment 27.0)
if(CMAKE_OSX_DEPLOYMENT_TARGET VERSION_GREATER _coreai_deployment)
  set(_coreai_deployment "${CMAKE_OSX_DEPLOYMENT_TARGET}")
endif()
# CMake 3.31 does not infer an iOS target for its Swift compiler probe.
if(CMAKE_SYSTEM_NAME STREQUAL "iOS"
   AND NOT CMAKE_GENERATOR STREQUAL "Xcode"
   AND NOT CMAKE_Swift_COMPILER_TARGET
)
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
      "${_coreai_probe_arch}-apple-ios${_coreai_deployment}${_coreai_probe_environment}"
  )
endif()
enable_language(Swift)

# CMake has no per-target OSX_DEPLOYMENT_TARGET property for Ninja.
function(coreai_require_os27 target)
  set(deployment "${_coreai_deployment}")
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

function(coreai_configure_swift_target target module_name)
  set_target_properties(
    ${target} PROPERTIES Swift_MODULE_NAME "${module_name}"
                         Swift_LANGUAGE_VERSION 6
  )
  coreai_require_os27(${target})
  target_link_libraries(${target} PRIVATE executorch::coreai_dependencies)
  if(CMAKE_GENERATOR STREQUAL "Xcode")
    set_target_properties(
      ${target}
      PROPERTIES XCODE_ATTRIBUTE_PRODUCT_MODULE_NAME "${module_name}"
                 XCODE_ATTRIBUTE_SWIFT_MODULE_NAME "${module_name}"
                 XCODE_ATTRIBUTE_SWIFT_OBJC_BRIDGING_HEADER
                 "${CMAKE_CURRENT_SOURCE_DIR}/runtime/ETCoreAIBridge.h"
    )
  else()
    target_compile_options(
      ${target}
      PRIVATE
        "$<$<COMPILE_LANGUAGE:Swift>:-import-objc-header;${CMAKE_CURRENT_SOURCE_DIR}/runtime/ETCoreAIBridge.h>"
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
set(_coreai_swift_sources
    runtime/ETCoreAIError.swift runtime/ETCoreAITensors.swift
    runtime/ETCoreAIModelLoader.swift
)
add_library(coreai_swift STATIC ${_coreai_swift_sources})
coreai_configure_swift_target(coreai_swift CoreAIBridge)
set_target_properties(
  coreai_swift PROPERTIES Swift_MODULE_DIRECTORY "${_coreai_generated_dir}"
                          POSITION_INDEPENDENT_CODE ON
)
if(CMAKE_GENERATOR STREQUAL "Xcode")
  set_target_properties(
    coreai_swift PROPERTIES XCODE_ATTRIBUTE_SWIFT_OBJC_INTERFACE_HEADER_NAME
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
  target_compile_options(
    coreai_swift
    PRIVATE
      "$<$<COMPILE_LANGUAGE:Swift>:-emit-objc-header;-emit-objc-header-path;${_coreai_swift_header}>"
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
  SOURCE ${_coreai_swift_sources}
  APPEND
  PROPERTY OBJECT_DEPENDS
           "${CMAKE_CURRENT_SOURCE_DIR}/runtime/ETCoreAIBridge.h"
)

# Keep SDK entry points separate so delegate tests can substitute a fake.
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

set(_coreai_run_sdk_tests FALSE)
if(EXECUTORCH_BUILD_TESTS
   AND CMAKE_SYSTEM_NAME STREQUAL "Darwin"
   AND NOT CMAKE_CROSSCOMPILING
)
  execute_process(
    COMMAND sw_vers -productVersion
    OUTPUT_VARIABLE _coreai_host_version
    OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY
  )
  if(_coreai_host_version VERSION_GREATER_EQUAL _coreai_deployment)
    set(_coreai_run_sdk_tests TRUE)
  endif()
endif()

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
    ${_coreai_swift_sources} runtime/test/SwiftBridgeTests.swift
    runtime/ETCoreAITensor.mm
  )
  set_target_properties(
    coreai_swift_bridge_test PROPERTIES BUNDLE TRUE BUNDLE_EXTENSION xctest
  )
  coreai_configure_objc_target(coreai_swift_bridge_test)
  coreai_configure_swift_target(coreai_swift_bridge_test CoreAIBridgeTests)
  target_compile_options(
    coreai_swift_bridge_test
    PRIVATE
      "$<$<COMPILE_LANGUAGE:Swift>:-F;${_coreai_xctest_frameworks};-I;${_coreai_xctest_libs}>"
  )
  target_link_options(
    coreai_swift_bridge_test PRIVATE "-F${_coreai_xctest_frameworks}"
    "-L${_coreai_xctest_libs}" "LINKER:-rpath,${_coreai_xctest_frameworks}"
    "LINKER:-rpath,${_coreai_xctest_libs}"
  )
  target_link_libraries(
    coreai_swift_bridge_test PRIVATE "-framework XCTest" XCTestSwiftSupport
  )
  file(GLOB_RECURSE _coreai_fixture_sources CONFIGURE_DEPENDS
       "${CMAKE_CURRENT_SOURCE_DIR}/*.py"
  )
  set(_coreai_fixture_dir "${CMAKE_CURRENT_BINARY_DIR}/bridge_fixture")
  add_custom_command(
    OUTPUT "${_coreai_fixture_dir}/manifest.json"
           "${_coreai_fixture_dir}/dynamic/manifest.json"
           "${_coreai_fixture_dir}/model.pte"
    BYPRODUCTS "${_coreai_fixture_dir}/model.aimodel/main.hash"
               "${_coreai_fixture_dir}/model.aimodel/main.mlirb"
               "${_coreai_fixture_dir}/model.aimodel/metadata.json"
               "${_coreai_fixture_dir}/dynamic/model.aimodel/main.hash"
               "${_coreai_fixture_dir}/dynamic/model.aimodel/main.mlirb"
               "${_coreai_fixture_dir}/dynamic/model.aimodel/metadata.json"
    COMMAND
      "${PYTHON_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/runtime/test/export_bridge_fixture.py"
      "${_coreai_fixture_dir}"
    DEPENDS ${_coreai_fixture_sources}
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    COMMENT "Export the Core AI session test model"
    VERBATIM
  )
  add_custom_target(
    coreai_bridge_fixture
    DEPENDS "${_coreai_fixture_dir}/manifest.json"
            "${_coreai_fixture_dir}/dynamic/manifest.json"
            "${_coreai_fixture_dir}/model.pte"
  )
  add_dependencies(coreai_swift_bridge_test coreai_bridge_fixture)
  add_executable(
    coreai_pte_export_test runtime/test/coreai_pte_export_test.cpp
                           runtime/coreai_pte.cpp
  )
  coreai_configure_objc_target(coreai_pte_export_test)
  target_link_libraries(
    coreai_pte_export_test PRIVATE ${_coreai_runtime} ${_coreai_data_loader}
                                   program_schema
  )
  add_dependencies(coreai_pte_export_test coreai_bridge_fixture)
  if(_coreai_run_sdk_tests)
    add_test(NAME coreai_pte_export_test
             COMMAND coreai_pte_export_test "${_coreai_fixture_dir}/model.pte"
                     "${_coreai_fixture_dir}/manifest.json"
    )
    set_tests_properties(coreai_pte_export_test PROPERTIES TIMEOUT 30)
    add_test(NAME coreai_swift_bridge_test
             COMMAND xcrun xctest
                     "$<TARGET_BUNDLE_DIR:coreai_swift_bridge_test>"
    )
    set_tests_properties(
      coreai_swift_bridge_test
      PROPERTIES TIMEOUT 600 ENVIRONMENT
                 "COREAI_BRIDGE_FIXTURE_DIR=${_coreai_fixture_dir}"
    )
  endif()
  add_executable(
    coreai_sdk_source_lifetime_test
    runtime/test/SDKSourceLifetimeTest.swift ${_coreai_swift_sources}
    runtime/ETCoreAITensor.mm
  )
  coreai_configure_objc_target(coreai_sdk_source_lifetime_test)
  coreai_configure_swift_target(
    coreai_sdk_source_lifetime_test coreai_sdk_source_lifetime_test
  )
  target_compile_options(
    coreai_sdk_source_lifetime_test
    PRIVATE "$<$<COMPILE_LANGUAGE:Swift>:-parse-as-library>"
  )
  if(_coreai_run_sdk_tests)
    add_test(
      NAME coreai_sdk_source_lifetime_test
      COMMAND
        "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/runtime/test/run_sdk_source_lifetime.py"
        "$<TARGET_FILE:coreai_sdk_source_lifetime_test>"
    )
    set_tests_properties(coreai_sdk_source_lifetime_test PROPERTIES TIMEOUT 240)
  endif()
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
if(_coreai_run_sdk_tests)
  add_test(NAME coreai_runtime_smoke COMMAND coreai_runtime_smoke)
endif()
if(NOT EXECUTORCH_BUILD_TESTS)
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
