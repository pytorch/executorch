# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.24)

if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

if(EXECUTORCH_BUILD_NATIVE_CPU)
  add_library(
    cpu_runtime runtime/KernelProvider.cpp runtime/CPUPlan.cpp
                runtime/op_helpers/Convolution.cpp
  )
  target_link_libraries(
    cpu_runtime
    PUBLIC native_runtime
    PRIVATE extension_threadpool
  )
  if(EXECUTORCH_BUILD_SHARED)
    target_link_libraries(cpu_runtime PUBLIC executorch_shared)
  else()
    target_link_libraries(cpu_runtime PUBLIC executorch_core)
  endif()

  add_library(cpu_reference runtime/providers/reference/ReferenceProvider.cpp)
  target_link_libraries(cpu_reference PUBLIC cpu_runtime)

  add_library(cpu_xnnpack runtime/providers/xnnpack/XNNPACKProvider.cpp)
  target_link_libraries(
    cpu_xnnpack
    PUBLIC cpu_runtime
    PRIVATE extension_threadpool
  )
  if(EXECUTORCH_BUILD_SHARED)
    target_link_libraries(cpu_xnnpack PRIVATE xnnpack_shared)
  else()
    target_link_libraries(cpu_xnnpack PRIVATE XNNPACK)
  endif()

  foreach(target cpu_runtime cpu_reference cpu_xnnpack)
    add_library(executorch::${target} ALIAS ${target})
    target_compile_features(${target} PUBLIC cxx_std_20)
    target_compile_options(
      ${target} PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/EHsc>
                        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fexceptions>
    )
  endforeach()

  install(
    TARGETS cpu_runtime cpu_reference cpu_xnnpack
    EXPORT ExecuTorchTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
  install(
    DIRECTORY runtime/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/executorch/backends/cpu/runtime
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "test" EXCLUDE
  )
endif()

if(NOT EXECUTORCH_BUILD_CPU)
  return()
endif()

# The provider composition that cpu_backend() in targets.bzl generates for the
# default :cpu_backend target.
set(_cpu_provider_registry ${CMAKE_CURRENT_BINARY_DIR}/provider_registry.cpp)
file(
  GENERATE
  OUTPUT ${_cpu_provider_registry}
  CONTENT
    "#include <executorch/backends/cpu/runtime/KernelProvider.h>
#include <executorch/backends/cpu/runtime/providers/xnnpack/XNNPACKProvider.h>
#include <executorch/backends/cpu/runtime/providers/reference/ReferenceProvider.h>
namespace executorch::backends::cpu {
namespace {
RuntimeConfiguration runtime_configuration() {
static const ProviderFactory factories[] = {create_xnnpack_provider,create_reference_provider};
return {{factories, sizeof(factories) / sizeof(factories[0])}, {}, false};
}
const bool registered = (register_runtime_configuration(runtime_configuration), true);
}
}
"
)

list(TRANSFORM _cpu_backend__srcs PREPEND "${EXECUTORCH_ROOT}/")
if(EXECUTORCH_BUILD_SHARED)
  set(_cpu_backend_library_type SHARED)
else()
  set(_cpu_backend_library_type STATIC)
endif()
add_library(
  cpu_backend ${_cpu_backend_library_type} ${_cpu_backend__srcs}
              ${_cpu_provider_registry}
)
target_link_libraries(
  cpu_backend PRIVATE native_program extension_threadpool
                      optimized_native_cpu_ops_lib
)
if(EXECUTORCH_BUILD_SHARED)
  set_target_properties(
    cpu_backend PROPERTIES OUTPUT_NAME executorch_backend_cpu
  )
  executorch_target_soname_policy(cpu_backend)
  executorch_target_link_shared_runtime(cpu_backend)
  # The XNNPACK delegate library already bundles XNNPACK, so resolve against it
  # rather than shipping a second copy.
  target_link_libraries(
    cpu_backend
    PUBLIC executorch_shared
    PRIVATE xnnpack_backend
  )
  target_include_directories(
    cpu_backend
    PRIVATE $<TARGET_PROPERTY:XNNPACK,INTERFACE_INCLUDE_DIRECTORIES>
  )
  executorch_target_shipped_runtime_path(cpu_backend)
else()
  target_link_libraries(
    cpu_backend
    PUBLIC executorch_core
    PRIVATE XNNPACK xnnpack-microkernels-prod
  )
endif()
target_compile_features(cpu_backend PRIVATE cxx_std_20)
target_compile_options(
  cpu_backend PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/EHsc>
                      $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fexceptions>
)
set_target_properties(cpu_backend PROPERTIES POSITION_INDEPENDENT_CODE ON)
executorch_target_link_options_shared_lib(cpu_backend)

install(
  TARGETS cpu_backend
  EXPORT ExecuTorchTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
