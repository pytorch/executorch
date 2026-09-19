# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.19)

if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

include(GNUInstallDirs)
find_package(Threads REQUIRED)

set(GPU_SHARED_VULKAN_HEADERS_PATH
    ${EXECUTORCH_ROOT}/backends/vulkan/third-party/Vulkan-Headers
)

if(NOT EXISTS "${GPU_SHARED_VULKAN_HEADERS_PATH}/include/vulkan/vulkan.h")
  message(
    FATAL_ERROR
      "The shared GPU runtime requires the vendored Vulkan-Headers submodule. "
      "Run from the repository root:\n" "  git submodule update --init "
      "backends/vulkan/third-party/Vulkan-Headers"
  )
endif()

# SHARED on purpose: VGF and Vulkan may be separate delegate DSOs, but they must
# observe one process-wide registry. A static copy in each DSO would create
# independent registries and defeat context sharing.
add_library(
  executorch_gpu_shared_runtime SHARED
  ${CMAKE_CURRENT_SOURCE_DIR}/runtime/SharedGpuContext.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/runtime/SharedGpuContextRegistry.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/runtime/SharedGpuRuntimeConfig.cpp
)
add_library(executorch::gpu_shared_runtime ALIAS executorch_gpu_shared_runtime)

target_compile_definitions(
  executorch_gpu_shared_runtime PRIVATE EXECUTORCH_GPU_SHARED_BUILDING
)

target_include_directories(
  executorch_gpu_shared_runtime
  PUBLIC $<BUILD_INTERFACE:${EXECUTORCH_ROOT}/..>
         $<BUILD_INTERFACE:${EXECUTORCH_ROOT}/runtime/core/portable_type/c10>
         $<BUILD_INTERFACE:${GPU_SHARED_VULKAN_HEADERS_PATH}/include>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(
  executorch_gpu_shared_runtime
  PUBLIC executorch_core
  PRIVATE Threads::Threads
)

set_target_properties(
  executorch_gpu_shared_runtime
  PROPERTIES CXX_STANDARD 17
             CXX_STANDARD_REQUIRED YES
             CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN YES
)

install(
  TARGETS executorch_gpu_shared_runtime
  EXPORT ExecuTorchTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/runtime/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/executorch/backends/gpu_shared/runtime
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "test" EXCLUDE
)

# SharedGpuContext.h is a public installed header and includes
# <vulkan/vulkan.h>. Install the same vendored headers used by the build so an
# installed ExecuTorch package does not depend on an unrelated system Vulkan
# SDK.
install(DIRECTORY ${GPU_SHARED_VULKAN_HEADERS_PATH}/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

if(EXECUTORCH_BUILD_TESTS)
  add_subdirectory(runtime/test)
endif()
