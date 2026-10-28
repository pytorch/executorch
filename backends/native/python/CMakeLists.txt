# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.26)
project(executorch_native_wheel LANGUAGES C CXX)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "The Native SDK wheel currently supports Linux only")
endif()

get_filename_component(
  EXECUTORCH_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE
)
set(BUILD_SHARED_LIBS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INSTALL_LIBDIR lib)
set(EXECUTORCH_OPTIMIZE_SIZE OFF)
set(EXECUTORCH_BUILD_NATIVE ON)
set(EXECUTORCH_BUILD_NATIVE_VULKAN ON)
set(EXECUTORCH_BUILD_NATIVE_GGML ON)
set(EXECUTORCH_BUILD_NATIVE_VISION_RUNNER OFF)
set(EXECUTORCH_BUILD_SHARED OFF)
set(EXECUTORCH_BUILD_XNNPACK OFF)
set(EXECUTORCH_BUILD_VULKAN OFF)
set(EXECUTORCH_BUILD_EXECUTOR_RUNNER OFF)
set(EXECUTORCH_BUILD_PYBIND OFF)
set(EXECUTORCH_BUILD_TESTS OFF)
set(EXECUTORCH_BUILD_CPUINFO OFF)
set(EXECUTORCH_BUILD_PTHREADPOOL OFF)
set(EXECUTORCH_BUILD_PORTABLE_OPS OFF)
set(BUILD_TESTING OFF)
foreach(
  feature
  NATIVE
  AVX
  AVX2
  AVX_VNNI
  BMI2
  FMA
  F16C
  AVX512
  AVX512_VBMI
  AVX512_VNNI
  AVX512_BF16
  AMX_TILE
  AMX_INT8
  AMX_BF16
)
  set(GGML_${feature}
      OFF
      CACHE BOOL "Portable wheel CPU baseline" FORCE
  )
endforeach()

set(GGML_BACKEND_DL OFF)
set(GGML_CPU ON)
set(GGML_OPENMP OFF)

add_subdirectory("${EXECUTORCH_ROOT}" executorch EXCLUDE_FROM_ALL)

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/sdk.cpp" "")
add_library(executorch_native_sdk SHARED "${CMAKE_CURRENT_BINARY_DIR}/sdk.cpp")
target_link_libraries(
  executorch_native_sdk
  PRIVATE
    "$<LINK_LIBRARY:WHOLE_ARCHIVE,native_api,native_engine,native_runtime,native_package,native_graph,native_vulkan_engine,native_ggml_engine>"
)
target_compile_features(executorch_native_sdk INTERFACE cxx_std_20)
target_include_directories(
  executorch_native_sdk INTERFACE $<INSTALL_INTERFACE:include>
)
target_link_options(
  executorch_native_sdk PRIVATE
  "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.map"
  "LINKER:--no-undefined" "LINKER:--gc-sections"
)
set_target_properties(
  executorch_native_sdk
  PROPERTIES
    OUTPUT_NAME executorch_native
    EXPORT_NAME runtime
    INSTALL_RPATH "$ORIGIN"
    LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/exports.map"
    LINK_LIBRARY_OVERRIDE
    "WHOLE_ARCHIVE,native_api,native_engine,native_runtime,native_package,native_graph,native_vulkan_engine,native_ggml_engine"
)

add_subdirectory(../pybindings pybindings)

install(
  TARGETS executorch_native_sdk
  EXPORT ExecuTorchNativeTargets
  LIBRARY DESTINATION lib COMPONENT executorch_native
)
install(
  EXPORT ExecuTorchNativeTargets
  NAMESPACE executorch_native::
  DESTINATION lib/cmake/executorch_native
  COMPONENT executorch_native
)
install(
  FILES executorch_native-config.cmake
  DESTINATION lib/cmake/executorch_native
  COMPONENT executorch_native
)
install(
  DIRECTORY ../runtime/
  DESTINATION include/executorch/backends/native/runtime
  COMPONENT executorch_native
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "test" EXCLUDE
  PATTERN "vulkan" EXCLUDE
  PATTERN "ggml" EXCLUDE
  PATTERN "Json.h" EXCLUDE
)
install(
  FILES ../runtime/vulkan/VulkanEngine.h
  DESTINATION include/executorch/backends/native/runtime/vulkan
  COMPONENT executorch_native
)
install(
  FILES ../runtime/engine/ggml/GgmlEngine.h
  DESTINATION include/executorch/backends/native/runtime/engine/ggml
  COMPONENT executorch_native
)
install(
  DIRECTORY ../examples/vision ../examples/runner_utils
  DESTINATION share/examples
  COMPONENT executorch_native
  FILES_MATCHING
  PATTERN "CMakeLists.txt"
  PATTERN "run_classification.cpp"
  PATTERN "runner_utils.cpp"
  PATTERN "runner_utils.h"
)
install(
  FILES ../examples/runner_utils/runner_utils.h
  DESTINATION include/executorch/backends/native/examples/runner_utils
  COMPONENT executorch_native
)
install(
  FILES "${EXECUTORCH_ROOT}/LICENSE"
  DESTINATION share/licenses
  COMPONENT executorch_native
)
install(
  FILES "${EXECUTORCH_ROOT}/backends/native/third-party/llama.cpp/LICENSE"
  DESTINATION share/licenses/ggml
  COMPONENT executorch_native
)
foreach(
  license
  third-party/flatbuffers/LICENSE
  third-party/json/LICENSE.MIT
  backends/vulkan/third-party/volk/LICENSE.md
  backends/vulkan/third-party/Vulkan-Headers/LICENSE.md
  backends/vulkan/third-party/VulkanMemoryAllocator/LICENSE.txt
)
  get_filename_component(dependency "${license}" DIRECTORY)
  install(
    FILES "${EXECUTORCH_ROOT}/${license}"
    DESTINATION "share/licenses/${dependency}"
    COMPONENT executorch_native
  )
endforeach()

set(EXECUTORCH_NATIVE_EXTRA_LICENSES
    ""
    CACHE STRING "License files for external libraries bundled into the wheel"
)
foreach(license IN LISTS EXECUTORCH_NATIVE_EXTRA_LICENSES)
  get_filename_component(dependency "${license}" DIRECTORY)
  get_filename_component(dependency "${dependency}" NAME)
  install(
    FILES "${license}"
    DESTINATION "share/licenses/dependencies/${dependency}"
    COMPONENT executorch_native
  )
endforeach()
