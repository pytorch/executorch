# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.19)

if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

# Ensure vulkan_schema is available even when EXECUTORCH_BUILD_VULKAN is OFF.
# The WebGPU backend reuses the Vulkan FlatBuffer serialization format.
if(NOT TARGET vulkan_schema)
  # We need the schema generation from the Vulkan backend. Build only the schema
  # target by including the Vulkan CMakeLists.txt. The full Vulkan backend will
  # only build if EXECUTORCH_BUILD_VULKAN is ON (which gates the vulkan_backend
  # target), but vulkan_schema is unconditionally defined.
  add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/../vulkan
    ${CMAKE_CURRENT_BINARY_DIR}/_vulkan_schema EXCLUDE_FROM_ALL
  )
endif()

set(WEBGPU_SRCS
    runtime/WebGPUBackend.cpp
    runtime/WebGPUGraph.cpp
    runtime/WebGPUDelegateHeader.cpp
    runtime/WebGPUDevice.cpp
    runtime/WebGPUQueryPool.cpp
    runtime/ops/OperatorRegistry.cpp
    runtime/ops/add/BinaryOp.cpp
    runtime/ops/rms_norm/RmsNorm.cpp
    runtime/ops/update_cache/UpdateCache.cpp
    runtime/ops/sdpa/Sdpa.cpp
    runtime/ops/select_as_symint/SelectAsSymint.cpp
    runtime/ops/quantized_linear/QuantizedLinear.cpp
    runtime/ops/embedding_q4gsw/EmbeddingQ4gsw.cpp
)

add_library(webgpu_backend ${WEBGPU_SRCS})

# Verify committed *_wgsl.h match their *.wgsl (drift fails the build).
resolve_python_executable()
add_custom_target(
  webgpu_wgsl_headers_check ALL
  COMMAND "${PYTHON_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/scripts/gen_wgsl_headers.py" --check
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  COMMENT "Checking WebGPU embedded-WGSL headers are in sync"
)
add_dependencies(webgpu_backend webgpu_wgsl_headers_check)

target_include_directories(
  webgpu_backend PRIVATE $<BUILD_INTERFACE:${EXECUTORCH_ROOT}/..>
)

target_link_libraries(webgpu_backend PRIVATE vulkan_schema executorch_core)

# Native WebGPU backend: Dawn (Tint) + SwiftShader; deps script sets Dawn_DIR.
# Native-only: browser/Emscripten builds use the system webgpu.h and never reach
# this find_package (root CMake gates it via EXECUTORCH_BUILD_WEBGPU).
# dawn::webgpu_dawn's link interface references Threads::Threads.
find_package(Threads REQUIRED)
find_package(Dawn REQUIRED)
set(WEBGPU_GPU_LIB dawn::webgpu_dawn)
target_link_libraries(webgpu_backend PUBLIC ${WEBGPU_GPU_LIB})

if(APPLE)
  target_link_libraries(
    webgpu_backend PRIVATE "-framework Metal" "-framework QuartzCore"
                           "-framework CoreGraphics" "-framework Foundation"
  )
else()
  target_link_libraries(webgpu_backend PRIVATE dl m pthread)
endif()

target_compile_options(webgpu_backend PRIVATE -fexceptions)

# Opt-in GPU timestamp profiling (WebGPUQueryPool); OFF so production builds
# request no TimestampQuery device feature. Mirrors Vulkan's compile-flag gate.
option(EXECUTORCH_BUILD_WEBGPU_PROFILING
       "Enable WebGPU GPU timestamp-query profiling" OFF
)
if(EXECUTORCH_BUILD_WEBGPU_PROFILING)
  target_compile_definitions(
    webgpu_backend PRIVATE WGPU_BACKEND_ENABLE_PROFILING
  )
endif()

# Link with --whole-archive for static registration of backend + ops
executorch_target_link_options_shared_lib(webgpu_backend)

set_property(TARGET webgpu_backend PROPERTY CXX_STANDARD 17)

install(
  TARGETS webgpu_backend
  EXPORT ExecuTorchTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# Native test targets. Helper mirrors backends/vulkan's vulkan_op_test: every
# test executable links the same backend + runtime libs.
function(add_webgpu_native_test test_name test_src)
  add_executable(${test_name} ${test_src})
  target_include_directories(
    ${test_name} PRIVATE $<BUILD_INTERFACE:${EXECUTORCH_ROOT}/..>
  )
  target_link_libraries(
    ${test_name}
    PRIVATE webgpu_backend
            ${WEBGPU_GPU_LIB}
            executorch_core
            extension_module_static
            extension_data_loader
            extension_tensor
            portable_kernels
            portable_ops_lib
  )
  if(APPLE)
    target_link_libraries(
      ${test_name} PRIVATE "-framework Metal" "-framework QuartzCore"
                           "-framework CoreGraphics"
    )
  else()
    target_link_libraries(${test_name} PRIVATE dl m pthread)
  endif()
  target_compile_options(${test_name} PRIVATE -fexceptions)
  if(EXECUTORCH_BUILD_WEBGPU_PROFILING)
    target_compile_definitions(
      ${test_name} PRIVATE WGPU_BACKEND_ENABLE_PROFILING
    )
  endif()
  set_property(TARGET ${test_name} PROPERTY CXX_STANDARD 17)
endfunction()

if(EXECUTORCH_BUILD_WEBGPU_TEST)
  add_webgpu_native_test(webgpu_native_test test/test_webgpu_native.cpp)
  add_webgpu_native_test(webgpu_rms_norm_test test/native/test_rms_norm.cpp)
  add_webgpu_native_test(
    webgpu_dispatch_order_test test/native/test_dispatch_order.cpp
  )
  add_webgpu_native_test(
    webgpu_scratch_buffer_test test/native/test_scratch_buffer.cpp
  )
  add_webgpu_native_test(
    webgpu_update_cache_test test/native/test_update_cache.cpp
  )
endif()
