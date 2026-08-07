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
    runtime/WebGPUExecutionOptions.cpp
    runtime/WebGPUGraph.cpp
    runtime/passes/SwiGLU.cpp
    runtime/passes/QkvBk64.cpp
    runtime/WebGPUDelegateHeader.cpp
    runtime/WebGPUDevice.cpp
    runtime/WebGPUQueryPool.cpp
    runtime/WebGPUShaderRegistry.cpp
    runtime/ops/OperatorRegistry.cpp
)

# Op handlers: glob so adding an op needs no CMakeLists edit. CONFIGURE_DEPENDS
# re-globs at build time when op sources are added or removed.
file(GLOB WEBGPU_OP_SRCS CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/runtime/ops/*/*.cpp"
)
list(APPEND WEBGPU_SRCS ${WEBGPU_OP_SRCS})

add_library(webgpu_backend ${WEBGPU_SRCS})

add_library(webgpu_model_loader runner/webgpu_model_loader.cpp)
target_include_directories(
  webgpu_model_loader PUBLIC $<BUILD_INTERFACE:${EXECUTORCH_ROOT}/..>
)
target_link_libraries(webgpu_model_loader PUBLIC extension_module_static)
target_compile_options(webgpu_model_loader PRIVATE -fexceptions)
target_compile_definitions(
  webgpu_model_loader PUBLIC C10_USING_CUSTOM_GENERATED_MACROS
)
set_property(TARGET webgpu_model_loader PROPERTY CXX_STANDARD 17)

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

# WASM gets its WebGPU implementation from emdawnwebgpu at executable link time.
# Native builds link Dawn (Tint) and the platform GPU libraries.
if(EMSCRIPTEN)
  target_compile_options(webgpu_backend PUBLIC "--use-port=emdawnwebgpu")
  # --use-port is also required at link time: the link step pulls in the port's
  # headers, JS glue, and libraries for consumers that link webgpu_backend.
  target_link_options(webgpu_backend PUBLIC "--use-port=emdawnwebgpu")
else()
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

if(EMSCRIPTEN)
  include(cmake/ValidateGemma4WasmNames.cmake)
  set(GEMMA4_SPEC_WASM_EXPORT_NAME
      "createGemma4Mtp"
      CACHE STRING "JavaScript factory exported by the Gemma 4 MTP WASM module"
  )
  set(GEMMA4_SPEC_WASM_OUTPUT_NAME
      "gemma4_mtp"
      CACHE STRING "Output file stem for the Gemma 4 MTP WASM module"
  )
  validate_gemma4_wasm_names(
    GEMMA4_SPEC_WASM_EXPORT_NAME GEMMA4_SPEC_WASM_OUTPUT_NAME
  )

  add_executable(
    gemma4_plain_wasm
    ${EXECUTORCH_ROOT}/examples/models/gemma4/runner/gemma4_plain_wasm.cpp
  )
  target_include_directories(
    gemma4_plain_wasm PRIVATE $<BUILD_INTERFACE:${EXECUTORCH_ROOT}/..>
  )
  target_link_libraries(
    gemma4_plain_wasm PRIVATE webgpu_backend webgpu_model_loader
                              extension_tensor
  )
  target_compile_options(
    gemma4_plain_wasm PRIVATE -fexceptions "--use-port=emdawnwebgpu"
  )
  if(EXECUTORCH_BUILD_WEBGPU_PROFILING)
    target_compile_definitions(
      gemma4_plain_wasm PRIVATE WGPU_BACKEND_ENABLE_PROFILING
    )
  endif()
  target_link_options(
    gemma4_plain_wasm
    PRIVATE
    -fexceptions
    "--use-port=emdawnwebgpu"
    -sASYNCIFY
    -sALLOW_MEMORY_GROWTH=1
    -sMAXIMUM_MEMORY=4GB
    -sFORCE_FILESYSTEM=1
    --no-entry
    "-sEXPORTED_FUNCTIONS=_et_init,_et_load,_et_unload,_et_reset,_et_prefill_batch,_et_prefill_step,_et_step,_et_profile_enable,_et_profile,_et_get_last_prefill_token_count,_et_get_route_contract_version,_et_get_last_route_mask,_et_get_last_route_conflict_count,_malloc,_free"
    "-sEXPORTED_RUNTIME_METHODS=ccall,cwrap,FS,HEAP32"
    -sSTACK_SIZE=8388608
    -sASYNCIFY_STACK_SIZE=1048576
    -sMODULARIZE=1
    -sEXPORT_NAME=createWebGPULlama
  )
  set_target_properties(
    gemma4_plain_wasm
    PROPERTIES OUTPUT_NAME "webgpu_llama"
               RUNTIME_OUTPUT_DIRECTORY
               "${CMAKE_CURRENT_BINARY_DIR}/browser_gemma4_plain"
               CXX_STANDARD 17
  )
  add_executable(
    gemma4_spec_browser
    ${EXECUTORCH_ROOT}/examples/models/gemma4/runner/gemma4_spec_runner.cpp
    ${EXECUTORCH_ROOT}/examples/models/gemma4/runner/gemma4_spec_wasm.cpp
  )
  target_include_directories(
    gemma4_spec_browser PRIVATE $<BUILD_INTERFACE:${EXECUTORCH_ROOT}/..>
  )
  target_link_libraries(
    gemma4_spec_browser PRIVATE webgpu_backend webgpu_model_loader
                                extension_tensor
  )
  target_compile_options(gemma4_spec_browser PRIVATE -fexceptions)
  if(EXECUTORCH_BUILD_WEBGPU_PROFILING)
    target_compile_definitions(
      gemma4_spec_browser PRIVATE WGPU_BACKEND_ENABLE_PROFILING
    )
  endif()
  target_link_options(
    gemma4_spec_browser
    PRIVATE
    -fexceptions
    "--use-port=emdawnwebgpu"
    "-sASYNCIFY"
    "-sALLOW_MEMORY_GROWTH=1"
    "-sMAXIMUM_MEMORY=4GB"
    "-sFORCE_FILESYSTEM=1"
    "--no-entry"
    "-sEXPORTED_FUNCTIONS=['_et_init','_et_load','_et_unload','_et_reset','_et_prefill_batch','_et_prefill_step','_et_step','_et_mtp_execute_count','_et_mtp_accepted_drafts','_et_mtp_buffered_tokens','_et_mtp_execute','_et_mtp_execution_attestation','_et_profile_enable','_et_profile','_malloc','_free']"
    "-sEXPORTED_RUNTIME_METHODS=['ccall','cwrap','FS','HEAP32']"
    "-sSTACK_SIZE=8388608"
    "-sASYNCIFY_STACK_SIZE=1048576"
    "-sMODULARIZE=1"
    "-sEXPORT_NAME=${GEMMA4_SPEC_WASM_EXPORT_NAME}"
  )
  set_target_properties(
    gemma4_spec_browser
    PROPERTIES OUTPUT_NAME "${GEMMA4_SPEC_WASM_OUTPUT_NAME}"
               RUNTIME_OUTPUT_DIRECTORY
               "${CMAKE_CURRENT_BINARY_DIR}/browser_gemma4_mtp"
               CXX_STANDARD 17
  )
endif()

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
            vulkan_schema
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
  # All WebGPU native tests use GTest (device-dependent ones bring up the device
  # in their own main(); the fold unit test is device-free via gtest_main).
  # GTest needs -DEXECUTORCH_BUILD_TESTS=ON.
  if(NOT TARGET GTest::gtest)
    find_package(GTest QUIET)
  endif()
  if(TARGET GTest::gtest)
    # Reuse add_webgpu_native_test for the backend link + frameworks + flags;
    # add only driver_util, GTest, and the header-only nlohmann/json include.
    add_webgpu_native_test(webgpu_op_test test/op_tests/op_test_driver.cpp)
    target_sources(webgpu_op_test PRIVATE test/op_tests/driver_util.cpp)
    target_link_libraries(webgpu_op_test PRIVATE GTest::gtest)
    target_include_directories(
      webgpu_op_test
      PRIVATE "${EXECUTORCH_ROOT}/third-party/json/single_include"
    )

    # Device-free util unit test. WebGPUUtils.h needs Dawn declarations, but the
    # test calls only pure helpers and does not link the Dawn implementation.
    add_executable(
      webgpu_op_test_util_test
      test/op_tests/test_driver_util.cpp test/op_tests/driver_util.cpp
      test/native/test_webgpu_utils.cpp
    )
    target_include_directories(
      webgpu_op_test_util_test
      PRIVATE $<BUILD_INTERFACE:${EXECUTORCH_ROOT}/..>
              "${EXECUTORCH_ROOT}/third-party/json/single_include"
    )
    target_link_libraries(
      webgpu_op_test_util_test PRIVATE executorch_core GTest::gtest
                                       GTest::gtest_main
    )
    target_include_directories(
      webgpu_op_test_util_test
      PRIVATE
        "$<TARGET_PROPERTY:dawn::webgpu_dawn,INTERFACE_INCLUDE_DIRECTORIES>"
    )
    target_compile_options(webgpu_op_test_util_test PRIVATE -fexceptions)
    set_property(TARGET webgpu_op_test_util_test PROPERTY CXX_STANDARD 17)

    # Device-dependent native tests: each has its own main() that brings up the
    # device once, then RUN_ALL_TESTS(); link GTest::gtest (not gtest_main).
    add_webgpu_native_test(webgpu_native_test test/test_webgpu_native.cpp)
    target_link_libraries(webgpu_native_test PRIVATE GTest::gtest)
    add_webgpu_native_test(
      webgpu_dispatch_order_test test/native/test_dispatch_order.cpp
    )
    target_link_libraries(webgpu_dispatch_order_test PRIVATE GTest::gtest)
    add_webgpu_native_test(
      webgpu_scratch_buffer_test test/native/test_scratch_buffer.cpp
    )
    target_link_libraries(webgpu_scratch_buffer_test PRIVATE GTest::gtest)
    add_webgpu_native_test(
      webgpu_update_cache_test test/native/test_update_cache.cpp
    )
    target_link_libraries(webgpu_update_cache_test PRIVATE GTest::gtest)
    target_include_directories(
      webgpu_update_cache_test
      PRIVATE "${EXECUTORCH_ROOT}/third-party/json/single_include"
    )
    add_webgpu_native_test(
      webgpu_dynamic_shape_test test/native/test_dynamic_shape.cpp
    )
    target_link_libraries(webgpu_dynamic_shape_test PRIVATE GTest::gtest)
    add_webgpu_native_test(webgpu_index_test test/native/test_index.cpp)
    target_link_libraries(webgpu_index_test PRIVATE GTest::gtest)
    add_webgpu_native_test(webgpu_topk_test test/native/test_topk.cpp)
    target_link_libraries(webgpu_topk_test PRIVATE GTest::gtest)
    add_webgpu_native_test(webgpu_scatter_test test/native/test_scatter.cpp)
    target_link_libraries(webgpu_scatter_test PRIVATE GTest::gtest)
    add_webgpu_native_test(webgpu_q4gsw_m3_test test/native/test_q4gsw_m3.cpp)
    target_link_libraries(webgpu_q4gsw_m3_test PRIVATE GTest::gtest)

    # Device-free fold unit test (gtest_main provides main; no device needed).
    add_webgpu_native_test(
      webgpu_dispatch_2d_test test/native/test_dispatch_2d.cpp
    )
    target_link_libraries(
      webgpu_dispatch_2d_test PRIVATE GTest::gtest GTest::gtest_main
    )
    add_webgpu_native_test(
      webgpu_update_cache_state_test
      test/native/test_update_cache_state.cpp
    )
    target_link_libraries(
      webgpu_update_cache_state_test PRIVATE GTest::gtest GTest::gtest_main
    )
    add_webgpu_native_test(
      webgpu_slice_chain_test test/native/test_slice_chain.cpp
    )
    target_link_libraries(
      webgpu_slice_chain_test PRIVATE GTest::gtest GTest::gtest_main
    )
    add_executable(
      webgpu_execution_options_test test/native/test_execution_options.cpp
                                    runtime/WebGPUExecutionOptions.cpp
    )
    target_include_directories(
      webgpu_execution_options_test
      PRIVATE $<BUILD_INTERFACE:${EXECUTORCH_ROOT}/..>
    )
    target_link_libraries(
      webgpu_execution_options_test PRIVATE GTest::gtest GTest::gtest_main
    )
    target_compile_options(webgpu_execution_options_test PRIVATE -fexceptions)
    set_property(TARGET webgpu_execution_options_test PROPERTY CXX_STANDARD 17)
    add_webgpu_native_test(
      webgpu_output_suppression_test test/native/test_output_suppression.cpp
    )
    target_link_libraries(webgpu_output_suppression_test PRIVATE GTest::gtest)

    add_webgpu_native_test(
      webgpu_compute_dispatch_test test/native/test_compute_dispatch.cpp
    )
    target_link_libraries(webgpu_compute_dispatch_test PRIVATE GTest::gtest)

    add_executable(webgpu_model_loader_test test/native/test_model_loader.cpp)
    target_include_directories(
      webgpu_model_loader_test PRIVATE $<BUILD_INTERFACE:${EXECUTORCH_ROOT}/..>
    )
    target_link_libraries(
      webgpu_model_loader_test PRIVATE webgpu_model_loader GTest::gtest
                                       GTest::gtest_main
    )
    target_compile_options(webgpu_model_loader_test PRIVATE -fexceptions)
    set_property(TARGET webgpu_model_loader_test PROPERTY CXX_STANDARD 17)
  endif()
endif()
