# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.24)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
project(samsung_example)

message(EXECUTORCH_ROOT ${EXECUTORCH_ROOT})

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)
include(${EXECUTORCH_ROOT}/tools/cmake/Codegen.cmake)

if(NOT PYTHON_EXECUTABLE)
  resolve_python_executable()
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

if(CMAKE_TOOLCHAIN_FILE MATCHES ".*(iOS|ios\.toolchain)\.cmake$")
  message(FATAL_ERROR "ios is not supported by Samsung AI system.")
endif()

# prebuilt libraries. executorch package should contain portable_ops_lib,
# etdump, bundled_program.
find_package(tokenizers REQUIRED)
find_package(executorch CONFIG REQUIRED)
target_compile_options(executorch INTERFACE -DET_EVENT_TRACER_ENABLED)
find_package(gflags REQUIRED)

set(_common_compile_options -Wno-deprecated-declarations -fPIC)

#
# The `_<target>_srcs` lists are defined by executorch_load_build_variables.
#
executorch_load_build_variables()

set(EXECUTORCH_SRCS_FILE
    "${CMAKE_CURRENT_BINARY_DIR}/../../executorch_srcs.cmake"
)

get_filename_component(
  EXECUTORCH_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE
)

add_compile_options(
  -Wall
  -Werror
  -Wno-deprecated-declarations
  -Wno-unused-const-variable
  -Wno-unused-variable
  -Wno-inconsistent-missing-override
  -fPIC
)

message("Build Samsung Android Examples")

link_directories(${EXECUTORCH_ROOT}/build_samsung_android/lib)

add_library(lib_portable_ops STATIC IMPORTED)

set_target_properties(
  lib_portable_ops
  PROPERTIES
    IMPORTED_LOCATION
    "${EXECUTORCH_ROOT}/build_samsung_android/lib/libportable_ops_lib.a"
)

set(__enn_executor_runner__srcs
    ${CMAKE_CURRENT_LIST_DIR}/executor_runner/enn_executor_runner.cpp
)

add_executable(enn_executor_runner ${__enn_executor_runner__srcs})

target_include_directories(enn_executor_runner PRIVATE ${EXECUTORCH_ROOT}/..)

target_compile_options(enn_executor_runner PRIVATE ${_common_compile_options})

target_link_libraries(
  enn_executor_runner PRIVATE enn_logging enn_backend gflags executorch
                              extension_data_loader portable_ops_lib
)

set_target_properties(
  enn_executor_runner PROPERTIES CXX_VISIBILITY_PRESET hidden
)

# LLM runner library
add_subdirectory(executor_runner/llm/runner)

target_link_directories(enn_llm_runner PRIVATE)

add_library(lib_enn_llm_runner STATIC IMPORTED)
set_target_properties(
  lib_enn_llm_runner
  PROPERTIES
    IMPORTED_LOCATION
    "${EXECUTORCH_ROOT}/build_samsung_android/examples/samsung/executor_runner/llm/runner/libenn_llm_runner.a"
)

add_library(lib_tokenizer STATIC IMPORTED)
set_target_properties(
  lib_tokenizer
  PROPERTIES
    IMPORTED_LOCATION
    "${EXECUTORCH_ROOT}/build_samsung_android/extension/llm/tokenizers/libtokenizers.a"
)

add_library(lib_extension_llm_runner STATIC IMPORTED)
set_target_properties(
  lib_extension_llm_runner
  PROPERTIES
    IMPORTED_LOCATION
    "${EXECUTORCH_ROOT}/build_samsung_android/extension/llm/runner/libextension_llm_runner.a"
)

add_library(lib_enn_backend STATIC IMPORTED)
set_target_properties(
  lib_enn_backend
  PROPERTIES IMPORTED_LOCATION
             "${EXECUTORCH_ROOT}/build_samsung_android/lib/libenn_backend.a"
)

add_library(lib_executorch STATIC IMPORTED)
set_target_properties(
  lib_executorch
  PROPERTIES IMPORTED_LOCATION
             "${EXECUTORCH_ROOT}/build_samsung_android/lib/libexecutorch.a"
)

add_library(lib_custom_op STATIC IMPORTED)
set_target_properties(
  lib_custom_op
  PROPERTIES IMPORTED_LOCATION
             "${EXECUTORCH_ROOT}/build_samsung_android/lib/libcustom_ops.a"
)

set(enn_executor_llm_runner_deps
    executorch_core extension_data_loader extension_module extension_tensor
    extension_flat_tensor lib_extension_llm_runner
)

set(xnnpack_backend_libs xnnpack_backend XNNPACK xnnpack-microkernels-prod
                         extension_threadpool pthreadpool kleidiai
)

list(
  APPEND
  xnnpack_backend_libs
  optimized_native_cpu_ops_lib
  optimized_kernels
  portable_kernels
  cpublas
  eigen_blas
)

# Extra sources for cpuinfo
set(XNNPACK_ROOT ${EXECUTORCH_ROOT}/backends/xnnpack)
list(APPEND xnnpack_backend_libs cpuinfo)
list(APPEND _common_include_directories
     ${XNNPACK_ROOT}/third-party/cpuinfo/include
)

# quantized_ops_lib: Register quantized op kernels into the runtime
list(APPEND xnnpack_backend_libs quantized_kernels quantized_ops_lib)

set(exynos_llm_link_libs
    log
    gflags
    android
    lib_portable_ops
    ${enn_executor_llm_runner_deps}
    ${xnnpack_backend_libs}
    sentencepiece
    re2
    lib_tokenizer
    -Wl,--whole-archive
    lib_executorch
    lib_custom_op
    lib_enn_backend
    enn_logging
    -Wl,--no-whole-archive
)

# C API in exynos_llm_wrapper.h. Always compiled into enn_executor_llm_runner;
# additionally shipped as libexynos_llm_wrapper.so when
# EXYNOS_BUILD_LLM_WRAPPER_SO is ON.
option(EXYNOS_BUILD_LLM_WRAPPER_SO
       "Build libexynos_llm_wrapper.so and exynos_llm_wrapper_sample" OFF
)

add_library(
  exynos_llm_wrapper_obj OBJECT
  ${CMAKE_CURRENT_LIST_DIR}/executor_runner/exynos_llm_wrapper.cpp
)
target_compile_definitions(exynos_llm_wrapper_obj PRIVATE EXECUTORCH_VERSION=2)
target_link_libraries(exynos_llm_wrapper_obj PRIVATE enn_llm_runner)
set_target_properties(
  exynos_llm_wrapper_obj PROPERTIES CXX_VISIBILITY_PRESET hidden
                                    VISIBILITY_INLINES_HIDDEN ON
)

add_executable(
  enn_executor_llm_runner
  ${CMAKE_CURRENT_LIST_DIR}/executor_runner/enn_executor_llm_runner.cpp
)

target_include_directories(
  enn_executor_llm_runner
  PUBLIC ${_common_include_directories}
         ${EXECUTORCH_ROOT}/backends/samsung/runtime
         ${EXECUTORCH_ROOT}/extension/llm/tokenizers/include
)

target_link_libraries(
  enn_executor_llm_runner PRIVATE exynos_llm_wrapper_obj enn_llm_runner
                                  ${exynos_llm_link_libs}
)

if(EXYNOS_BUILD_LLM_WRAPPER_SO)
  # Bundles all dependencies so only the .so + header need to be shipped.
  add_library(
    exynos_llm_wrapper SHARED $<TARGET_OBJECTS:exynos_llm_wrapper_obj>
  )
  set_target_properties(exynos_llm_wrapper PROPERTIES LINKER_LANGUAGE CXX)
  target_include_directories(
    exynos_llm_wrapper PUBLIC ${CMAKE_CURRENT_LIST_DIR}/executor_runner
  )
  target_link_libraries(
    exynos_llm_wrapper PRIVATE enn_llm_runner ${exynos_llm_link_libs}
                               -Wl,--exclude-libs,ALL
  )

  # Sample CLI consuming the public C API the way an app uses the .so.
  add_executable(
    exynos_llm_wrapper_sample
    ${CMAKE_CURRENT_LIST_DIR}/executor_runner/exynos_llm_wrapper_sample_main.cpp
  )
  target_link_libraries(exynos_llm_wrapper_sample PRIVATE exynos_llm_wrapper)
  set_target_properties(
    exynos_llm_wrapper_sample PROPERTIES CXX_VISIBILITY_PRESET hidden
  )
endif()
