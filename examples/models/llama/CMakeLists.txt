# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Simple CMake build system for selective build demo.
#
# ### Editing this file ###
#
# This file should be formatted with
# ~~~
# cmake-format -i CMakeLists.txt
# ~~~
# It should also be cmake-lint clean.
#
cmake_minimum_required(VERSION 3.24)

# When included as part of the root ExecuTorch build (EXCLUDE_FROM_ALL),
# required targets like extension_llm_runner are already defined. If they're
# missing, the user configured without the right options. Provide a clear
# build-time error rather than a confusing "target not found".
if(NOT TARGET extension_llm_runner)
  file(
    WRITE ${CMAKE_CURRENT_BINARY_DIR}/_llama_missing_deps.cmake
    "message(FATAL_ERROR \"llama_main requires EXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=ON (enabled by the 'llm' preset)\")\n"
  )
  add_custom_target(
    llama_main COMMAND ${CMAKE_COMMAND} -P
                       ${CMAKE_CURRENT_BINARY_DIR}/_llama_missing_deps.cmake
  )
  return()
endif()

option(EXECUTORCH_BUILD_KERNELS_TORCHAO_MPS "Build the torchao MPS kernels" OFF)

#
# llama_main: test binary to run llama, with tokenizer and sampler integrated
#

# llama_runner library
add_subdirectory(runner)

set(link_libraries executorch gflags)
set(_srcs main.cpp)

# Whole-archive linking for libraries with load-time constructors is handled
# automatically: executorch_target_link_options_shared_lib() (called during
# target creation in the root build) bakes -force_load / --whole-archive into
# each target's INTERFACE_LINK_OPTIONS, so simply listing them here is enough.

if(TARGET optimized_native_cpu_ops_lib)
  list(
    APPEND
    link_libraries
    optimized_native_cpu_ops_lib
    optimized_kernels
    portable_kernels
    cpublas
    eigen_blas
  )
else()
  list(APPEND link_libraries portable_ops_lib portable_kernels)
endif()

# quantized_ops_lib: Register quantized op kernels into the runtime
if(TARGET quantized_ops_lib)
  list(APPEND link_libraries quantized_kernels quantized_ops_lib)
endif()

if(TARGET custom_ops)
  list(APPEND link_libraries custom_ops)
endif()

if(TARGET torchao_ops_executorch)
  list(APPEND link_libraries torchao_ops_executorch)
endif()

if(EXECUTORCH_BUILD_KERNELS_TORCHAO_MPS)
  # Currently only enable this on Arm-based Macs
  if(CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND CMAKE_SYSTEM_PROCESSOR STREQUAL
                                             "arm64"
  )
    if(EXECUTORCH_BUILD_MPS)
      add_subdirectory(
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../third-party/ao/torchao/experimental/ops/mps
        ${CMAKE_CURRENT_BINARY_DIR}/../../../third-party/ao/torchao/experimental/ops/mps
      )
      list(APPEND link_libraries torchao_ops_mps_executorch)
    endif()
  endif()
endif()

# Extra compile option and include dir for pthreadpool
if(EXECUTORCH_BUILD_PTHREADPOOL)
  list(APPEND link_libraries extension_threadpool pthreadpool)
endif()

# Extra sources for cpuinfo
if(EXECUTORCH_BUILD_CPUINFO)
  list(APPEND link_libraries cpuinfo)
endif()

# XNNPACK
if(TARGET xnnpack_backend)
  list(APPEND link_libraries xnnpack_backend XNNPACK xnnpack-microkernels-prod)
  if(TARGET kleidiai)
    list(APPEND link_libraries kleidiai)
  endif()
endif()

# CUDA backend
if(EXECUTORCH_BUILD_CUDA)
  list(APPEND link_libraries aoti_cuda_backend)
endif()

# Vulkan backend
if(TARGET vulkan_backend)
  list(APPEND link_libraries vulkan_backend)
endif()

# Qnn backend
if(TARGET qnn_executorch_backend)
  list(APPEND link_libraries qnn_executorch_backend)
endif()

# MPS backend
if(TARGET mpsdelegate)
  list(
    APPEND
    link_libraries
    mpsdelegate
    "-framework Foundation"
    "-weak_framework MetalPerformanceShaders"
    "-weak_framework MetalPerformanceShadersGraph"
    "-weak_framework Metal"
  )
endif()

# MLX backend
if(TARGET mlxdelegate)
  list(APPEND link_libraries mlxdelegate mlx)
endif()

# Openvino backend
if(TARGET openvino_backend)
  list(APPEND link_libraries openvino_backend)
endif()

if(TARGET coremldelegate)
  list(
    APPEND
    link_libraries
    coremldelegate
    sqlite3
    "-framework Foundation"
    "-framework CoreML"
    "-framework Accelerate"
  )
endif()

# This one is needed for cpuinfo where it uses android specific log lib
if(ANDROID)
  list(APPEND link_libraries log)
endif()

add_executable(llama_main ${_srcs})

# Copy MLX metallib for runtime if MLX delegate is enabled
if(TARGET mlxdelegate)
  executorch_target_copy_mlx_metallib(llama_main)
endif()

# Only strip symbols for Release and MinSizeRel builds.
if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL
                                          "MinSizeRel"
)
  target_link_options_gc_sections(llama_main)
  if(NOT APPLE)
    target_link_options(llama_main PRIVATE "LINKER:-s")
  endif()
endif()

target_include_directories(llama_main PUBLIC ${_common_include_directories})
target_link_libraries(llama_main PUBLIC llama_runner ${link_libraries})
target_compile_options(llama_main PUBLIC ${_common_compile_options})
if(APPLE)
  target_link_options(llama_main PRIVATE -Wl,-rpath,@loader_path)
elseif(UNIX)
  set_target_properties(llama_main PROPERTIES LINK_FLAGS "-Wl,-rpath='$ORIGIN'")
endif()
# Windows doesn't need rpath - DLLs are found via standard Windows search order
