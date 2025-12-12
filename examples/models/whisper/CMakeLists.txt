# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.29)
project(whisper_runner)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EXECUTORCH_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

# Let files say "include <executorch/path/to/header.h>"
set(_common_include_directories ${EXECUTORCH_ROOT}/..)

# Need this for gflags for some reason
set(gflags_DIR ${CMAKE_CURRENT_BINARY_DIR}/../../../third-party/gflags)
find_package(gflags REQUIRED)

list(APPEND CMAKE_FIND_ROOT_PATH ${CMAKE_CURRENT_BINARY_DIR}/../../..)
find_package(executorch CONFIG REQUIRED FIND_ROOT_PATH_BOTH)

set(_link_libraries executorch gflags)
set(_srcs multimodal.cpp)

list(
  APPEND
  _link_libraries
  optimized_native_cpu_ops_lib
  quantized_ops_lib
  custom_ops
  cpublas
  eigen_blas
)

# XNNPACK
if(TARGET xnnpack_backend)
  list(APPEND _link_libraries xnnpack_backend)
endif()

# Add LLM runner and extension module
if(NOT TARGET extension_asr_runner)
  message(
    FATAL_ERROR
      "ExecuTorch must be installed with EXECUTORCH_BUILD_EXTENSION_ASR_RUNNER enabled."
  )
endif()

# Needed for cpuinfo where it uses android specific log lib
if(ANDROID)
  list(APPEND _link_libraries log)
endif()

# Add the required ExecuTorch extensions for multimodal LLM runner
list(
  APPEND
  _link_libraries
  extension_asr_runner
  extension_llm_runner # Needed for load_tokenizer()
  extension_module
  extension_data_loader
  extension_tensor
  extension_flat_tensor
)

# Link CUDA backend
if(EXECUTORCH_BUILD_CUDA)
  find_package(CUDAToolkit REQUIRED)
  list(APPEND _link_libraries aoti_cuda_backend)
  executorch_target_link_options_shared_lib(aoti_cuda_backend)
endif()

if(EXECUTORCH_BUILD_METAL)
  list(APPEND _link_libraries metal_backend)
  executorch_target_link_options_shared_lib(metal_backend)
endif()

# Add tokenizers
list(APPEND _link_libraries tokenizers::tokenizers)

add_executable(whisper_runner main.cpp)

target_include_directories(whisper_runner PUBLIC ${_common_include_directories})

target_link_libraries(whisper_runner PUBLIC ${_link_libraries})
target_compile_options(whisper_runner PUBLIC ${_common_compile_options})

add_executable(
  whisper_torch_runner
  torch_main.cpp
  torch_whisper_runner.cpp
  torch_aoti_model.cpp)

target_include_directories(
  whisper_torch_runner PUBLIC ${_common_include_directories})
target_link_libraries(whisper_torch_runner PUBLIC ${_link_libraries})
target_compile_options(
  whisper_torch_runner PUBLIC ${_common_compile_options})
if(UNIX AND NOT APPLE)
  target_link_libraries(whisper_torch_runner PUBLIC dl)
endif()
