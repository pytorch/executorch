# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.24)
project(muse_glimmer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(DEFINED ENV{CONDA_PREFIX} AND EXISTS "$ENV{CONDA_PREFIX}/lib")
  list(PREPEND CMAKE_BUILD_RPATH "$ENV{CONDA_PREFIX}/lib")
endif()

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)
executorch_get_build_include_dir(
  "${EXECUTORCH_ROOT}" EXECUTORCH_SOURCE_INCLUDE_DIR
)

set(_common_include_directories ${EXECUTORCH_SOURCE_INCLUDE_DIR})
set(_json_include
    ${EXECUTORCH_ROOT}/extension/llm/tokenizers/third-party/json/single_include
)

# gflags
set(gflags_DIR ${EXECUTORCH_ROOT}/cmake-out/third-party/gflags)
find_package(gflags REQUIRED)
find_package(Threads REQUIRED)

# executorch
list(APPEND CMAKE_FIND_ROOT_PATH ${CMAKE_CURRENT_BINARY_DIR}/../../../..)
find_package(executorch CONFIG REQUIRED FIND_ROOT_PATH_BOTH)
executorch_target_link_options_shared_lib(executorch)

set(link_libraries executorch gflags)

# CPU ops (for host-side helpers not delegated to CUDA)
list(APPEND link_libraries optimized_native_cpu_ops_lib cpublas eigen_blas)
executorch_target_link_options_shared_lib(optimized_native_cpu_ops_lib)

# Extensions
list(
  APPEND
  link_libraries
  extension_llm_runner
  extension_module
  extension_data_loader
  extension_tensor
  extension_flat_tensor
)

# Backend: CUDA or MLX (exactly one required)
if(EXECUTORCH_BUILD_CUDA)
  find_package(CUDAToolkit REQUIRED)
  list(APPEND link_libraries aoti_cuda_backend)
  executorch_target_link_options_shared_lib(aoti_cuda_backend)
  add_compile_definitions(EXECUTORCH_BUILD_CUDA)
elseif(TARGET mlxdelegate)
  list(APPEND link_libraries mlxdelegate mlx)
  executorch_target_link_options_shared_lib(mlxdelegate)
  add_compile_definitions(EXECUTORCH_BUILD_MLX)
else()
  message(FATAL_ERROR "Set EXECUTORCH_BUILD_CUDA=ON or EXECUTORCH_BUILD_MLX=ON")
endif()

# Use a supplied or local stb checkout before fetching it. The resize header is
# under deprecated/ in the pinned revision.
if(EXECUTORCH_BUILD_CUDA OR TARGET mlxdelegate)
  if(NOT MUSE_GLIMMER_STB_INCLUDE_DIR
     AND EXISTS "${EXECUTORCH_ROOT}/third-party/stb/stb_image.h"
  )
    set(MUSE_GLIMMER_STB_INCLUDE_DIR "${EXECUTORCH_ROOT}/third-party/stb")
  endif()
  if(MUSE_GLIMMER_STB_INCLUDE_DIR
     AND EXISTS "${MUSE_GLIMMER_STB_INCLUDE_DIR}/stb_image.h"
  )
    message(
      STATUS "Using local stb headers from: ${MUSE_GLIMMER_STB_INCLUDE_DIR}"
    )
    list(APPEND _common_include_directories ${MUSE_GLIMMER_STB_INCLUDE_DIR}
         ${MUSE_GLIMMER_STB_INCLUDE_DIR}/deprecated
    )
  else()
    include(FetchContent)
    FetchContent_Declare(
      stb
      GIT_REPOSITORY https://github.com/nothings/stb.git
      GIT_TAG f0569113c93ad095470c54bf34a17b36646bbbb5
    )
    FetchContent_MakeAvailable(stb)
    list(APPEND _common_include_directories ${stb_SOURCE_DIR}
         ${stb_SOURCE_DIR}/deprecated
    )
  endif()
endif()

# Tokenizer (tiktoken via pytorch/tokenizers)
list(APPEND link_libraries tokenizers::tokenizers)

add_executable(
  solo_runner runtime/runners/solo.cpp runtime/engine/muse_glimmer_engine.cpp
              runtime/engine/dflash_session.cpp
)
if(EXECUTORCH_BUILD_CUDA OR TARGET mlxdelegate)
  target_sources(
    solo_runner PRIVATE runtime/engine/muse_glimmer_vision_runtime.cpp
  )
endif()
target_include_directories(
  solo_runner PUBLIC ${_common_include_directories} ${_json_include}
)
target_link_libraries(solo_runner PUBLIC ${link_libraries})

# Export symbols required by the dynamically loaded AOTI CUDA blob.
if(EXECUTORCH_BUILD_CUDA
   AND NOT APPLE
   AND NOT MSVC
)
  target_link_options(solo_runner PRIVATE "LINKER:--export-dynamic")
endif()

add_executable(
  dflash_runner
  runtime/runners/dflash.cpp runtime/engine/muse_glimmer_engine.cpp
  runtime/engine/dflash_session.cpp
)
if(EXECUTORCH_BUILD_CUDA OR TARGET mlxdelegate)
  target_sources(
    dflash_runner PRIVATE runtime/engine/muse_glimmer_vision_runtime.cpp
  )
endif()
target_include_directories(
  dflash_runner PUBLIC ${_common_include_directories} ${_json_include}
)
target_link_libraries(dflash_runner PUBLIC ${link_libraries})

add_executable(sampling_test tests/sampling_test.cpp)
target_include_directories(sampling_test PUBLIC ${_common_include_directories})
target_link_libraries(sampling_test PUBLIC extension_tensor executorch)

add_executable(dflash_timing_test tests/dflash_timing_test.cpp)
target_include_directories(
  dflash_timing_test PUBLIC ${_common_include_directories}
)

# Worker used by the OpenAI-compatible serving harness.
add_executable(
  muse_glimmer_worker
  runtime/runners/muse_glimmer_worker.cpp
  runtime/engine/muse_glimmer_engine.cpp runtime/engine/dflash_session.cpp
)
if(EXECUTORCH_BUILD_CUDA OR TARGET mlxdelegate)
  target_sources(
    muse_glimmer_worker PRIVATE runtime/engine/muse_glimmer_vision_runtime.cpp
  )
endif()
target_include_directories(
  muse_glimmer_worker PUBLIC ${_common_include_directories} ${_json_include}
)
target_link_libraries(
  muse_glimmer_worker PUBLIC ${link_libraries} Threads::Threads
)

if(EXECUTORCH_BUILD_CUDA
   AND NOT APPLE
   AND NOT MSVC
)
  target_link_options(dflash_runner PRIVATE "LINKER:--export-dynamic")
  target_link_options(muse_glimmer_worker PRIVATE "LINKER:--export-dynamic")
endif()

if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_link_options_gc_sections(solo_runner)
  target_link_options_gc_sections(dflash_runner)
  target_link_options_gc_sections(muse_glimmer_worker)
  if(NOT APPLE AND NOT MSVC)
    # CUDA AOTI blobs resolve symbols from the host executable.
    if(NOT EXECUTORCH_BUILD_CUDA)
      target_link_options(solo_runner PRIVATE "LINKER:-s")
      target_link_options(dflash_runner PRIVATE "LINKER:-s")
      target_link_options(muse_glimmer_worker PRIVATE "LINKER:-s")
    endif()
  endif()
endif()

if(TARGET mlxdelegate)
  executorch_target_copy_mlx_metallib(solo_runner)
  executorch_target_copy_mlx_metallib(dflash_runner)
  executorch_target_copy_mlx_metallib(muse_glimmer_worker)
endif()
