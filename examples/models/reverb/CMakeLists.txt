# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.24)
project(reverb_runner)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

set(_json_include ${EXECUTORCH_ROOT}/third-party/json/include)
set(_helper_sources runtime/reverb_audio.cpp runtime/reverb_sampler.cpp
                    runtime/reverb_text.cpp runtime/reverb_voice.cpp
)

# Pure helpers: text, sampling, audio and voice files. No ExecuTorch needed.
enable_testing()
add_executable(
  reverb_runtime_test runtime/tests/reverb_runtime_test.cpp ${_helper_sources}
)
target_include_directories(reverb_runtime_test PRIVATE runtime ${_json_include})
add_test(NAME reverb_runtime_helpers COMMAND reverb_runtime_test)

set(gflags_DIR ${CMAKE_CURRENT_BINARY_DIR}/../../../third-party/gflags)
find_package(gflags REQUIRED)

list(APPEND CMAKE_FIND_ROOT_PATH ${CMAKE_CURRENT_BINARY_DIR}/../../..)
find_package(executorch CONFIG REQUIRED FIND_ROOT_PATH_BOTH)
executorch_target_link_options_shared_lib(executorch)

set(link_libraries
    executorch
    extension_module
    extension_data_loader
    extension_tensor
    extension_flat_tensor
    tokenizers::tokenizers
    gflags
)

# Kernels for the operators no delegate takes.
if(TARGET optimized_native_cpu_ops_lib)
  list(APPEND link_libraries optimized_native_cpu_ops_lib)
  executorch_target_link_options_shared_lib(optimized_native_cpu_ops_lib)
  foreach(lib cpublas eigen_blas)
    if(TARGET ${lib})
      list(APPEND link_libraries ${lib})
    endif()
  endforeach()
else()
  list(APPEND link_libraries portable_ops_lib portable_kernels)
  executorch_target_link_options_shared_lib(portable_ops_lib)
endif()
if(TARGET quantized_ops_lib)
  list(APPEND link_libraries quantized_ops_lib)
  executorch_target_link_options_shared_lib(quantized_ops_lib)
endif()

set(_backends "")
if(TARGET mlxdelegate)
  list(APPEND link_libraries mlxdelegate mlx)
  executorch_target_link_options_shared_lib(mlxdelegate)
  list(APPEND _backends MLX)
endif()
if(TARGET xnnpack_backend)
  list(APPEND link_libraries xnnpack_backend XNNPACK xnnpack-microkernels-prod)
  if(TARGET kleidiai)
    list(APPEND link_libraries kleidiai)
  endif()
  executorch_target_link_options_shared_lib(xnnpack_backend)
  list(APPEND _backends XNNPACK)
endif()
if(ANDROID)
  list(APPEND link_libraries log)
endif()
message(STATUS "Reverb runner backends: ${_backends}")

add_executable(
  reverb_runner runtime/main.cpp runtime/reverb_runner.cpp ${_helper_sources}
)
target_include_directories(
  reverb_runner PRIVATE ${EXECUTORCH_ROOT}/.. runtime ${_json_include}
)
target_link_libraries(reverb_runner PRIVATE ${link_libraries})
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_link_options_gc_sections(reverb_runner)
  if(NOT APPLE AND NOT MSVC)
    target_link_options(reverb_runner PRIVATE "LINKER:-s")
  endif()
endif()
if(TARGET mlxdelegate)
  executorch_target_copy_mlx_metallib(reverb_runner)
endif()
install(TARGETS reverb_runner RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
