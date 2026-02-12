# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.24)
project(parakeet_runner)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

# Let files say "include <executorch/path/to/header.h>"
set(_common_include_directories ${EXECUTORCH_ROOT}/..)

# Need this for gflags
set(gflags_DIR ${CMAKE_CURRENT_BINARY_DIR}/../../../third-party/gflags)
find_package(gflags REQUIRED)

# Find executorch libraries
list(APPEND CMAKE_FIND_ROOT_PATH ${CMAKE_CURRENT_BINARY_DIR}/../../..)
find_package(executorch CONFIG REQUIRED FIND_ROOT_PATH_BOTH)
get_target_property(_executorch_imported executorch IMPORTED)
if(NOT _executorch_imported)
  executorch_target_link_options_shared_lib(executorch)
endif()

set(link_libraries executorch gflags)

# Common ops for all builds
if(TARGET optimized_native_cpu_ops_lib)
  list(APPEND link_libraries optimized_native_cpu_ops_lib cpublas eigen_blas)
  get_target_property(_is_imported optimized_native_cpu_ops_lib IMPORTED)
  if(NOT _is_imported)
    executorch_target_link_options_shared_lib(optimized_native_cpu_ops_lib)
  endif()
endif()

# CPU-only builds need quantized and custom ops
if(NOT EXECUTORCH_BUILD_CUDA)
  list(APPEND link_libraries quantized_ops_lib custom_ops)
  executorch_target_link_options_shared_lib(quantized_ops_lib)
  executorch_target_link_options_shared_lib(custom_ops)
endif()

# XNNPACK
if(TARGET xnnpack_backend)
  set(xnnpack_backend_libs xnnpack_backend XNNPACK xnnpack-microkernels-prod)
  if(TARGET kleidiai)
    list(APPEND xnnpack_backend_libs kleidiai)
  endif()
  list(APPEND link_libraries ${xnnpack_backend_libs})
  get_target_property(_xnnpack_imported xnnpack_backend IMPORTED)
  if(NOT _xnnpack_imported)
    executorch_target_link_options_shared_lib(xnnpack_backend)
  endif()
endif()

# Needed for cpuinfo where it uses android specific log lib
if(ANDROID)
  list(APPEND link_libraries log)
endif()

# Add the required ExecuTorch extensions
list(
  APPEND
  link_libraries
  extension_llm_runner
  extension_module
  extension_data_loader
  extension_tensor
  extension_flat_tensor
  tokenizers::tokenizers
)

# Link CUDA backend
if(EXECUTORCH_BUILD_CUDA)
  find_package(CUDAToolkit REQUIRED)
  list(APPEND link_libraries aoti_cuda_backend)
  if(NOT MSVC)
    executorch_target_link_options_shared_lib(aoti_cuda_backend)
  endif()
endif()

if(EXECUTORCH_BUILD_METAL)
  list(APPEND link_libraries metal_backend)
  executorch_target_link_options_shared_lib(metal_backend)
endif()

add_executable(parakeet_runner main.cpp timestamp_utils.cpp tokenizer_utils.cpp)
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_link_options_gc_sections(parakeet_runner)
  if(NOT APPLE AND NOT MSVC)
    target_link_options(parakeet_runner PRIVATE "LINKER:-s")
  endif()
endif()

target_include_directories(
  parakeet_runner PUBLIC ${_common_include_directories}
)
target_link_libraries(parakeet_runner PUBLIC ${link_libraries})
target_compile_options(parakeet_runner PUBLIC ${_common_compile_options})

# Streaming runner with microphone support
find_package(PkgConfig)
if(PkgConfig_FOUND)
  pkg_check_modules(PORTAUDIO IMPORTED_TARGET portaudio-2.0)
endif()

if(PORTAUDIO_FOUND)
  add_executable(parakeet_streaming_runner
    streaming_runner.cpp
    audio_stream.cpp
    tokenizer_utils.cpp)

  if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_link_options_gc_sections(parakeet_streaming_runner)
    if(NOT APPLE AND NOT MSVC)
      target_link_options(parakeet_streaming_runner PRIVATE "LINKER:-s")
    endif()
  endif()

  target_include_directories(
    parakeet_streaming_runner
    PUBLIC ${_common_include_directories}
  )

  target_link_libraries(
    parakeet_streaming_runner
    PUBLIC ${link_libraries}
    PUBLIC PkgConfig::PORTAUDIO
  )

  target_compile_options(parakeet_streaming_runner PUBLIC ${_common_compile_options})

  message(STATUS "Building parakeet_streaming_runner with PortAudio support")
else()
  message(WARNING "PortAudio not found. Skipping parakeet_streaming_runner build. Install with: apt-get install portaudio19-dev (Linux) or brew install portaudio (macOS)")
endif()

# Multimodal runner with camera and microphone support
find_package(OpenCV)

if(PORTAUDIO_FOUND AND OpenCV_FOUND)
  add_executable(parakeet_multimodal_runner
    multimodal_runner.cpp
    audio_stream.cpp
    video_stream.cpp
    yolo_detector.cpp
    tokenizer_utils.cpp)

  if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_link_options_gc_sections(parakeet_multimodal_runner)
    if(NOT APPLE AND NOT MSVC)
      target_link_options(parakeet_multimodal_runner PRIVATE "LINKER:-s")
    endif()
  endif()

  target_include_directories(
    parakeet_multimodal_runner
    PUBLIC ${_common_include_directories}
    PUBLIC ${OpenCV_INCLUDE_DIRS}
  )

  target_link_libraries(
    parakeet_multimodal_runner
    PUBLIC ${link_libraries}
    PUBLIC PkgConfig::PORTAUDIO
    PUBLIC ${OpenCV_LIBS}
  )

  target_compile_options(parakeet_multimodal_runner PUBLIC ${_common_compile_options})

  message(STATUS "Building parakeet_multimodal_runner with PortAudio and OpenCV support")
elseif(NOT PORTAUDIO_FOUND)
  message(WARNING "PortAudio not found. Skipping parakeet_multimodal_runner build.")
elseif(NOT OpenCV_FOUND)
  message(WARNING "OpenCV not found. Skipping parakeet_multimodal_runner build. Install with: apt-get install libopencv-dev (Linux) or brew install opencv (macOS)")
endif()

# On Windows, copy required DLLs to the executable directory
if(MSVC AND EXECUTORCH_BUILD_CUDA)
  add_custom_command(
    TARGET parakeet_runner
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:aoti_cuda_shims>
            $<TARGET_FILE_DIR:parakeet_runner>
    COMMENT "Copying aoti_cuda_shims.dll to parakeet_runner directory"
  )
endif()
