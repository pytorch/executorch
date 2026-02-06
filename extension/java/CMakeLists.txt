# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.24)

# This can be built standalone or as part of the main ExecutorTorch build
if(NOT TARGET executorch)
  project(executorch_java)
  set(EXECUTORCH_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
  include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)
endif()

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Skip if building for Android (use extension/android instead)
if(ANDROID)
  message(STATUS "Skipping extension/java on Android - use extension/android instead")
  return()
endif()

set(_common_compile_options
    $<$<CXX_COMPILER_ID:MSVC>:/wd4996>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wno-deprecated-declarations -fPIC>
)

# Find JNI - provide hints for common locations
# Users can set JAVA_HOME environment variable or -DJAVA_HOME=<path>
if(DEFINED ENV{JAVA_HOME})
  set(JAVA_HOME $ENV{JAVA_HOME})
endif()

if(JAVA_HOME)
  message(STATUS "Using JAVA_HOME: ${JAVA_HOME}")
  set(JAVA_AWT_LIBRARY "${JAVA_HOME}/lib" CACHE PATH "AWT library path")
  set(JAVA_JVM_LIBRARY "${JAVA_HOME}/lib" CACHE PATH "JVM library path")
  set(JAVA_INCLUDE_PATH "${JAVA_HOME}/include" CACHE PATH "JNI include path")

  # Platform-specific include path for jni_md.h
  if(APPLE)
    set(JAVA_INCLUDE_PATH2 "${JAVA_HOME}/include/darwin" CACHE PATH "JNI platform include path")
  elseif(WIN32)
    set(JAVA_INCLUDE_PATH2 "${JAVA_HOME}/include/win32" CACHE PATH "JNI platform include path")
  else()
    set(JAVA_INCLUDE_PATH2 "${JAVA_HOME}/include/linux" CACHE PATH "JNI platform include path")
  endif()

  # Set hints for FindJNI
  set(JAVA_AWT_INCLUDE_PATH "${JAVA_HOME}/include" CACHE PATH "AWT include path")
endif()

# Find JNI - we don't actually need AWT, so make it optional
set(JNI_FIND_REQUIRED_AWT FALSE)
find_package(JNI COMPONENTS JNI)

if(NOT JNI_FOUND)
  # Try again with explicit paths
  if(JAVA_HOME)
    set(JNI_INCLUDE_DIRS "${JAVA_HOME}/include")
    if(APPLE)
      list(APPEND JNI_INCLUDE_DIRS "${JAVA_HOME}/include/darwin")
    elseif(WIN32)
      list(APPEND JNI_INCLUDE_DIRS "${JAVA_HOME}/include/win32")
    else()
      list(APPEND JNI_INCLUDE_DIRS "${JAVA_HOME}/include/linux")
    endif()
    message(STATUS "JNI include dirs set manually: ${JNI_INCLUDE_DIRS}")
  else()
    message(FATAL_ERROR
      "Could not find JNI. Please set JAVA_HOME environment variable or pass -DJAVA_HOME=<path>\n"
      "Example: cmake -DJAVA_HOME=/usr/lib/jvm/java-11-openjdk ..."
    )
  endif()
endif()

# Build fbjni from source using FetchContent
include(FetchContent)

if(NOT FBJNI_VERSION)
  set(FBJNI_VERSION 0.7.0)
endif()

FetchContent_Declare(
  fbjni
  GIT_REPOSITORY https://github.com/facebookincubator/fbjni.git
  GIT_TAG v${FBJNI_VERSION}
)

# Configure fbjni build options
set(FBJNI_BUILD_TESTS OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(fbjni)

# Use shared JNI sources from the android directory
set(JNI_SOURCES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../android/jni")

add_library(
  executorch_jni SHARED
  ${JNI_SOURCES_DIR}/jni_layer.cpp
  ${JNI_SOURCES_DIR}/log.cpp
  ${JNI_SOURCES_DIR}/jni_layer_runtime.cpp
  ${JNI_SOURCES_DIR}/jni_helper.cpp
)

set(link_libraries)
list(
  APPEND
  link_libraries
  executorch
  extension_data_loader
  extension_flat_tensor
  extension_module
  extension_runner_util
  extension_tensor
  extension_threadpool
  fbjni
)

if(EXECUTORCH_JAVA_PROFILING)
  list(APPEND link_libraries etdump flatccrt)
  target_compile_definitions(
    executorch_jni PUBLIC EXECUTORCH_JAVA_PROFILING=1
  )
endif()

if(TARGET optimized_native_cpu_ops_lib)
  list(APPEND link_libraries optimized_native_cpu_ops_lib)
  executorch_target_link_options_shared_lib(optimized_native_cpu_ops_lib)
elseif(TARGET portable_ops_lib)
  list(APPEND link_libraries portable_ops_lib portable_kernels)
  executorch_target_link_options_shared_lib(portable_ops_lib)
endif()

if(TARGET quantized_kernels)
  list(APPEND link_libraries quantized_kernels quantized_ops_lib)
  executorch_target_link_options_shared_lib(quantized_ops_lib)
endif()

if(TARGET xnnpack_backend)
  executorch_target_link_options_shared_lib(xnnpack_backend)
  list(
    APPEND
    link_libraries
    xnnpack_backend
    XNNPACK
    pthreadpool
    cpuinfo
    xnnpack-microkernels-prod
  )
  if(TARGET kleidiai)
    list(APPEND link_libraries kleidiai)
  endif()
endif()

if(EXECUTORCH_BUILD_KERNELS_LLM)
  list(APPEND link_libraries $<LINK_LIBRARY:WHOLE_ARCHIVE,custom_ops>)
endif()

if(TARGET pthreadpool)
  target_include_directories(
    executorch_jni
    PUBLIC
      ${EXECUTORCH_ROOT}/backends/xnnpack/third-party/cpuinfo/include
  )
  target_include_directories(
    executorch_jni
    PUBLIC
      ${EXECUTORCH_ROOT}/backends/xnnpack/third-party/pthreadpool/include
  )
endif()

if(EXECUTORCH_JNI_CUSTOM_LIBRARY)
  list(APPEND link_libraries ${EXECUTORCH_JNI_CUSTOM_LIBRARY})
  target_link_libraries(
    executorch_jni -Wl,--whole-archive ${EXECUTORCH_JNI_CUSTOM_LIBRARY}
    -Wl,--no-whole-archive
  )
endif()

if(EXECUTORCH_BUILD_EXTENSION_TRAINING)
  target_sources(executorch_jni PRIVATE ${JNI_SOURCES_DIR}/jni_layer_training.cpp)
  list(APPEND link_libraries extension_training)
  target_compile_definitions(
    executorch_jni PUBLIC EXECUTORCH_BUILD_EXTENSION_TRAINING=1
  )
endif()

if(EXECUTORCH_BUILD_LLAMA_JNI)
  target_sources(executorch_jni PRIVATE ${JNI_SOURCES_DIR}/jni_layer_llama.cpp)
  list(APPEND link_libraries extension_llm_runner)
  target_compile_definitions(executorch_jni PUBLIC EXECUTORCH_BUILD_LLAMA_JNI=1)
endif()

target_include_directories(
  executorch_jni
  PRIVATE
    ${_common_include_directories}
    ${JNI_INCLUDE_DIRS}
    ${fbjni_SOURCE_DIR}/cxx
)

target_compile_options(executorch_jni PUBLIC ${_common_compile_options})

target_link_libraries(executorch_jni ${link_libraries})

# Platform-specific logging library
if(APPLE)
  # No additional logging library needed on macOS
elseif(UNIX)
  # No additional logging library needed on Linux
elseif(WIN32)
  # No additional logging library needed on Windows
endif()
