# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.24) # 3.24 is required for WHOLE_ARCHIVE

project(executorch_jni)

include(ExternalProject)

if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(NOT ANDROID)
  message(FATAL_ERROR "This directory is for Android build only")
endif()

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)
set(_common_compile_options -Wno-deprecated-declarations -fPIC)
set(_common_include_directories ${EXECUTORCH_ROOT}/..)
if(NOT ANDROID_PLATFORM)
  set(ANDROID_PLATFORM android-30)
endif()

# We need to download fbjni library from maven, and use its "prefab" library
# and headers, and link executorch library against that fbjni library.
# We don't know which NDK is used to compile fbjni, and we need to link our
# executorch library to the version which Android APK links against for runtime
# to ensure the libc++ dependencies are consistent.
# WARNING #
# Users need to use the SAME fbjni version here and in app gradle dependency
# for runtime compatibility!
if(NOT FBJNI_VERSION)
  set(FBJNI_VERSION 0.5.1)
endif()

set(FBJNI_AAR_URL https://repo1.maven.org/maven2/com/facebook/fbjni/fbjni/${FBJNI_VERSION}/fbjni-${FBJNI_VERSION}.aar)
set(FBJNI_DOWNLOAD_PATH ${CMAKE_CURRENT_BINARY_DIR}/third-party/fbjni/fbjni.aar)

if(NOT EXISTS "${FBJNI_DOWNLOAD_PATH}")
  file(DOWNLOAD "${FBJNI_AAR_URL}" "${FBJNI_DOWNLOAD_PATH}")
endif()

add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/third-party/fbjni/prefab/modules/fbjni/include/" "${CMAKE_CURRENT_BINARY_DIR}/third-party/fbjni/prefab/modules/fbjni/libs/android.${ANDROID_ABI}/libfbjni.so"
  COMMAND unzip -o ${FBJNI_DOWNLOAD_PATH} -d ${CMAKE_CURRENT_BINARY_DIR}/third-party/fbjni
  DEPENDS "${FBJNI_DOWNLOAD_PATH}"
)

add_custom_target(
  fbjni_prefab
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/third-party/fbjni/prefab/modules/fbjni/include/" "${CMAKE_CURRENT_BINARY_DIR}/third-party/fbjni/prefab/modules/fbjni/libs/android.${ANDROID_ABI}/libfbjni.so"
)

add_library(fbjni SHARED IMPORTED)
add_dependencies(fbjni fbjni_prefab)
set_target_properties(fbjni PROPERTIES
  IMPORTED_LOCATION "${CMAKE_CURRENT_BINARY_DIR}/third-party/fbjni/prefab/modules/fbjni/libs/android.${ANDROID_ABI}/libfbjni.so"
)

ExternalProject_Add(executorch_root
  PREFIX executorch_root
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.."
  CMAKE_ARGS
        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=TRUE
        -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake
        -DANDROID_ABI=${ANDROID_ABI}
        -DANDROID_PLATFORM=android-26
        -DBUILD_TESTING=OFF
        -DCMAKE_INSTALL_PREFIX=${CMAKE_OUT}
        -DEXECUTORCH_ENABLE_LOGGING=ON
        -DEXECUTORCH_LOG_LEVEL=Info
        -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH
        -DEXECUTORCH_BUILD_XNNPACK=ON
        -DEXECUTORCH_XNNPACK_SHARED_WORKSPACE=ON
        -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON
        -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON
        -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON
        -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON
        -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON
        -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON
        -DEXECUTORCH_BUILD_KERNELS_CUSTOM=${EXECUTORCH_BUILD_EXTENSION_LLM}
        -DEXECUTORCH_BUILD_LLAMA_JNI=${EXECUTORCH_BUILD_EXTENSION_LLM}
        -DEXECUTORCH_BUILD_NEURON=${EXECUTORCH_BUILD_NEURON}
        -DNEURON_BUFFER_ALLOCATOR_LIB=${NEURON_BUFFER_ALLOCATOR_LIB}
        -DEXECUTORCH_BUILD_QNN=${EXECUTORCH_BUILD_QNN}
        -DQNN_SDK_ROOT=${QNN_SDK_ROOT}
        -DEXECUTORCH_BUILD_VULKAN=${EXECUTORCH_BUILD_VULKAN}
)

target_link_options_shared_lib(executorch)

add_library(executorch_jni SHARED jni/jni_layer.cpp jni/log.cpp)

set(link_libraries)
list(
  APPEND
  link_libraries
  executorch
  extension_data_loader
  extension_module
  extension_runner_util
  extension_tensor
  extension_threadpool
  fbjni
)

if(TARGET optimized_native_cpu_ops_lib)
  list(
    APPEND
    link_libraries
    optimized_native_cpu_ops_lib
  )
  target_link_options_shared_lib(optimized_native_cpu_ops_lib)
else()
  list(APPEND link_libraries portable_ops_lib portable_kernels)
  target_link_options_shared_lib(portable_ops_lib)
endif()

if(TARGET quantized_kernels)
  list(APPEND link_libraries quantized_kernels quantized_ops_lib)
  target_link_options_shared_lib(quantized_ops_lib)
endif()

if(TARGET qnn_executorch_backend)
  list(APPEND link_libraries qnn_executorch_backend)
endif()

if(TARGET xnnpack_backend)
  target_link_options_shared_lib(xnnpack_backend)
  list(APPEND link_libraries xnnpack_backend XNNPACK pthreadpool cpuinfo microkernels-prod)
endif()

if(TARGET vulkan_backend)
  target_link_options_shared_lib(vulkan_backend)
  list(APPEND link_libraries vulkan_backend)
endif()

if(EXECUTORCH_BUILD_KERNELS_CUSTOM)
  list(APPEND link_libraries $<LINK_LIBRARY:WHOLE_ARCHIVE,custom_ops>)
endif()

if(TARGET pthreadpool)
  target_include_directories(
    executorch_jni
    PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/../../backends/xnnpack/third-party/cpuinfo/include
  )
  target_include_directories(
    executorch_jni
    PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/../../backends/xnnpack/third-party/pthreadpool/include
  )
endif()

if(EXECUTORCH_JNI_CUSTOM_LIBRARY)
  list(APPEND link_libraries ${EXECUTORCH_JNI_CUSTOM_LIBRARY})
  target_link_libraries(
    executorch_jni -Wl,--whole-archive ${EXECUTORCH_JNI_CUSTOM_LIBRARY}
    -Wl,--no-whole-archive
  )
endif()

target_include_directories(
  executorch_jni PRIVATE ${_common_include_directories}
  "${CMAKE_CURRENT_BINARY_DIR}/third-party/fbjni/prefab/modules/fbjni/include/"
)

target_compile_options(executorch_jni PUBLIC ${_common_compile_options})

target_link_libraries(executorch_jni ${link_libraries} log)
