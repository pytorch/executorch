# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.24) # 3.24 is required for WHOLE_ARCHIVE

project(executorch_jni)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(NOT ANDROID)
  message(FATAL_ERROR "This directory is for Android build only")
endif()

set(EXECUTORCH_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)
set(_common_compile_options -Wno-deprecated-declarations -fPIC)
if(NOT ANDROID_PLATFORM)
  set(ANDROID_PLATFORM android-30)
endif()

# We need to download fbjni library from maven, and use its "prefab" library and
# headers, and link executorch library against that fbjni library. We don't know
# which NDK is used to compile fbjni, and we need to link our executorch library
# to the version which Android APK links against for runtime to ensure the
# libc++ dependencies are consistent. WARNING # Users need to use the SAME fbjni
# version here and in app gradle dependency for runtime compatibility!
if(NOT FBJNI_VERSION)
  set(FBJNI_VERSION 0.5.1)
endif()

set(FBJNI_AAR_URL
    https://repo1.maven.org/maven2/com/facebook/fbjni/fbjni/${FBJNI_VERSION}/fbjni-${FBJNI_VERSION}.aar
)
set(FBJNI_DOWNLOAD_PATH ${CMAKE_CURRENT_BINARY_DIR}/third-party/fbjni/fbjni.aar)

if(NOT EXISTS "${FBJNI_DOWNLOAD_PATH}")
  file(DOWNLOAD "${FBJNI_AAR_URL}" "${FBJNI_DOWNLOAD_PATH}")
endif()

add_custom_command(
  OUTPUT
    "${CMAKE_CURRENT_BINARY_DIR}/third-party/fbjni/prefab/modules/fbjni/include/"
    "${CMAKE_CURRENT_BINARY_DIR}/third-party/fbjni/prefab/modules/fbjni/libs/android.${ANDROID_ABI}/libfbjni.so"
  COMMAND unzip -o ${FBJNI_DOWNLOAD_PATH} -d
          ${CMAKE_CURRENT_BINARY_DIR}/third-party/fbjni
  DEPENDS "${FBJNI_DOWNLOAD_PATH}"
)

add_custom_target(
  fbjni_prefab
  DEPENDS
    "${CMAKE_CURRENT_BINARY_DIR}/third-party/fbjni/prefab/modules/fbjni/include/"
    "${CMAKE_CURRENT_BINARY_DIR}/third-party/fbjni/prefab/modules/fbjni/libs/android.${ANDROID_ABI}/libfbjni.so"
)

add_library(fbjni SHARED IMPORTED)
add_dependencies(fbjni fbjni_prefab)
set_target_properties(
  fbjni
  PROPERTIES
    IMPORTED_LOCATION
    "${CMAKE_CURRENT_BINARY_DIR}/third-party/fbjni/prefab/modules/fbjni/libs/android.${ANDROID_ABI}/libfbjni.so"
)

executorch_target_link_options_shared_lib(executorch)

add_library(
  executorch_jni SHARED jni/jni_layer.cpp jni/log.cpp jni/jni_layer_runtime.cpp
                        jni/jni_helper.cpp
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

if(EXECUTORCH_ANDROID_PROFILING)
  list(APPEND link_libraries etdump flatccrt)
  target_compile_definitions(
    executorch_jni PUBLIC EXECUTORCH_ANDROID_PROFILING=1
  )
endif()

if(TARGET optimized_native_cpu_ops_lib)
  list(APPEND link_libraries optimized_native_cpu_ops_lib)
  executorch_target_link_options_shared_lib(optimized_native_cpu_ops_lib)
else()
  list(APPEND link_libraries portable_ops_lib portable_kernels)
  executorch_target_link_options_shared_lib(portable_ops_lib)
endif()

if(TARGET quantized_kernels)
  list(APPEND link_libraries quantized_kernels quantized_ops_lib)
  executorch_target_link_options_shared_lib(quantized_ops_lib)
endif()

if(TARGET qnn_executorch_backend)
  list(APPEND link_libraries qnn_executorch_backend)
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

if(TARGET vulkan_backend)
  executorch_target_link_options_shared_lib(vulkan_backend)
  list(APPEND link_libraries vulkan_backend)
endif()

if(EXECUTORCH_BUILD_KERNELS_LLM)
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

if(EXECUTORCH_BUILD_EXTENSION_TRAINING)
  target_sources(executorch_jni PRIVATE jni/jni_layer_training.cpp jni/log.cpp)
  list(APPEND link_libraries extension_training)
  target_compile_definitions(
    executorch_jni PUBLIC EXECUTORCH_BUILD_EXTENSION_TRAINING=1
  )
endif()

if(EXECUTORCH_BUILD_EXTENSION_LLM)
  target_sources(executorch_jni PRIVATE jni/jni_layer_llama.cpp jni/log.cpp)
  list(APPEND link_libraries extension_llm_runner)
  target_compile_definitions(executorch_jni PUBLIC EXECUTORCH_BUILD_EXTENSION_LLM=1)

  if(QNN_SDK_ROOT)
    target_sources(
      executorch_jni
      PRIVATE
        ${EXECUTORCH_ROOT}/examples/qualcomm/oss_scripts/llama/runner/runner.cpp
        ${EXECUTORCH_ROOT}/examples/qualcomm/oss_scripts/llama/runner/decoder_runner.cpp
        ${EXECUTORCH_ROOT}/examples/qualcomm/oss_scripts/llama/runner/prompt_processor.cpp
        ${EXECUTORCH_ROOT}/examples/qualcomm/oss_scripts/llama/runner/token_generator.cpp
        ${EXECUTORCH_ROOT}/examples/qualcomm/oss_scripts/llama/runner/lhd_token_generator.cpp
        ${EXECUTORCH_ROOT}/examples/qualcomm/oss_scripts/llama/runner/rpc_mem.cpp
        ${EXECUTORCH_ROOT}/examples/qualcomm/oss_scripts/llama/runner/kv_manager.cpp
    )

    target_include_directories(
      executorch_jni
      PRIVATE ${EXECUTORCH_ROOT}/examples/qualcomm/oss_scripts/llama/runner
    )
    target_compile_definitions(executorch_jni PRIVATE EXECUTORCH_BUILD_QNN=1)
  endif()

  if(NEURON_BUFFER_ALLOCATOR_LIB)
    target_sources(
      executorch_jni
      PRIVATE
        ${EXECUTORCH_ROOT}/examples/mediatek/executor_runner/mtk_llama_runner.cpp
        ${EXECUTORCH_ROOT}/examples/mediatek/executor_runner/llama_runner/LlamaModelChunk.cpp
        ${EXECUTORCH_ROOT}/examples/mediatek/executor_runner/llama_runner/LlamaRuntime.cpp
        ${EXECUTORCH_ROOT}/examples/mediatek/executor_runner/llama_runner/ModelChunk.cpp
        ${EXECUTORCH_ROOT}/examples/mediatek/executor_runner/llama_runner/MultiModelLoader.cpp
        ${EXECUTORCH_ROOT}/examples/mediatek/executor_runner/llama_runner/llm_helper/mask_builder.cpp
        ${EXECUTORCH_ROOT}/examples/mediatek/executor_runner/llama_runner/llm_helper/rotary_embedding.cpp
        ${EXECUTORCH_ROOT}/examples/mediatek/executor_runner/llama_runner/llm_helper/token_embedding.cpp
    )
    target_include_directories(
      executorch_jni
      PRIVATE ${EXECUTORCH_ROOT}/examples/mediatek/executor_runner/
              ${EXECUTORCH_ROOT}/examples/mediatek/executor_runner/llama_runner
    )
    add_library(libneuron_buffer_allocator SHARED IMPORTED)
    set_property(
      TARGET libneuron_buffer_allocator PROPERTY IMPORTED_LOCATION
                                                 ${NEURON_BUFFER_ALLOCATOR_LIB}
    )
    list(APPEND link_libraries neuron_backend libneuron_buffer_allocator)
    target_compile_definitions(
      executorch_jni PRIVATE EXECUTORCH_BUILD_MEDIATEK=1
    )
  endif()
endif()

if(EXECUTORCH_BUILD_EXTENSION_AUDIO)
  target_sources(executorch_jni PRIVATE jni/jni_layer_asr.cpp jni/log.cpp)
  target_compile_definitions(executorch_jni PUBLIC EXECUTORCH_BUILD_EXTENSION_AUDIO=1)

  if(QNN_SDK_ROOT)
    target_sources(
      executorch_jni
      PRIVATE
        ${EXECUTORCH_ROOT}/examples/qualcomm/oss_scripts/whisper/runner/encoder.cpp
        ${EXECUTORCH_ROOT}/examples/qualcomm/oss_scripts/whisper/runner/decoder.cpp
        ${EXECUTORCH_ROOT}/examples/qualcomm/oss_scripts/whisper/runner/runner.cpp
    )

    target_include_directories(
      executorch_jni
      PRIVATE ${EXECUTORCH_ROOT}/examples/qualcomm/oss_scripts/whisper/runner
    )
    target_compile_definitions(executorch_jni PRIVATE EXECUTORCH_BUILD_QNN=1)
  endif()
endif()

target_include_directories(
  executorch_jni
  PRIVATE
    ${_common_include_directories}
    "${CMAKE_CURRENT_BINARY_DIR}/third-party/fbjni/prefab/modules/fbjni/include/"
)

target_compile_options(executorch_jni PUBLIC ${_common_compile_options})

target_link_libraries(executorch_jni ${link_libraries} log)
