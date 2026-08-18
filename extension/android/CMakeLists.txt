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
set(_common_compile_options
    $<$<CXX_COMPILER_ID:MSVC>:/wd4996>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wno-deprecated-declarations -fPIC>
)
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
  set(FBJNI_VERSION 0.7.0)
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
  set(_xnnpack_so_link_libs xnnpack_backend XNNPACK pthreadpool cpuinfo
                            xnnpack-microkernels-prod
  )
  if(TARGET kleidiai)
    list(APPEND _xnnpack_so_link_libs kleidiai)
  endif()

  if(EXECUTORCH_BUILD_XNNPACK_BACKEND_SHARED)
    # Same rationale as EXECUTORCH_BUILD_VULKAN_BACKEND_SHARED below: ship
    # xnnpack_backend as its own libxnnpack_executorch_backend.so so the
    # consumer can opt out at install time. Link only the backend itself
    # (whole-archived so the register_backend static initializer fires) and the
    # XNNPACK third-party libs + executorch_core. Kernel registration archives
    # stay in libexecutorch_jni.so so loading both shared libraries together
    # does not trigger Error::RegistrationAlreadyRegistered (0x16).
    add_library(
      xnnpack_executorch_backend SHARED
      ${EXECUTORCH_ROOT}/extension/android/jni/jni_noop.cpp
    )
    executorch_target_link_options_shared_lib(xnnpack_backend)
    target_link_options(
      xnnpack_executorch_backend PRIVATE -fopenmp -static-openmp
    )
    target_link_options_gc_sections(xnnpack_executorch_backend)
    target_link_libraries(
      xnnpack_executorch_backend PRIVATE ${_xnnpack_so_link_libs}
                                         executorch_core log
    )
    set_target_properties(
      xnnpack_executorch_backend PROPERTIES LINK_FLAGS "-Wl,-rpath='$ORIGIN'"
    )
    # executorch_jni still needs cpuinfo headers / lib for thread count
    # detection.
    list(APPEND link_libraries cpuinfo)
  else()
    executorch_target_link_options_shared_lib(xnnpack_backend)
    list(APPEND link_libraries ${_xnnpack_so_link_libs})
  endif()
endif()

if(TARGET vulkan_backend)
  if(EXECUTORCH_BUILD_VULKAN_BACKEND_SHARED)
    # Build vulkan_backend as a standalone shared library that React Native
    # Executorch can ship as an opt-in artifact
    # (libvulkan_executorch_backend.so). The .so links only vulkan_backend (with
    # --whole-archive so its register_backend static initializer fires) plus
    # executorch_core/vulkan_schema. Kernel registration libs
    # (optimized_native_cpu_ops_lib, custom_ops, etc.) stay in
    # libexecutorch_jni.so to avoid duplicate Kernel registration when both .so
    # are loaded together (Error::RegistrationAlreadyRegistered, 0x16).
    add_library(
      vulkan_executorch_backend SHARED
      ${EXECUTORCH_ROOT}/extension/android/jni/jni_noop.cpp
    )
    executorch_target_link_options_shared_lib(vulkan_backend)
    target_link_options_gc_sections(vulkan_executorch_backend)
    target_link_libraries(
      vulkan_executorch_backend PRIVATE vulkan_backend vulkan_schema
                                        executorch_core log
    )
    set_target_properties(
      vulkan_executorch_backend PROPERTIES LINK_FLAGS "-Wl,-rpath='$ORIGIN'"
    )
  else()
    executorch_target_link_options_shared_lib(vulkan_backend)
    list(APPEND link_libraries vulkan_backend)
  endif()
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
  target_sources(executorch_jni PRIVATE jni/jni_layer_training.cpp)
  list(APPEND link_libraries extension_training)
  target_compile_definitions(
    executorch_jni PUBLIC EXECUTORCH_BUILD_EXTENSION_TRAINING=1
  )
endif()

if(EXECUTORCH_BUILD_LLAMA_JNI)
  target_sources(
    executorch_jni PRIVATE jni/jni_layer_llama.cpp jni/jni_layer_asr.cpp
  )
  list(APPEND link_libraries extension_llm_runner extension_asr_runner)
  target_compile_definitions(executorch_jni PUBLIC EXECUTORCH_BUILD_LLAMA_JNI=1)

  if(QNN_SDK_ROOT)
    target_sources(
      executorch_jni
      PRIVATE
        ${EXECUTORCH_ROOT}/examples/qualcomm/oss_scripts/llama/runner/runner.cpp
        ${EXECUTORCH_ROOT}/examples/qualcomm/oss_scripts/llama/runner/decoder_runner.cpp
        ${EXECUTORCH_ROOT}/examples/qualcomm/oss_scripts/llama/runner/prompt_processor.cpp
        ${EXECUTORCH_ROOT}/examples/qualcomm/oss_scripts/llama/runner/token_generator.cpp
        ${EXECUTORCH_ROOT}/examples/qualcomm/oss_scripts/llama/runner/lhd_token_generator.cpp
        ${EXECUTORCH_ROOT}/examples/qualcomm/oss_scripts/llama/runner/attention_sink_rope_runner.cpp
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

# Hook for extended build targets (e.g., internal or partner-specific JNI
# extensions). This is a no-op when extended_targets/ does not exist.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/fb/extended_targets/CMakeLists.txt")
  include("${CMAKE_CURRENT_SOURCE_DIR}/fb/extended_targets/CMakeLists.txt")
endif()

target_include_directories(
  executorch_jni
  PRIVATE
    ${_common_include_directories}
    "${CMAKE_CURRENT_BINARY_DIR}/third-party/fbjni/prefab/modules/fbjni/include/"
)

target_compile_options(executorch_jni PUBLIC ${_common_compile_options})

target_link_options(
  executorch_jni PRIVATE
  "LINKER:--version-script,${CMAKE_CURRENT_SOURCE_DIR}/jni/version_script.txt"
)
target_link_options_gc_sections(executorch_jni)

target_link_libraries(executorch_jni ${link_libraries} log)
