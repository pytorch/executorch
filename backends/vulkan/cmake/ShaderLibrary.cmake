# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# ### Editing this file ###
#
# This file should be formatted with
# ~~~
# cmake-format --first-comment-is-literal=True -i ATenVulkan.cmake
# ~~~
# It should also be cmake-lint clean.
#
# The targets in this file will be built if EXECUTORCH_BUILD_VULKAN is ON

if(ANDROID)
  if(NOT ANDROID_NDK)
    message(FATAL_ERROR "ANDROID_NDK not set")
  endif()

  set(GLSLC_PATH
      "${ANDROID_NDK}/shader-tools/${ANDROID_NDK_HOST_SYSTEM_NAME}/glslc")
else()
  find_program(
    GLSLC_PATH glslc
    PATHS ENV VULKAN_SDK "$ENV{VULKAN_SDK}/${CMAKE_HOST_SYSTEM_PROCESSOR}/bin"
          "$ENV{VULKAN_SDK}/bin")

  if(NOT GLSLC_PATH)
    message(FATAL_ERROR "USE_VULKAN glslc not found")
  endif()
endif()

# Required to enable linking with --whole-archive
include(${EXECUTORCH_ROOT}/build/Utils.cmake)

# Convenience macro to generate a SPIR-V shader library target. Given the path
# to the shaders to compile and the name of the library, it will create a static
# library containing the generated SPIR-V shaders. The generated_spv_cpp
# variable can be used to reference the generated CPP file outside the macro.
macro(VULKAN_SHADER_LIBRARY shaders_path library_name)
  set(VULKAN_SHADERGEN_ENV "")
  set(VULKAN_SHADERGEN_OUT_PATH ${CMAKE_BINARY_DIR}/${library_name})

  execute_process(
    COMMAND
      "${PYTHON_EXECUTABLE}"
      ${CMAKE_CURRENT_SOURCE_DIR}/runtime/api/gen_vulkan_spv.py --glsl-path
      ${shaders_path} --output-path ${VULKAN_SHADERGEN_OUT_PATH}
      --glslc-path=${GLSLC_PATH} --tmp-dir-path=${VULKAN_SHADERGEN_OUT_PATH}
      --env ${VULKAN_GEN_ARG_ENV}
    RESULT_VARIABLE error_code)
  set(ENV{PYTHONPATH} ${PYTHONPATH})

  set(generated_spv_cpp ${VULKAN_SHADERGEN_OUT_PATH}/spv.cpp)

  add_library(${library_name} STATIC ${generated_spv_cpp})
  target_include_directories(${library_name} PRIVATE ${COMMON_INCLUDES})
  target_link_libraries(${library_name} vulkan_graph_lib)
  target_compile_options(${library_name} PRIVATE ${VULKAN_CXX_FLAGS})
  # Link this library with --whole-archive due to dynamic shader registrations
  target_link_options_shared_lib(${library_name})

  unset(VULKAN_SHADERGEN_ENV)
  unset(VULKAN_SHADERGEN_OUT_PATH)
endmacro()
