# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# ### Editing this file ###
#
# This file should be formatted with
# ~~~
# cmake-format -i ATenVulkan.cmake
# ~~~
# It should also be cmake-lint clean.
#
# The targets in this file will be built if EXECUTORCH_BUILD_VULKAN is ON

if(NOT PYTHON_EXECUTABLE)
  message(
    "WARNING: PYTHON_EXECUTABLE is not set! A failure is likely imminent."
  )
endif()

if(NOT EXECUTORCH_ROOT)
  message("WARNING: EXECUTORCH_ROOT is not set! A failure is likely imminent.")
endif()

if(ANDROID)
  if(NOT ANDROID_NDK)
    message(FATAL_ERROR "ANDROID_NDK not set")
  endif()

  set(GLSLC_PATH
      "${ANDROID_NDK}/shader-tools/${ANDROID_NDK_HOST_SYSTEM_NAME}/glslc"
  )
else()
  find_program(GLSLC_PATH glslc PATHS $ENV{PATH})

  if(NOT GLSLC_PATH)
    message(FATAL_ERROR "USE_VULKAN glslc not found")
  endif()
endif()

# Required to enable linking with --whole-archive
include(${EXECUTORCH_ROOT}/build/Utils.cmake)

function(gen_vulkan_shader_lib_cpp shaders_path)
  set(VULKAN_SHADERGEN_ENV "")
  set(VULKAN_SHADERGEN_OUT_PATH ${CMAKE_BINARY_DIR}/${ARGV1})

  execute_process(
    COMMAND
      "${PYTHON_EXECUTABLE}"
      ${EXECUTORCH_ROOT}/backends/vulkan/runtime/gen_vulkan_spv.py --glsl-path
      ${shaders_path} --output-path ${VULKAN_SHADERGEN_OUT_PATH}
      --glslc-path=${GLSLC_PATH} --tmp-dir-path=${VULKAN_SHADERGEN_OUT_PATH}
      --env ${VULKAN_GEN_ARG_ENV}
    RESULT_VARIABLE error_code
  )

  set(generated_spv_cpp
      ${VULKAN_SHADERGEN_OUT_PATH}/spv.cpp
      PARENT_SCOPE
  )
endfunction()

function(vulkan_shader_lib library_name generated_spv_cpp)
  add_library(${library_name} STATIC ${generated_spv_cpp})
  target_include_directories(
    ${library_name}
    PRIVATE
      ${EXECUTORCH_ROOT}/..
      ${EXECUTORCH_ROOT}/backends/vulkan/third-party/Vulkan-Headers/include
      ${EXECUTORCH_ROOT}/backends/vulkan/third-party/volk
  )
  target_link_libraries(${library_name} vulkan_backend)
  target_compile_options(${library_name} PRIVATE ${VULKAN_CXX_FLAGS})
  # Link this library with --whole-archive due to dynamic shader registrations
  target_link_options_shared_lib(${library_name})
endfunction()

# Convenience macro to generate a SPIR-V shader library target. Given the path
# to the shaders to compile and the name of the library, it will create a static
# library containing the generated SPIR-V shaders. The generated_spv_cpp
# variable can be used to reference the generated CPP file outside the macro.
macro(vulkan_shader_library shaders_path library_name)
  set(VULKAN_SHADERGEN_ENV "")
  set(VULKAN_SHADERGEN_OUT_PATH ${CMAKE_BINARY_DIR}/${library_name})

  # execute_process( COMMAND "${PYTHON_EXECUTABLE}"
  # ${EXECUTORCH_ROOT}/backends/vulkan/runtime/gen_vulkan_spv.py --glsl-path
  # ${shaders_path} --output-path ${VULKAN_SHADERGEN_OUT_PATH}
  # --glslc-path=${GLSLC_PATH} --tmp-dir-path=${VULKAN_SHADERGEN_OUT_PATH} --env
  # ${VULKAN_GEN_ARG_ENV} RESULT_VARIABLE error_code ) set(ENV{PYTHONPATH}
  # ${PYTHONPATH})

  set(generated_spv_cpp ${VULKAN_SHADERGEN_OUT_PATH}/spv.cpp)

  add_library(${library_name} STATIC ${generated_spv_cpp})
  target_include_directories(
    ${library_name}
    PRIVATE
      ${EXECUTORCH_ROOT}/..
      ${EXECUTORCH_ROOT}/backends/vulkan/third-party/Vulkan-Headers/include
      ${EXECUTORCH_ROOT}/backends/vulkan/third-party/volk
  )
  target_link_libraries(${library_name} vulkan_backend)
  target_compile_options(${library_name} PRIVATE ${VULKAN_CXX_FLAGS})
  # Link this library with --whole-archive due to dynamic shader registrations
  target_link_options_shared_lib(${library_name})

  unset(VULKAN_SHADERGEN_ENV)
  unset(VULKAN_SHADERGEN_OUT_PATH)
endmacro()
