# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# ### Editing this file ###
#
# This file should be formatted with
# ~~~
# cmake-format -i CMakeLists.txt
# ~~~
# It should also be cmake-lint clean.
#
# The targets in this file will be built if EXECUTORCH_BUILD_VULKAN is ON

cmake_minimum_required(VERSION 3.19)
project(executorch)

find_package(executorch CONFIG REQUIRED COMPONENTS vulkan_backend)
find_package(GTest CONFIG REQUIRED)

# Only build tests if Vulkan was compiled
find_library(LIB_VULKAN_BACKEND vulkan_backend)

if(LIB_VULKAN_BACKEND)

  if(NOT EXECUTORCH_ROOT)
    set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
  endif()

  if(NOT PYTHON_EXECUTABLE)
    set(PYTHON_EXECUTABLE python3)
  endif()

  # Include this file to access target_link_options_shared_lib This is required
  # to provide access to target_link_options_shared_lib which allows libraries
  # to be linked with the --whole-archive flag. This is required for libraries
  # that perform dynamic registration via static initialization.
  include(${EXECUTORCH_ROOT}/build/Utils.cmake)

  include(../cmake/ShaderLibrary.cmake)

  # Third party include paths

  set(VULKAN_THIRD_PARTY_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../third-party)

  set(GTEST_INCLUDE_PATH
      ${EXECUTORCH_ROOT}/third-party/googletest/googletest/include set
      (PYTORCH_PATH ${EXECUTORCH_ROOT}/third-party/pytorch)
  )
  set(VULKAN_HEADERS_PATH ${VULKAN_THIRD_PARTY_PATH}/Vulkan-Headers/include)
  set(VOLK_PATH ${VULKAN_THIRD_PARTY_PATH}/volk)
  set(VMA_PATH ${VULKAN_THIRD_PARTY_PATH}/VulkanMemoryAllocator)

  set(COMMON_INCLUDES ${EXECUTORCH_ROOT}/.. ${VULKAN_HEADERS_PATH} ${VOLK_PATH}
                      ${VMA_PATH} ${GTEST_INCLUDE_PATH} ${PYTORCH_PATH}
  )

  # Test Utility files

  set(TEST_UTILS_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/utils)
  file(GLOB TEST_UTILS_CPP ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cpp)

  # Test shaders

  set(TEST_SHADERS_PATH ${CMAKE_CURRENT_SOURCE_DIR}/glsl)
  gen_vulkan_shader_lib_cpp(${TEST_SHADERS_PATH})
  vulkan_shader_lib(test_shaderlib ${generated_spv_cpp})

  # API Test binary

  set(COMPUTE_API_TEST_CPP
      ${CMAKE_CURRENT_SOURCE_DIR}/vulkan_compute_api_test.cpp
  )

  target_link_options_shared_lib(vulkan_backend)

  add_executable(
    vulkan_compute_api_test ${COMPUTE_API_TEST_CPP} ${TEST_UTILS_CPP}
  )
  target_include_directories(vulkan_compute_api_test PRIVATE ${COMMON_INCLUDES})
  target_link_libraries(
    vulkan_compute_api_test PRIVATE GTest::gtest_main vulkan_backend executorch
                                    test_shaderlib
  )
  target_compile_options(vulkan_compute_api_test PRIVATE ${VULKAN_CXX_FLAGS})

  set_property(TARGET vulkan_compute_api_test PROPERTY CXX_STANDARD 17)

endif()
