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

if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

if(NOT RUNTIME_PATH)
  set(RUNTIME_PATH ${CMAKE_CURRENT_SOURCE_DIR}/runtime)
endif()

if(NOT PYTHON_EXECUTABLE)
  set(PYTHON_EXECUTABLE python3)
endif()

if(NOT FLATC_EXECUTABLE)
  set(FLATC_EXECUTABLE flatc)
endif()

# Include this file to access target_link_options_shared_lib This is required to
# provide access to target_link_options_shared_lib which allows libraries to be
# linked with the --whole-archive flag. This is required for libraries that
# perform dynamic registration via static initialization.
include(${EXECUTORCH_ROOT}/build/Utils.cmake)

include(cmake/ShaderLibrary.cmake)

# Third party include paths

set(VULKAN_THIRD_PARTY_PATH ${CMAKE_CURRENT_SOURCE_DIR}/third-party)

set(VULKAN_HEADERS_PATH ${VULKAN_THIRD_PARTY_PATH}/Vulkan-Headers)
set(VOLK_PATH ${VULKAN_THIRD_PARTY_PATH}/volk)
set(VMA_PATH ${VULKAN_THIRD_PARTY_PATH}/VulkanMemoryAllocator)

set(COMMON_INCLUDES ${EXECUTORCH_ROOT}/.. ${VULKAN_HEADERS_PATH}/include
                    ${VOLK_PATH} ${VMA_PATH}
)

# Compile settings

set(VULKAN_CXX_FLAGS "-fexceptions")
list(APPEND VULKAN_CXX_FLAGS "-DUSE_VULKAN_WRAPPER")
list(APPEND VULKAN_CXX_FLAGS "-DUSE_VULKAN_VOLK")

# vulkan API files

file(GLOB_RECURSE vulkan_api_cpp ${RUNTIME_PATH}/api/*)
file(GLOB_RECURSE vulkan_vkapi_cpp ${RUNTIME_PATH}/vk_api/*)
list(APPEND vulkan_api_cpp ${vulkan_vkapi_cpp})
list(APPEND vulkan_api_cpp ${VOLK_PATH}/volk.c)

# vulkan ComputeGraph files

file(GLOB_RECURSE vulkan_graph_cpp ${RUNTIME_PATH}/graph/*)
list(APPEND vulkan_graph_cpp ${vulkan_api_cpp})

# Standard GLSL shader library

set(VULKAN_GRAPH_SHADERS_PATH ${RUNTIME_PATH}/graph/ops/glsl/)
# Generates a spv.cpp file containing compiled GLSL shaders
gen_vulkan_shader_lib_cpp(${VULKAN_GRAPH_SHADERS_PATH})
# Save the path of the generated cpp file
set(vulkan_standard_shaders_cpp ${generated_spv_cpp})

# Generate Vulkan Delegate Schema Files from flatc

set(SCHEMA_INCLUDE_DIR ${CMAKE_BINARY_DIR}/schema/include)

set(GENERATED_HEADER
    ${SCHEMA_INCLUDE_DIR}/executorch/backends/vulkan/schema_generated.h
)

add_custom_command(
  OUTPUT ${GENERATED_HEADER}
  COMMAND
    ${FLATC_EXECUTABLE} --cpp --cpp-std c++11 --scoped-enums -o
    "${SCHEMA_INCLUDE_DIR}/executorch/backends/vulkan/" ${_vulkan_schema__srcs}
  WORKING_DIRECTORY ${EXECUTORCH_ROOT}
  COMMENT "Generating vulkan_schema headers"
  VERBATIM
)

# vulkan_schema library

add_library(vulkan_schema INTERFACE ${GENERATED_HEADER})
set_target_properties(vulkan_schema PROPERTIES LINKER_LANGUAGE CXX)

target_include_directories(
  vulkan_schema INTERFACE ${SCHEMA_INCLUDE_DIR}
                          ${EXECUTORCH_ROOT}/third-party/flatbuffers/include
)

# vulkan_backend

file(GLOB vulkan_backend_cpp ${RUNTIME_PATH}/*.cpp)
list(APPEND vulkan_backend_cpp ${vulkan_graph_cpp})
list(APPEND vulkan_backend_cpp ${vulkan_standard_shaders_cpp})

add_library(vulkan_backend STATIC ${vulkan_backend_cpp})
target_include_directories(
  vulkan_backend PRIVATE ${SCHEMA_INCLUDE_DIR} ${COMMON_INCLUDES}
)
target_link_libraries(vulkan_backend PRIVATE vulkan_schema executorch)
target_compile_options(vulkan_backend PRIVATE ${VULKAN_CXX_FLAGS})
# Link this library with --whole-archive due to dynamic backend registration
target_link_options_shared_lib(vulkan_backend)

set_property(TARGET vulkan_backend PROPERTY CXX_STANDARD 17)

# Executor Runner

if(NOT CMAKE_TOOLCHAIN_FILE MATCHES ".*(iOS|ios\.toolchain)\.cmake$")
  set(VULKAN_RUNNER_SRCS ${_executor_runner__srcs})
  list(TRANSFORM VULKAN_RUNNER_SRCS PREPEND "${EXECUTORCH_ROOT}/")

  add_executable(vulkan_executor_runner ${VULKAN_RUNNER_SRCS})
  target_link_libraries(
    vulkan_executor_runner ${_executor_runner_libs} vulkan_schema
    vulkan_backend
  )
  target_compile_options(vulkan_executor_runner PUBLIC ${VULKAN_CXX_FLAGS})

  add_library(vulkan_executor_runner_lib STATIC ${VULKAN_RUNNER_SRCS})
  target_link_libraries(
    vulkan_executor_runner_lib ${_executor_runner_libs} vulkan_schema
    vulkan_backend
  )
  target_compile_options(vulkan_executor_runner_lib PUBLIC ${VULKAN_CXX_FLAGS})
endif()

# Test targets

install(
  TARGETS vulkan_backend
  DESTINATION lib
  INCLUDES
  DESTINATION ${COMMON_INCLUDES}
)
