# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025 Arm Limited and/or its affiliates.
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

# Include this file to access executorch_target_link_options_shared_lib This is
# required to provide access to executorch_target_link_options_shared_lib which
# allows libraries to be linked with the --whole-archive flag. This is required
# for libraries that perform dynamic registration via static initialization.
include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

include(cmake/ShaderLibrary.cmake)

# Third party include paths

set(VULKAN_THIRD_PARTY_PATH ${CMAKE_CURRENT_SOURCE_DIR}/third-party)

set(VULKAN_HEADERS_PATH ${VULKAN_THIRD_PARTY_PATH}/Vulkan-Headers)
set(VOLK_PATH ${VULKAN_THIRD_PARTY_PATH}/volk)
set(VMA_PATH ${VULKAN_THIRD_PARTY_PATH}/VulkanMemoryAllocator)

# These third-party dependencies are git submodules. They are not part of the
# default submodule set checked out by install_executorch.py, so fail early with
# an actionable message rather than a confusing missing-header error.
if(NOT EXISTS "${VOLK_PATH}/volk.c"
   OR NOT EXISTS "${VULKAN_HEADERS_PATH}/include/vulkan/vulkan.h"
   OR NOT EXISTS "${VMA_PATH}/include/vk_mem_alloc.h"
)
  message(
    FATAL_ERROR
      "The Vulkan backend third-party submodules are missing. "
      "Run the following from the repository root:\n"
      "  git submodule update --init "
      "backends/vulkan/third-party/Vulkan-Headers "
      "backends/vulkan/third-party/volk "
      "backends/vulkan/third-party/VulkanMemoryAllocator"
  )
endif()

# Install the vendored Vulkan headers under ExecuTorch's include namespace so
# they cannot clobber or shadow a system or SDK copy in a shared prefix.
set(_vulkan_third_party_install_dir
    ${CMAKE_INSTALL_INCLUDEDIR}/executorch/backends/vulkan/third-party
)

set(COMMON_INCLUDES
    $<BUILD_INTERFACE:${EXECUTORCH_ROOT}/..>
    $<BUILD_INTERFACE:${VULKAN_HEADERS_PATH}/include>
    $<BUILD_INTERFACE:${VOLK_PATH}>
    $<BUILD_INTERFACE:${VMA_PATH}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    $<INSTALL_INTERFACE:${_vulkan_third_party_install_dir}>
)

# Compile settings

# Exceptions are required: the vk_api layer throws on Vulkan errors (see
# vk_api/Exception.h). MSVC does not understand -fexceptions and enables C++
# exceptions via /EHsc instead, so select the flag per compiler.
set(VULKAN_CXX_FLAGS "$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fexceptions>")
list(APPEND VULKAN_CXX_FLAGS "$<$<CXX_COMPILER_ID:MSVC>:/EHsc>")

# vulkan API files

file(GLOB_RECURSE vulkan_api_cpp CONFIGURE_DEPENDS ${RUNTIME_PATH}/api/*.cpp)
file(GLOB_RECURSE vulkan_vkapi_cpp CONFIGURE_DEPENDS
     ${RUNTIME_PATH}/vk_api/*.cpp
)
list(APPEND vulkan_api_cpp ${vulkan_vkapi_cpp})
list(APPEND vulkan_api_cpp ${VOLK_PATH}/volk.c)
file(GLOB_RECURSE vulkan_runtime_utils_cpp CONFIGURE_DEPENDS
     ${RUNTIME_PATH}/utils/*.cpp
)

find_package(Threads REQUIRED)
add_library(vulkan_compute_api ${vulkan_api_cpp} ${vulkan_runtime_utils_cpp})
add_library(executorch::vulkan_compute_api ALIAS vulkan_compute_api)
target_include_directories(vulkan_compute_api PUBLIC ${COMMON_INCLUDES})
target_compile_features(vulkan_compute_api PUBLIC cxx_std_17)
target_compile_definitions(
  vulkan_compute_api PUBLIC USE_VULKAN_WRAPPER USE_VULKAN_VOLK
)
target_compile_options(vulkan_compute_api PRIVATE ${VULKAN_CXX_FLAGS})
target_link_libraries(
  vulkan_compute_api
  PUBLIC executorch_core
  PRIVATE Threads::Threads ${CMAKE_DL_LIBS}
)

find_package(Boost 1.89 COMPONENTS stacktrace_basic stacktrace_addr2line)
if(TARGET Boost::stacktrace_basic AND TARGET Boost::stacktrace_addr2line)
  target_include_directories(vulkan_compute_api PRIVATE ${Boost_INCLUDE_DIRS})
  target_link_libraries(
    vulkan_compute_api PRIVATE ${Boost_STACKTRACE_BASIC_LIBRARY}
                               ${Boost_STACKTRACE_ADDR2LINE_LIBRARY}
  )
  target_compile_definitions(
    vulkan_compute_api PRIVATE ETVK_BOOST_STACKTRACE_AVAILABLE
  )
endif()

# vulkan ComputeGraph files

file(GLOB_RECURSE vulkan_graph_cpp CONFIGURE_DEPENDS
     ${RUNTIME_PATH}/graph/*.cpp
)

# Standard GLSL shader library

set(VULKAN_GRAPH_SHADERS_PATH ${RUNTIME_PATH}/graph/ops/glsl/)
# Generates a spv.cpp file containing compiled GLSL shaders
gen_vulkan_shader_lib_cpp(${VULKAN_GRAPH_SHADERS_PATH})
# Save the path of the generated cpp file
set(_vulkan_standard_shaders_cpp ${generated_spv_cpp})

add_library(
  vulkan_graph_runtime ${vulkan_graph_cpp} ${_vulkan_standard_shaders_cpp}
)
add_library(executorch::vulkan_graph_runtime ALIAS vulkan_graph_runtime)
target_link_libraries(vulkan_graph_runtime PUBLIC vulkan_compute_api)
target_compile_options(vulkan_graph_runtime PRIVATE ${VULKAN_CXX_FLAGS})
executorch_target_link_options_shared_lib(vulkan_graph_runtime)

install(
  TARGETS vulkan_compute_api vulkan_graph_runtime
  EXPORT ExecuTorchTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(
  DIRECTORY ${RUNTIME_PATH}/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/executorch/backends/vulkan/runtime
  FILES_MATCHING
  PATTERN "*.h"
)
install(DIRECTORY ${VULKAN_HEADERS_PATH}/include/
        DESTINATION ${_vulkan_third_party_install_dir}
)
install(FILES ${VOLK_PATH}/volk.h
        DESTINATION ${_vulkan_third_party_install_dir}
)
install(FILES ${VMA_PATH}/include/vk_mem_alloc.h
        DESTINATION ${_vulkan_third_party_install_dir}/include
)

if(NOT EXECUTORCH_BUILD_VULKAN AND NOT EXECUTORCH_BUILD_WEBGPU)
  return()
endif()

# Generate Vulkan Delegate Schema Files from flatc

set(SCHEMA_INCLUDE_DIR ${CMAKE_BINARY_DIR}/schema/include)

set(GENERATED_HEADER
    ${SCHEMA_INCLUDE_DIR}/executorch/backends/vulkan/serialization/schema_generated.h
)

list(TRANSFORM _vulkan_schema__srcs
     PREPEND "${EXECUTORCH_ROOT}/" OUTPUT_VARIABLE _vulkan_schema_abs_srcs
)

add_custom_command(
  OUTPUT ${GENERATED_HEADER}
  COMMAND
    flatc --cpp --cpp-std c++11 --scoped-enums -o
    "${SCHEMA_INCLUDE_DIR}/executorch/backends/vulkan/serialization/"
    ${_vulkan_schema__srcs}
  WORKING_DIRECTORY ${EXECUTORCH_ROOT}
  DEPENDS flatc ${_vulkan_schema_abs_srcs}
  COMMENT "Generating vulkan_schema headers"
  VERBATIM
)

# vulkan_schema library

add_library(vulkan_schema INTERFACE ${GENERATED_HEADER})
set_target_properties(vulkan_schema PROPERTIES LINKER_LANGUAGE CXX)

target_include_directories(
  vulkan_schema
  INTERFACE
    $<BUILD_INTERFACE:${SCHEMA_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${EXECUTORCH_ROOT}/third-party/flatbuffers/include>
)

# vulkan_backend

if(NOT EXECUTORCH_BUILD_VULKAN)
  install(TARGETS vulkan_schema EXPORT ExecuTorchTargets)
  return()
endif()

file(GLOB vulkan_backend_cpp CONFIGURE_DEPENDS ${RUNTIME_PATH}/*.cpp)

add_library(vulkan_backend ${vulkan_backend_cpp})
target_link_libraries(
  vulkan_backend
  PUBLIC vulkan_graph_runtime
  PRIVATE vulkan_schema
)
target_compile_options(vulkan_backend PRIVATE ${VULKAN_CXX_FLAGS})
# Link this library with --whole-archive due to dynamic backend registration
executorch_target_link_options_shared_lib(vulkan_backend)

set_property(TARGET vulkan_backend PROPERTY CXX_STANDARD 17)

# Test targets

install(
  TARGETS vulkan_backend vulkan_schema
  EXPORT ExecuTorchTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
