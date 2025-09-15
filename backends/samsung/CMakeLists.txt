# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.15)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

get_filename_component(
  EXECUTORCH_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE
)

if(NOT DEFINED EXYNOS_AI_LITECORE_ROOT)
  message(
    FATAL_ERROR
      "Please define EXYNOS_AI_LIRECORE_PATH by adding cmake parameter -DEXYNOS_AI_LITECORE_ROOT=<...>"
  )
endif()
if(CMAKE_TOOLCHAIN_FILE MATCHES ".*(iOS|ios\.toolchain)\.cmake$")
  message(FATAL_ERROR "IOS is not supported on Exynos.")
endif()

if(NOT FLATC_EXECUTABLE)
  set(FLATC_EXECUTABLE flatc)
endif()

add_compile_options(-Wall -Werror -fPIC)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  # strip symbols
  add_link_options("-s")
  # hide dynamic symbols
  set(CMAKE_C_VISIBILITY_PRESET hidden)
  set(CMAKE_CXX_VISIBILITY_PRESET hidden)
  add_definitions(-DNDEBUG)
endif()

include_directories(
  ${EXECUTORCH_SOURCE_DIR}/..
  ${EXECUTORCH_SOURCE_DIR}/runtime/core/portable_type/c10
  ${EXYNOS_AI_LITECORE_ROOT}
)
add_compile_definitions(C10_USING_CUSTOM_GENERATED_MACROS)

if(${ANDROID})
  find_library(android_log log)
endif()

# add logging library
add_library(enn_logging STATIC)

if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64")
  add_subdirectory(
    ${EXECUTORCH_SOURCE_DIR}/third-party/pybind11
    ${CMAKE_CURRENT_BINARY_DIR}/pybind11
  )
  add_library(PyEnnWrapperAdaptor MODULE)

  find_library(
    GG_API_LIB
    NAMES graphgen_api
    HINTS ${EXYNOS_AI_LITECORE_ROOT}/lib/x86_64-linux
  )
  add_library(graphgen_api SHARED IMPORTED GLOBAL)
  set_target_properties(
    graphgen_api
    PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
               "${EXYNOS_AI_LITECORE_ROOT}/include" IMPORTED_LOCATION
                                                    "${GG_API_LIB}"
  )

  set(_enn_compile_options_schema
      ${CMAKE_CURRENT_SOURCE_DIR}/serialization/compile_options_def.fbs
  )

  set(_enn_schema_generate_dir
      "${CMAKE_BINARY_DIR}/schema/include/executorch/backends/samsung"
  )
  # Paths to headers generated from the .fbs files.
  string(REGEX REPLACE "serialization/([^/]+)[.]fbs$" "\\1_generated.h"
                       generated_header "${fbs_file}"
  )
  set(_enn_schema_output "${_enn_schema_generate_dir}/${generated_header}")

  # Generate the headers from the .fbs files.
  add_custom_command(
    OUTPUT ${_enn_schema_output}
    COMMAND ${FLATC_EXECUTABLE} --cpp --cpp-std c++11 --scoped-enums -o
            "${_enn_schema_generate_dir}" ${_enn_compile_options_schema}
    DEPENDS ${_enn_compile_options_schema}
    WORKING_DIRECTORY ${EXECUTORCH_SOURCE_DIR}
    COMMENT "Generating enn compile options headers"
    VERBATIM
  )
  add_custom_target(
    enn_compile_options_output ALL DEPENDS ${_enn_schema_output}
  )

  set_target_properties(
    PyEnnWrapperAdaptor PROPERTIES CXX_VISIBILITY_PRESET hidden
  )
  target_link_libraries(
    PyEnnWrapperAdaptor PRIVATE pybind11::module pybind11::lto graphgen_api
                                enn_logging
  )
  target_include_directories(
    PyEnnWrapperAdaptor BEFORE
    PRIVATE ${CMAKE_BINARY_DIR}/schema/include
            ${EXECUTORCH_SOURCE_DIR}/third-party/flatbuffers/include
  )
  add_dependencies(PyEnnWrapperAdaptor enn_compile_options_output)
  pybind11_extension(PyEnnWrapperAdaptor)

  # PyGraphWrapperAdaptor
  add_library(PyGraphWrapperAdaptor MODULE)
  #
  find_library(
    GRAPH_WRAPPER_LIB
    NAMES graph_wrapper
    HINTS ${EXYNOS_AI_LITECORE_ROOT}/lib/x86_64-linux
  )
  add_library(graph_wrapper SHARED IMPORTED GLOBAL)
  set_target_properties(
    graph_wrapper
    PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
               "${EXYNOS_AI_LITECORE_ROOT}/include" IMPORTED_LOCATION
                                                    "${GRAPH_WRAPPER_LIB}"
  )
  set_target_properties(
    PyGraphWrapperAdaptor PROPERTIES CXX_VISIBILITY_PRESET hidden
  )
  target_link_libraries(
    PyGraphWrapperAdaptor PRIVATE pybind11::module pybind11::lto graph_wrapper
                                  enn_logging
  )
  pybind11_extension(PyGraphWrapperAdaptor)

  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/aot)
endif()

if(${ANDROID})
  target_link_libraries(enn_logging PRIVATE ${android_log})
  add_library(enn_backend STATIC)
  target_link_libraries(enn_backend PRIVATE enn_logging extension_runner_util)
  executorch_target_link_options_shared_lib(enn_backend)
  target_compile_options(enn_backend PRIVATE -Wno-deprecated-declarations)

  set(__enn_executor_runner_srcs
      ${EXECUTORCH_SOURCE_DIR}/examples/samsung/executor_runner/enn_executor_runner.cpp
  )
  add_executable(enn_executor_runner ${__enn_executor_runner_srcs})
  add_dependencies(enn_executor_runner enn_backend)
  target_link_libraries(
    enn_executor_runner PRIVATE enn_logging enn_backend gflags executorch
                                extension_data_loader portable_ops_lib
                                android
  )
  set_target_properties(
    enn_executor_runner PROPERTIES CXX_VISIBILITY_PRESET hidden
  )
  install(
    TARGETS enn_backend enn_logging
    EXPORT ExecuTorchTargets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/runtime)
