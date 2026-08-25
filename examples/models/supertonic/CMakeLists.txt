# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.24)
project(supertonic_runner)

include(GNUInstallDirs)

if(NOT APPLE)
  message(FATAL_ERROR "The Supertonic MLX runner is supported only on Darwin.")
endif()

set(_supertonic_target_processor "${CMAKE_OSX_ARCHITECTURES}")
if(NOT _supertonic_target_processor)
  set(_supertonic_target_processor "${CMAKE_SYSTEM_PROCESSOR}")
endif()
if(NOT _supertonic_target_processor STREQUAL "arm64")
  message(
    FATAL_ERROR
      "The Supertonic MLX runner requires an arm64 Darwin target; detected '${_supertonic_target_processor}'."
  )
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EXECUTORCH_SOURCE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
include(${EXECUTORCH_SOURCE_ROOT}/tools/cmake/Utils.cmake)

set(_json_include
    ${EXECUTORCH_SOURCE_ROOT}/extension/llm/tokenizers/third-party/json/single_include
)
find_library(COREFOUNDATION_FRAMEWORK CoreFoundation REQUIRED)

set(_helper_sources runtime/supertonic_runner.cpp runtime/text_processor.cpp
                    runtime/style_loader.cpp runtime/wav_writer.cpp
)

enable_testing()
add_executable(
  supertonic_runtime_test runtime/tests/supertonic_runtime_test.cpp
                          ${_helper_sources}
)
target_compile_definitions(
  supertonic_runtime_test PRIVATE SUPERTONIC_PURE_HELPERS_ONLY
)
target_include_directories(
  supertonic_runtime_test PRIVATE runtime ${_json_include}
)
target_link_libraries(
  supertonic_runtime_test PRIVATE ${COREFOUNDATION_FRAMEWORK}
)
add_test(NAME supertonic_runtime_helpers COMMAND supertonic_runtime_test)

set(gflags_DIR ${CMAKE_CURRENT_BINARY_DIR}/../../../third-party/gflags)
find_package(gflags REQUIRED)

list(APPEND CMAKE_FIND_ROOT_PATH ${CMAKE_CURRENT_BINARY_DIR}/../../..)
find_package(executorch CONFIG REQUIRED FIND_ROOT_PATH_BOTH)
executorch_target_link_options_shared_lib(executorch)

if(NOT TARGET mlxdelegate OR NOT TARGET mlx)
  message(
    FATAL_ERROR
      "ExecuTorch must be installed with the MLX delegate and runtime targets."
  )
endif()

foreach(required_target extension_module extension_data_loader extension_tensor
                        extension_flat_tensor
)
  if(NOT TARGET ${required_target})
    message(
      FATAL_ERROR
        "ExecuTorch installation is missing required target ${required_target}."
    )
  endif()
endforeach()

add_executable(supertonic_runner runtime/main.cpp ${_helper_sources})
target_compile_definitions(supertonic_runner PRIVATE EXECUTORCH_BUILD_MLX)
target_include_directories(
  supertonic_runner PRIVATE ${EXECUTORCH_SOURCE_ROOT}/.. runtime
                            ${_json_include}
)
target_link_libraries(
  supertonic_runner
  PRIVATE executorch
          extension_module
          extension_data_loader
          extension_tensor
          extension_flat_tensor
          mlxdelegate
          mlx
          gflags
          ${COREFOUNDATION_FRAMEWORK}
)
executorch_target_link_options_shared_lib(mlxdelegate)
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_link_options_gc_sections(supertonic_runner)
endif()
set(EXECUTORCH_BUILD_MLX ON)
if(NOT DEFINED MLX_METALLIB_PATH OR NOT EXISTS "${MLX_METALLIB_PATH}")
  message(FATAL_ERROR "The installed MLX package did not provide mlx.metallib.")
endif()
executorch_target_copy_mlx_metallib(supertonic_runner)
install(TARGETS supertonic_runner RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(
  FILES "${MLX_METALLIB_PATH}"
  DESTINATION ${CMAKE_INSTALL_BINDIR}
  RENAME mlx.metallib
)

set(SUPERTONIC_INTEGRATION_PTE
    ""
    CACHE FILEPATH "Optional Supertonic PTE for the integration test."
)
set(SUPERTONIC_INTEGRATION_ASSET_DIR
    ""
    CACHE PATH "Optional published Supertonic asset directory for integration."
)
set(SUPERTONIC_INTEGRATION_STYLE
    ""
    CACHE FILEPATH "Optional published voice-style JSON for integration."
)
add_test(
  NAME supertonic_integration
  COMMAND
    ${CMAKE_COMMAND} -DRUNNER=$<TARGET_FILE:supertonic_runner>
    -DPTE=${SUPERTONIC_INTEGRATION_PTE}
    -DASSET_DIR=${SUPERTONIC_INTEGRATION_ASSET_DIR}
    -DSTYLE=${SUPERTONIC_INTEGRATION_STYLE}
    -DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/supertonic-integration.wav -P
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/tests/run_integration.cmake
)
set_tests_properties(
  supertonic_integration
  PROPERTIES TIMEOUT 1800 SKIP_REGULAR_EXPRESSION
             "SKIP: Supertonic integration assets are unavailable"
)
