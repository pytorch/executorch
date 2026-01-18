# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# CMake build system for static LLM runner for CoreML.
#
# ### Building ###
#
# From the executorch root directory:
# ~~~
# mkdir cmake-out && cd cmake-out
# cmake .. \
#   -DEXECUTORCH_BUILD_EXTENSION_LLM=ON \
#   -DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=ON \
#   -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
#   -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
#   -DEXECUTORCH_BUILD_COREML=ON
# cmake --build . -j --target run_static_llm_coreml
# ~~~
#
# ### Running ###
#
# ~~~
# ./examples/apple/coreml/llama/runner/run_static_llm_coreml \
#   --model /path/to/model.pte \
#   --params /path/to/params.json \
#   --tokenizer /path/to/tokenizer.model \
#   --prompt "Once upon a time," \
#   --max_new_tokens 100
# ~~~

cmake_minimum_required(VERSION 3.19)
project(static_llm_coreml_runner)

if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../../..)
endif()

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

# Source files
set(static_llm_runner_srcs static_llm_runner.cpp)

set(static_llm_main_srcs main.cpp)

# Create the runner library
if(CMAKE_TOOLCHAIN_IOS
   OR ANDROID
   OR APPLE
)
  add_library(static_llm_runner STATIC ${static_llm_runner_srcs})
else()
  add_library(static_llm_runner SHARED ${static_llm_runner_srcs})
endif()

# Check for required targets
if(NOT TARGET extension_llm_runner)
  message(
    FATAL_ERROR
      "ExecuTorch must be built with EXECUTORCH_BUILD_EXTENSION_LLM_RUNNER enabled."
  )
endif()

if(NOT TARGET extension_module)
  message(
    FATAL_ERROR
      "ExecuTorch must be built with EXECUTORCH_BUILD_EXTENSION_MODULE enabled."
  )
endif()

# Dependencies for the runner library
set(static_llm_runner_deps
    executorch_core
    extension_data_loader
    extension_module
    extension_tensor
    extension_flat_tensor
    extension_llm_runner
    executorch_backends
)

# Add CoreML delegate if available (required for running CoreML models) Note:
# coremldelegate is linked transitively through executorch build system when
# EXECUTORCH_BUILD_COREML is ON. We don't need to link it again here to avoid
# duplicate symbol errors. if(TARGET coremldelegate) list(APPEND
# static_llm_runner_deps coremldelegate) endif()

target_link_libraries(static_llm_runner PUBLIC ${static_llm_runner_deps})
target_link_libraries(static_llm_runner PUBLIC tokenizers::tokenizers)

# Find nlohmann_json for params parsing - executorch already includes it in
# third-party
if(TARGET nlohmann_json)
  target_link_libraries(static_llm_runner PUBLIC nlohmann_json::nlohmann_json)
elseif(TARGET nlohmann_json::nlohmann_json)
  target_link_libraries(static_llm_runner PUBLIC nlohmann_json::nlohmann_json)
else()
  find_package(nlohmann_json QUIET CONFIG)
  if(nlohmann_json_FOUND)
    target_link_libraries(static_llm_runner PUBLIC nlohmann_json::nlohmann_json)
  else()
    # The executorch third-party directory includes nlohmann_json
    target_include_directories(
      static_llm_runner PUBLIC ${EXECUTORCH_ROOT}/third-party/json/include
    )
  endif()
endif()

target_include_directories(
  static_llm_runner
  PUBLIC ${EXECUTORCH_ROOT} ${EXECUTORCH_ROOT}/extension/llm/tokenizers/include
         ${EXECUTORCH_ROOT}/..
)

# Create the executable
add_executable(run_static_llm_coreml ${static_llm_main_srcs})

target_link_libraries(run_static_llm_coreml PRIVATE static_llm_runner)

# Find and link gflags - executorch builds gflags as a third-party dependency
if(TARGET gflags)
  target_link_libraries(run_static_llm_coreml PRIVATE gflags)
elseif(TARGET gflags_nothreads_static)
  target_link_libraries(run_static_llm_coreml PRIVATE gflags_nothreads_static)
else()
  find_package(gflags QUIET)
  if(gflags_FOUND)
    target_link_libraries(run_static_llm_coreml PRIVATE gflags::gflags)
  else()
    # Try to find gflags via pkg-config
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
      pkg_check_modules(GFLAGS QUIET gflags)
      if(GFLAGS_FOUND)
        target_include_directories(
          run_static_llm_coreml PRIVATE ${GFLAGS_INCLUDE_DIRS}
        )
        target_link_libraries(run_static_llm_coreml PRIVATE ${GFLAGS_LIBRARIES})
      else()
        message(
          FATAL_ERROR
            "gflags not found. Please install gflags or set GFLAGS_ROOT."
        )
      endif()
    else()
      message(
        FATAL_ERROR
          "gflags not found. Please install gflags or set GFLAGS_ROOT."
      )
    endif()
  endif()
endif()

target_include_directories(
  run_static_llm_coreml PRIVATE ${EXECUTORCH_ROOT} ${EXECUTORCH_ROOT}/..
)

# Set C++ standard
set_target_properties(
  static_llm_runner run_static_llm_coreml PROPERTIES CXX_STANDARD 17
                                                     CXX_STANDARD_REQUIRED ON
)

# Install targets
install(TARGETS run_static_llm_coreml DESTINATION bin)
install(TARGETS static_llm_runner DESTINATION lib)
install(FILES static_llm_runner.h
        DESTINATION include/executorch/examples/apple/coreml/llama/runner
)
