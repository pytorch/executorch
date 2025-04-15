# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# CMakeLists.txt for ExecuTorch LLM Runner Python bindings

if(NOT EXECUTORCH_BUILD_PYTHON_BINDINGS)
  return()
endif()

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(
  llm_runner
  bindings.cpp
)

target_link_libraries(
  llm_runner
  PRIVATE
  executorch_extension_llm_runner
  examples_models_llama_runner
)

target_include_directories(
  llm_runner
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../..  # executorch root
)

install(TARGETS llm_runner DESTINATION .)