# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Please this file formatted by running:
# ~~~
# cmake-format -i CMakeLists.txt
# ~~~

cmake_minimum_required(VERSION 3.26)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(NOT EMSCRIPTEN)
  message(FATAL_ERROR "Emscripten is required to build this target")
endif()

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)
set(_common_compile_options -Wno-deprecated-declarations -fPIC -Wall -Werror)
set(_common_include_directories ${EXECUTORCH_ROOT}/..)

set(link_libraries)
list(APPEND link_libraries embind tokenizers::tokenizers)

add_library(tokenizers_wasm OBJECT tokenizers.cpp)

target_compile_options(tokenizers_wasm PUBLIC ${_common_compile_options})
target_include_directories(
  tokenizers_wasm PUBLIC ${_common_include_directories}
)

target_link_libraries(tokenizers_wasm PUBLIC ${link_libraries})

# The tokenizers library catches its own exceptions in normal operation: for
# example, HFTokenizer probes optional JSON keys and {"String": ...} patterns
# with try/catch. Emscripten disables exception catching unless -fexceptions is
# given, so without it those throws escape and load() fails on common
# tokenizer.json files. Compile the library and the bindings with it, and pass
# it on to the final link.
target_compile_options(tokenizers PRIVATE -fexceptions)
if(TARGET regex_lookahead)
  target_compile_options(regex_lookahead PRIVATE -fexceptions)
endif()
target_compile_options(tokenizers_wasm PUBLIC -fexceptions)
target_link_options(tokenizers_wasm PUBLIC -fexceptions)
