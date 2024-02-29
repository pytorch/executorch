# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Simple CMake build system for selective build demo.
#
# ### Editing this file ###
#
# This file should be formatted with
# ~~~
# cmake-format --first-comment-is-literal=True CMakeLists.txt
# ~~~
# It should also be cmake-lint clean.
#

include(${EXECUTORCH_ROOT}/build/Utils.cmake)
include(${EXECUTORCH_ROOT}/build/Codegen.cmake)
#
# The `_<target>_srcs` lists are defined by including ${EXECUTORCH_SRCS_FILE}.
#
set(EXECUTORCH_SRCS_FILE
    "${CMAKE_CURRENT_BINARY_DIR}/../../../../executorch_srcs.cmake")

extract_sources(${EXECUTORCH_SRCS_FILE})

include(${EXECUTORCH_SRCS_FILE})

# build llama_runner library
list(TRANSFORM _llama_runner__srcs PREPEND "${EXECUTORCH_ROOT}/")

target_include_directories(extension_module
                           INTERFACE ${_common_include_directories})

add_library(llama_runner SHARED ${_llama_runner__srcs})
target_link_libraries(
  llama_runner PUBLIC executorch portable_kernels extension_module
                      extension_data_loader)
