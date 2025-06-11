# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Build llm runner lib.
#
# ### Editing this file ###
#
# This file should be formatted with
# ~~~
# cmake-format -i CMakeLists.txt
# ~~~
# It should also be cmake-lint clean.
#

if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
endif()

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)
include(${EXECUTORCH_ROOT}/tools/cmake/Codegen.cmake)

#
# The `_<target>_srcs` lists are defined by including ${EXECUTORCH_SRCS_FILE}.
#
set(EXECUTORCH_SRCS_FILE
    "${CMAKE_CURRENT_BINARY_DIR}/../../../executorch_srcs.cmake"
)

extract_sources(${EXECUTORCH_SRCS_FILE})

include(${EXECUTORCH_SRCS_FILE})

# build llm runner library
list(TRANSFORM _extension_llm_runner__srcs PREPEND "${EXECUTORCH_ROOT}/")

target_include_directories(
  extension_module INTERFACE ${_common_include_directories}
)

add_library(extension_llm_runner STATIC ${_extension_llm_runner__srcs})

# add tokenizers
ExternalProject_Add(
  tokenizers_external_project
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/../tokenizers
  SOURCE_DIR ${EXECUTORCH_ROOT}/extension/llm/tokenizers
  CMAKE_ARGS -DSUPPORT_REGEX_LOOKAHEAD=ON -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/../tokenizers
  INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/../tokenizers
)
add_dependencies(extension_llm_runner tokenizers_external_project)
find_package(tokenizers CONFIG HINTS ${CMAKE_INSTALL_PREFIX} ${CMAKE_CURRENT_BINARY_DIR}/../tokenizers)
set(runner_deps executorch extension_module extension_tensor ${TOKENIZERS_LIBRARIES})

target_link_libraries(extension_llm_runner PUBLIC ${runner_deps})

target_include_directories(
  extension_llm_runner PUBLIC ${_common_include_directories}
                                 ${TOKENIZERS_INCLUDE_DIRS}
                                 ${CMAKE_CURRENT_BINARY_DIR}/../tokenizers/include
)

if(BUILD_TESTING)
  add_subdirectory(test)
endif()
