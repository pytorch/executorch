# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Please this file formatted by running:
# ~~~
# cmake-format -i CMakeLists.txt
# ~~~

cmake_minimum_required(VERSION 3.19)

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
endif()

set(ABSL_ENABLE_INSTALL ON)
set(ABSL_PROPAGATE_CXX_STD ON)
set(_pic_flag ${CMAKE_POSITION_INDEPENDENT_CODE})
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(
  ${CMAKE_CURRENT_SOURCE_DIR}/../third-party/abseil-cpp
  ${CMAKE_CURRENT_BINARY_DIR}/abseil-cpp
)
add_subdirectory(
  ${CMAKE_CURRENT_SOURCE_DIR}/../third-party/re2
  ${CMAKE_CURRENT_BINARY_DIR}/re2
)
set(CMAKE_POSITION_INDEPENDENT_CODE ${_pic_flag})

list(TRANSFORM _extension_llm_tokenizer__srcs PREPEND "${EXECUTORCH_ROOT}/")
add_library(extension_llm_tokenizer ${_extension_llm_tokenizer__srcs})
target_include_directories(
  extension_llm_tokenizer PUBLIC ${EXECUTORCH_ROOT}/..
                                 ${_common_include_directories}
)

target_link_libraries(extension_llm_tokenizer re2::re2)
target_compile_options(
  extension_llm_tokenizer PUBLIC ${_common_compile_options}
)

# Install libraries
install(
  TARGETS extension_llm_tokenizer
  DESTINATION lib
  INCLUDES
  DESTINATION ${_common_include_directories}
)

target_include_directories(
  extension_llm_tokenizer
  PRIVATE ${CMAKE_INSTALL_PREFIX}/include
          ${CMAKE_CURRENT_SOURCE_DIR}/../third-party/abseil-cpp
)

if(BUILD_TESTING)
  add_subdirectory(test)
endif()
