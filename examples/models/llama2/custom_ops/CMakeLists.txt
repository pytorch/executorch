# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


cmake_minimum_required(VERSION 3.19)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(NOT PYTHON_EXECUTABLE)
  set(PYTHON_EXECUTABLE python3)
endif()

if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../..)
endif()

# Is this necessary?
set(_common_compile_options -Wno-deprecated-declarations)


list(TRANSFORM _custom_ops__srcs PREPEND "${EXECUTORCH_ROOT}/")
add_library(custom_ops ${_custom_ops__srcs})

#add_library(custom_ops OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/op_sdpa.cpp)
target_include_directories(custom_ops
                           PUBLIC ${_common_include_directories})

# target_link_libraries(custom_ops extension_module)
