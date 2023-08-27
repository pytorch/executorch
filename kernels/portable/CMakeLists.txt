# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Kernel library for portable kernels. Please this file formatted by running:
# ~~~
# cmake-format --first-comment-is-literal=True CMakeLists.txt
# ~~~

cmake_minimum_required(VERSION 3.19)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(NOT PYTHON_EXECUTABLE)
  set(PYTHON_EXECUTABLE python3)
endif()
# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()
# Source root directory for pytorch.
if(NOT TORCH_ROOT)
  set(TORCH_ROOT ${EXECUTORCH_ROOT}/third-party/pytorch)
endif()

set(_common_compile_options -Wno-deprecated-declarations)

include(${EXECUTORCH_ROOT}/build/Utils.cmake)
include(${EXECUTORCH_ROOT}/build/Codegen.cmake)
# Portable kernel sources TODO(larryliu0820): use buck2 to gather the sources
file(GLOB_RECURSE _portable_kernels__srcs
     "${CMAKE_CURRENT_SOURCE_DIR}/cpu/*.cpp")
list(FILTER _portable_kernels__srcs EXCLUDE REGEX "test/*.cpp")
list(FILTER _portable_kernels__srcs EXCLUDE REGEX "codegen")
# Generate C++ bindings to register kernels into both PyTorch (for AOT) and
# Executorch (for runtime). Here select all ops in functions.yaml
gen_selected_ops("${CMAKE_CURRENT_LIST_DIR}/functions.yaml" "" "")
# Expect gen_selected_ops output file to be selected_operators.yaml
generate_bindings_for_kernels(${CMAKE_CURRENT_SOURCE_DIR}/functions.yaml "")
message("Generated files ${gen_command_sources}")

#
# portable_kernels: Pure-C++ kernel library for ATen ops
#
# Focused on portability and understandability rather than speed.
#
add_library(portable_kernels ${_portable_kernels__srcs})
target_link_libraries(portable_kernels PRIVATE executorch)
target_compile_options(portable_kernels PUBLIC ${_common_compile_options})

# Build a library for _portable_kernels__srcs
#
# portable_ops_lib: Register portable_ops_lib ops kernels into Executorch
# runtime
gen_operators_lib("portable_ops_lib" portable_kernels executorch)
