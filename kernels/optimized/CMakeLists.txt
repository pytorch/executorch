# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Kernel library for optimized kernels. Please this file formatted by running:
# ~~~
# cmake-format --first-comment-is-literal=True CMakeLists.txt
# ~~~

cmake_minimum_required(VERSION 3.19)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()
# Source root directory for pytorch. This is needed for kernel binding.
if(NOT TORCH_ROOT)
  set(TORCH_ROOT ${EXECUTORCH_ROOT}/third-party/pytorch)
endif()

set(_common_compile_options -Wno-deprecated-declarations)

include(${EXECUTORCH_ROOT}/build/Utils.cmake)
include(${EXECUTORCH_ROOT}/build/Codegen.cmake)

if(NOT PYTHON_EXECUTABLE)
  resolve_python_executable()
endif()

# Generate C++ bindings to register kernels into both PyTorch (for AOT) and
# Executorch (for runtime). Here select all ops in optimized.yaml
set(_yaml "${CMAKE_CURRENT_LIST_DIR}/optimized-oss.yaml")
gen_selected_ops("${_yaml}" "" "")

generate_bindings_for_kernels(
  FUNCTIONS_YAML ${CMAKE_CURRENT_SOURCE_DIR}/optimized-oss.yaml)
message("Generated files ${gen_command_sources}")

list(TRANSFORM _optimized_kernels__srcs PREPEND "${EXECUTORCH_ROOT}/")
add_library(optimized_kernels ${_optimized_kernels__srcs})
target_link_libraries(optimized_kernels PRIVATE executorch)
target_compile_options(optimized_kernels PUBLIC ${_common_compile_options})
# Build a library for _optimized_kernels_srcs
#
# optimized_ops_lib: Register optimized ops kernels into Executorch runtime
gen_operators_lib(
  "optimized_ops_lib"
  KERNEL_LIBS optimized_kernels
  DEPS executorch)

install(TARGETS optimized_kernels optimized_ops_lib DESTINATION lib)
