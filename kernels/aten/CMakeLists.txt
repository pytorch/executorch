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

# Generate C++ bindings to register ATen kernels into Executorch (for runtime).
# Here select all ops in functions.yaml
gen_selected_ops(${CMAKE_CURRENT_SOURCE_DIR}/functions.yaml "" "")
# Expect gen_selected_ops output file to be selected_operators.yaml
# Not passing functions.yaml meaning this is ATen mode
generate_bindings_for_kernels("" "")

# Build a library for wrappers to ATen kernels
#
# portable_ops_lib: Register portable_ops_lib ops kernels into Executorch
# runtime
set(_deps torch executorch)
gen_operators_lib("aten_ops_lib" torch executorch)
