# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Please this file formatted by running:
# ~~~
# cmake-format --first-comment-is-literal=True CMakeLists.txt
# ~~~

find_package(Torch CONFIG REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python
               PATHS "${TORCH_INSTALL_PREFIX}/lib")

# Override compiler flags set in upper scope. ExecuTorch builds with
# -fno-exceptions and -fno-rtti, but we need these for ATen.
set(CMAKE_CXX_FLAGS_RELEASE "")

#list(TRANSFORM _xnnpack_convert_to_qc4w__srcs PREPEND "${EXECUTORCH_ROOT}/")
# add_library(aot_util ${_xnnpack_convert_to_qc4w__srcs})
add_library(aot_util "convert_to_qc4w.cpp")
target_include_directories(aot_util PUBLIC ${TORCH_INCLUDE_DIRS})
target_link_libraries(aot_util torch)
