# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set(CMAKE_XCODE_ATTRIBUTE_CLANG_CXX_LANGUAGE_STANDARD "c++${CMAKE_CXX_STANDARD}")
set(CMAKE_XCODE_ATTRIBUTE_CLANG_CXX_LIBRARY "libc++")

set(
  _compiler_flags
  "-ffile-prefix-map=${PROJECT_SOURCE_DIR}=/executorch"
  "-fdebug-prefix-map=${PROJECT_SOURCE_DIR}=/executorch"
)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${_compiler_flags}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_compiler_flags}")

set_overridable_option(EXECUTORCH_BUILD_XNNPACK ON)
set_overridable_option(EXECUTORCH_BUILD_COREML ON)
set_overridable_option(EXECUTORCH_BUILD_MPS ON)
set_overridable_option(EXECUTORCH_XNNPACK_SHARED_WORKSPACE ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_APPLE ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_MODULE ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_TENSOR ON)
set_overridable_option(EXECUTORCH_BUILD_KERNELS_CUSTOM ON)
set_overridable_option(EXECUTORCH_BUILD_KERNELS_OPTIMIZED ON)
set_overridable_option(EXECUTORCH_BUILD_KERNELS_QUANTIZED ON)
