# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set(CMAKE_XCODE_ATTRIBUTE_CLANG_CXX_LANGUAGE_STANDARD
    "c++${CMAKE_CXX_STANDARD}"
)
set(CMAKE_XCODE_ATTRIBUTE_CLANG_CXX_LIBRARY "libc++")

# Clean up the paths LLDB sees in DWARF.
add_compile_options(
  -ffile-prefix-map=${PROJECT_SOURCE_DIR}=/executorch
  -fdebug-prefix-map=${PROJECT_SOURCE_DIR}=/executorch
)

set_overridable_option(BUILD_TESTING OFF)
set_overridable_option(EXECUTORCH_BUILD_XNNPACK ON)
set_overridable_option(EXECUTORCH_BUILD_COREML ON)
set_overridable_option(EXECUTORCH_BUILD_MPS ON)
set_overridable_option(EXECUTORCH_XNNPACK_ENABLE_WEIGHT_CACHE ON)
set_overridable_option(EXECUTORCH_XNNPACK_SHARED_WORKSPACE ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_APPLE ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_LLM_APPLE ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_LLM ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_LLM_RUNNER ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_MODULE ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_TENSOR ON)
set_overridable_option(EXECUTORCH_BUILD_KERNELS_LLM ON)
set_overridable_option(EXECUTORCH_BUILD_KERNELS_OPTIMIZED ON)
set_overridable_option(EXECUTORCH_BUILD_KERNELS_QUANTIZED ON)
set_overridable_option(EXECUTORCH_BUILD_KERNELS_TORCHAO ON)
