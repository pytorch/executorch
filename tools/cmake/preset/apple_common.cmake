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
# MLX is Apple Silicon only and needs the Metal compiler (xcrun -sdk macosx
# metal), which ships with Xcode, not the Command Line Tools. Probe for it and
# degrade gracefully rather than force MLX on for every Apple preset, which
# would hard-fail a plain `cmake --preset ios` on a machine without the Metal
# toolchain or the MLX submodule. This mirrors how the wheel's pybind preset
# gates MLX.
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
  execute_process(
    COMMAND xcrun -sdk macosx --find metal
    RESULT_VARIABLE _metal_compiler_result
    OUTPUT_QUIET ERROR_QUIET
  )
  if(_metal_compiler_result EQUAL 0)
    set_overridable_option(EXECUTORCH_BUILD_MLX ON)
  else()
    message(
      STATUS
        "Metal compiler not found, disabling MLX backend. Install Xcode to enable MLX."
    )
  endif()
endif()
set_overridable_option(EXECUTORCH_XNNPACK_ENABLE_WEIGHT_CACHE ON)
set_overridable_option(EXECUTORCH_XNNPACK_SHARED_WORKSPACE ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_APPLE ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_IMAGE ON)
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
