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
# MLX needs the Metal compiler (xcrun -sdk macosx metal), which ships with
# Xcode, not the Command Line Tools. Probe for it and degrade gracefully rather
# than force MLX on for every Apple preset, which would hard-fail a plain `cmake
# --preset ios` on a machine without the Metal toolchain or the MLX submodule.
# This mirrors how the wheel's pybind preset gates MLX. The metal probe is the
# whole gate: do not add a CMAKE_SYSTEM_PROCESSOR check here, because under the
# Apple presets that variable is the target the toolchain set (aarch64, never
# arm64), so such a check silently disables MLX on every Apple build.
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
set_overridable_option(EXECUTORCH_XNNPACK_ENABLE_WEIGHT_CACHE ON)
set_overridable_option(EXECUTORCH_XNNPACK_SHARED_WORKSPACE ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_APPLE ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER ON)
# The profiler records events through the runtime's event tracer, which is a
# compile-time choice for the whole runtime, so the tracer stays on here. The
# etdump C++ library the wrapper links is pulled in on its own by the condition in
# the root CMakeLists that builds devtools for this case, so the full devtools
# umbrella is deliberately NOT turned on: that umbrella also flips the Core ML
# delegate into its protobuf path, which the shipped Core ML framework does not
# bundle. The wheel's pybind preset configures profiling the same way.
set_overridable_option(EXECUTORCH_ENABLE_EVENT_TRACER ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_ETDUMP_APPLE ON)
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
