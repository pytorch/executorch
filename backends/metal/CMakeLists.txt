# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# backends/metal — high-performance Metal runtime for the executorch
# portable / native / AOTI v2 backends.
#
# Owns its own preprocessor flags so consumers (AOTI v2 shim, native
# runtime, unit tests, any other downstream) compile against the SAME
# class layouts. Avoids the ODR violations that arise when one consumer
# compiles MetalStream.h with ET_METAL4_ENABLE=1 while another sees
# the default ET_METAL4_ENABLE=0.
#
# Note: namespace stays `executorch::backends::metal_v2` even though the
# directory is `backends/metal/` — the AOTI shim layer at
# `backends/apple/metal/` already occupies the `executorch::backends::metal`
# namespace, so we keep `metal_v2` for the runtime library.
#

cmake_minimum_required(VERSION 3.19)

set(_metal_dir ${CMAKE_CURRENT_SOURCE_DIR})

# Files that compile WITH ARC (-fobjc-arc). No explicit [retain]/[release]
# in source; ARC manages Obj-C lifetimes.
set(_metal_arc_srcs
    ${_metal_dir}/core/HazardTracker.mm
    ${_metal_dir}/core/BufferRegistry.mm
    ${_metal_dir}/core/MetalAllocator.mm
    ${_metal_dir}/core/MetalBufferPool.mm
    ${_metal_dir}/core/MetalCommandRecorder.mm
    ${_metal_dir}/core/MetalDeviceInfo.mm
    ${_metal_dir}/core/MetalHeap.mm
    ${_metal_dir}/core/MetalKernel.mm
    ${_metal_dir}/core/MetalKernelCache.mm
    ${_metal_dir}/core/MetalKernelCompiler.mm
    ${_metal_dir}/core/MetalMTL3Backend.mm
    ${_metal_dir}/core/MetalMTL4Backend.mm
    ${_metal_dir}/core/MetalStream.mm
    ${_metal_dir}/core/MpsInterop.mm
    ${_metal_dir}/core/ResidencyManager.mm
    ${_metal_dir}/ops/registry/MetalOp.mm
    ${_metal_dir}/ops/registry/MetalOpRegistry.mm
    ${_metal_dir}/ops/MPSGraphOp.mm
    ${_metal_dir}/ops/AddMMOp.mm
    ${_metal_dir}/ops/AffineQuantizedLinearOp.mm
    ${_metal_dir}/ops/BAddBMMOp.mm
    ${_metal_dir}/ops/BatchedMatMulOp.mm
    ${_metal_dir}/ops/BinaryOps.mm
    ${_metal_dir}/ops/MatMulOp.mm
    ${_metal_dir}/ops/SDPAOp.mm
    ${_metal_dir}/ops/UnaryOps.mm
)

# Files that compile WITHOUT ARC (-fno-objc-arc). These use explicit
# [retain] / [release] which ARC forbids. KernelLoader.mm follows the
# established MRC convention from MlxJit.cmake.
set(_metal_mrc_srcs
)
# NOTE: KernelLoader.mm is appended below via add_metal_v2_mlx_jit_sources.

set(_metal_srcs ${_metal_arc_srcs} ${_metal_mrc_srcs})

# Generate MLX JIT snippet .cpp files (consumed by KernelLoader.mm) and
# append them + KernelLoader.mm to the source list. The function uses
# add_custom_command to wire snippet generation as build-time deps.
include(${_metal_dir}/MlxJit.cmake)
add_metal_v2_mlx_jit_sources(_metal_srcs)

add_library(metal_v2 STATIC ${_metal_srcs})

# Per-file ARC/MRC flags. .mm extension auto-triggers ObjC++ compilation;
# we don't set LANGUAGE OBJCXX because that requires the parent
# CMakeLists to enable_language(OBJCXX).
set_source_files_properties(
  ${_metal_arc_srcs}
  PROPERTIES COMPILE_FLAGS "-fobjc-arc"
)
set_source_files_properties(
  ${_metal_mrc_srcs}
  PROPERTIES COMPILE_FLAGS "-fno-objc-arc"
)

# C++20 needed for std::span used internally by ResidencyManager.mm /
# MetalCommandRecorder.mm.
target_compile_features(metal_v2 PUBLIC cxx_std_20)

# PUBLIC compile definitions: every consumer sees the same
# ET_METAL4_ENABLE / ET_METAL_USE_MPSGRAPH preprocessor state, fixing
# ODR violations where consumers compiled MetalStream.h with
# different conditional-region selections.
#
# METAL_V2_USE_MTL4 selects between the MTL4 build (default; uses
# Metal 4 / MTL4 queue / residency set) and the MTL3 + MPSGraph build
# (lazy paging, MPSGraph fallback for unimplemented ops). The two are
# mutually exclusive (per MpsInterop.h's #error gate).
#
# Test fixtures should run under BOTH configurations — the design's
# Test Plan calls out: "All tests run under both MetalMTL3Backend and
# MetalMTL4Backend test fixtures — the single-set design is backend-
# uniform; behavior must be identical." See
# backends/metal/tests/build_and_run_tests.sh which configures two
# build dirs (one per option) and runs each.
option(METAL_V2_USE_MTL4
       "Build metal_v2 lib with MTL4 (else MTL3 + MPSGraph)"
       ON)
if(METAL_V2_USE_MTL4)
  message(STATUS "metal_v2: building with MTL4 (ET_METAL4_ENABLE=1, ET_METAL_USE_MPSGRAPH=0)")
  target_compile_definitions(
    metal_v2
    PUBLIC
      ET_METAL4_ENABLE=1
      ET_METAL_USE_MPSGRAPH=0
  )
else()
  message(STATUS "metal_v2: building with MTL3 + MPSGraph (ET_METAL4_ENABLE=0, ET_METAL_USE_MPSGRAPH=1)")
  target_compile_definitions(
    metal_v2
    PUBLIC
      ET_METAL4_ENABLE=0
      ET_METAL_USE_MPSGRAPH=1
  )
endif()

# Public headers root: consumers can `#include
# <executorch/backends/metal/...>`. EXECUTORCH_ROOT is
# `<repo>/third-party/executorch`, so the parent directory is what the
# `<executorch/...>` include prefix resolves against.
target_include_directories(
  metal_v2
  PUBLIC
    $<BUILD_INTERFACE:${EXECUTORCH_ROOT}/..>
    $<INSTALL_INTERFACE:include>
)

# executorch_core provides PyTorch's c10 headers transitively (array_ref.h
# includes <c10/util/irange.h>) plus the ExecuTorch runtime types
# (Tensor, Error, ArrayRef) that metal .mm files use throughout. Apple
# frameworks PUBLIC so downstream consumers (tests, AOTI shim, native
# runtime) pick them up.
target_link_libraries(
  metal_v2
  PUBLIC
    executorch_core
    program_schema
    "-framework Metal"
    "-framework Foundation"
    "-framework MetalPerformanceShaders"
    "-framework MetalPerformanceShadersGraph"
)

# program_schema generates flatbuffer headers used transitively by
# executorch_core. Make sure it's built before metal_v2 sources.
add_dependencies(metal_v2 program_schema)

# Add the build-dir include path for the generated schema headers (e.g.
# executorch/schema/program_generated.h, generated by program_schema
# under ${CMAKE_BINARY_DIR}/schema/include/executorch/schema/).
target_include_directories(
  metal_v2
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/schema/include>
)

# Tests live in tests/ and link against metal_v2 (so they inherit the
# matching PUBLIC compile flags).
if(EXECUTORCH_BUILD_TESTS)
  add_subdirectory(tests)
endif()

# Add metal_v2 to the ExecuTorchTargets export set so backends that
# depend on it (e.g. metal_backend in backends/apple/metal under
# EXECUTORCH_USE_METAL_V2=ON) can be installed without cmake erroring
# out with "metal_v2 not in any export set".
install(
  TARGETS metal_v2
  EXPORT ExecuTorchTargets
  ARCHIVE DESTINATION lib
)
