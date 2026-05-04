# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Portable Backend Runtime
# Architecture:
# - runtime/PortableBackend.cpp: BackendInterface (init/execute/destroy)
# - runtime/GraphRuntime.h: Abstract interface for hardware runtimes
# - runtime/OpRegistry.h: Generic op registration (template-based, backend-agnostic)
# - runtime/cpu/CpuRuntime.cpp: CPU implementation of GraphRuntime
# - runtime_v2/cpu/CpuOps.cpp: CPU op implementations using portable kernels
# - runtime/metal/MetalRuntime.mm: Metal GPU implementation (Apple only)
# - runtime/metal_v2/: High-performance Metal with ICB replay (Apple only)

# Enable Objective-C++ for .mm files
enable_language(OBJCXX)

# Option to enable metal_v2 (high-performance Metal with ICB, ResidencySet, etc.)
option(EXECUTORCH_PORTABLE_USE_METAL_V2 "Use metal_v2 runtime with Metal 4 features" ON)

set(_portable_backend__srcs
  ${CMAKE_CURRENT_SOURCE_DIR}/runtime/PortableBackend.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/runtime/cpu/CpuRuntime.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/runtime_v2/cpu/CpuOps.cpp
)

# Add Metal backend on Apple platforms
if(APPLE)
  if(EXECUTORCH_PORTABLE_USE_METAL_V2)
    # metal_v2: High-performance runtime with ICB replay, ResidencySet, Heap
    message(STATUS "Portable backend: Using metal_v2 (Metal 4 features enabled)")
    list(APPEND _portable_backend__srcs
      ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_v2/MetalStream.mm
      ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_v2/MetalHeap.mm
      ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_v2/MetalBufferPool.mm
      ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_v2/MetalKernel.mm
      ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_v2/MetalKernelCompiler.mm
      ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_v2/MetalOp.mm
      ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_v2/MetalOpRegistry.mm
      ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_v2/MetalRuntime.mm
      ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_v2/ops/BinaryOps.mm
      ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_v2/ops/UnaryOps.mm
      ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_v2/ops/MatMulOp.mm
      ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_v2/ops/MPSGraphOp.mm
    )
  else()
    # Original Metal runtime
    message(STATUS "Portable backend: Using original Metal runtime")
    list(APPEND _portable_backend__srcs
      ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal/MetalRuntime.mm
      ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal/MetalOp.mm
      ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal/MetalOpRegistry.mm
      ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal/MetalOps.mm
      ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal/ops/ElementwiseOps.mm
      ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal/ops/MatMulOps.mm
    )
  endif()
endif()

add_library(portable_backend ${_portable_backend__srcs})

target_include_directories(portable_backend
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/cpu
)

# Link executorch_core (the minimal runtime) and program_schema (for flatbuffer types)
target_link_libraries(portable_backend
  PRIVATE
  executorch_core
  program_schema
  portable_ops_lib
)

# Link Metal framework on Apple
if(APPLE)
  target_link_libraries(portable_backend
    PRIVATE
    "-framework Metal"
    "-framework Foundation"
    "-framework MetalPerformanceShaders"
    "-framework MetalPerformanceShadersGraph"
  )

  if(EXECUTORCH_PORTABLE_USE_METAL_V2)
    target_compile_definitions(portable_backend PRIVATE PORTABLE_HAS_METAL_V2=1)
  else()
    target_compile_definitions(portable_backend PRIVATE PORTABLE_HAS_METAL=1)
  endif()

  # Set .mm files to compile as Objective-C++ without ARC (code manages memory manually)
  set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_v2/MetalStream.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_v2/MetalHeap.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_v2/MetalBufferPool.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_v2/MetalKernel.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_v2/MetalKernelCompiler.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_v2/MetalOp.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_v2/MetalOpRegistry.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_v2/MetalRuntime.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_v2/ops/BinaryOps.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_v2/ops/UnaryOps.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_v2/ops/MatMulOp.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_v2/ops/MPSGraphOp.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal/MetalRuntime.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal/MetalOp.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal/MetalOpRegistry.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal/MetalOps.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal/ops/ElementwiseOps.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal/ops/MatMulOps.mm
    PROPERTIES
    LANGUAGE OBJCXX
    COMPILE_FLAGS "-fno-objc-arc"
  )
endif()

# Ensure schema is generated before compiling
add_dependencies(portable_backend program_schema)

install(
  TARGETS portable_backend
  EXPORT ExecuTorchTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# v2 portable backend (new architecture per
# runtime/PORTABLE_BACKEND_API_PROPOSAL.md). Registers as
# "PortableBackend_v2" so it coexists with v1.
add_subdirectory(runtime_v2)
