# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# PortableBackend_v2 — new architecture per PORTABLE_BACKEND_API_PROPOSAL.md
#
# Layout:
#   api/             — pure C++ headers + ProviderRegistry.cpp
#   cpu/             — CpuProvider, CpuInstance, HostBuffer, CpuEvent
#   metal/           — MetalProvider, MetalInstance, MetalBuffer, MetalEvent
#                      (Apple only; reuses ../runtime/metal_v2 sources without
#                       depending on the dead v1 MetalRuntime adapter)
#   routers/         — GreedyRouter
#   PortableBackend_v2.cpp — BackendInterface adapter; registers as
#                            "PortableBackend_v2" so it coexists with
#                            the existing "PortableBackend".
#
# Reuses the existing portable runtime's:
#   - Graph / OperatorCall (api/GraphTypes.h)
#   - cpu_op_registry()    (../runtime_v2/cpu/CpuOpRegistry.h)
# So this target depends on portable_backend (existing) for those.

set(_portable_backend_v2__srcs
  ${CMAKE_CURRENT_SOURCE_DIR}/PortableBackend_v2.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/api/ProviderRegistry.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpu/CpuInstance.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpu/CpuProvider.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/routers/GreedyRouter.cpp
)

# Apple-only Metal provider. Reuses the existing metal_v2 .mm sources
# without copying — we just compile them into this library directly.
# (We don't depend on portable_backend's metal_v2 wiring because that's
# behind EXECUTORCH_PORTABLE_USE_METAL_V2 and is currently broken on a
# stale GpuStream.h import in MetalRuntime.mm — which we don't use.)
if(APPLE)
  enable_language(OBJCXX)
  set(_metal_v2_dir ${CMAKE_CURRENT_SOURCE_DIR}/../runtime/metal_v2)
  list(APPEND _portable_backend_v2__srcs
    # New runtime_v2/metal sources
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/MetalBuffer.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/MetalProvider.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/MetalInstance.mm
    # Reused metal_v2 sources (compiled once into this lib)
    ${_metal_v2_dir}/MetalStream.mm
    ${_metal_v2_dir}/MetalKernel.mm
    ${_metal_v2_dir}/MetalKernelCompiler.mm
    ${_metal_v2_dir}/MetalBufferPool.mm
    ${_metal_v2_dir}/MetalHeap.mm
    ${_metal_v2_dir}/MetalOp.mm
    ${_metal_v2_dir}/MetalOpRegistry.mm
    ${_metal_v2_dir}/ops/BinaryOps.mm
    ${_metal_v2_dir}/ops/UnaryOps.mm
    ${_metal_v2_dir}/ops/MatMulOp.mm
    ${_metal_v2_dir}/ops/MPSGraphOp.mm
  )
endif()

add_library(portable_backend_v2 ${_portable_backend_v2__srcs})

# C++17 needed for std::variant / std::optional / structured bindings.
target_compile_features(portable_backend_v2 PRIVATE cxx_std_17)

# We #include from runtime/ for Graph and cpu_op_registry — needs the
# runtime_v2/ root on the include path so our own headers resolve, and
# the parent (backends/portable/) for the existing runtime/ tree.
target_include_directories(portable_backend_v2
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..        # so #include "runtime_v2/api/..." works
    ${CMAKE_CURRENT_SOURCE_DIR}/.. # so #include "runtime_v2/api/GraphTypes.h" works
)

# Link the v1 portable_backend so we can reuse its Graph types and
# CpuOpRegistry — note we depend on the *library*, not just headers,
# because cpu_op_registry() is defined in CpuOps.cpp.
target_link_libraries(portable_backend_v2
  PRIVATE
    executorch_core
    program_schema
    portable_ops_lib
    portable_backend
)

if(APPLE)
  # Metal frameworks for the v2 metal provider + reused metal_v2 sources.
  target_link_libraries(portable_backend_v2
    PRIVATE
      "-framework Metal"
      "-framework Foundation"
      "-framework MetalPerformanceShaders"
      "-framework MetalPerformanceShadersGraph"
  )
  # Tell PortableBackend_v2.cpp it can register MetalProvider.
  target_compile_definitions(portable_backend_v2 PRIVATE PORTABLE_V2_HAS_METAL=1)
  # All .mm files compile as Objective-C++ without ARC (memory managed
  # manually; matches the existing metal_v2 build settings in
  # backends/portable/CMakeLists.txt).
  set(_objcxx_no_arc_files
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/MetalBuffer.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/MetalProvider.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/MetalInstance.mm
    ${_metal_v2_dir}/MetalStream.mm
    ${_metal_v2_dir}/MetalKernel.mm
    ${_metal_v2_dir}/MetalKernelCompiler.mm
    ${_metal_v2_dir}/MetalBufferPool.mm
    ${_metal_v2_dir}/MetalHeap.mm
    ${_metal_v2_dir}/MetalOp.mm
    ${_metal_v2_dir}/MetalOpRegistry.mm
    ${_metal_v2_dir}/ops/BinaryOps.mm
    ${_metal_v2_dir}/ops/UnaryOps.mm
    ${_metal_v2_dir}/ops/MatMulOp.mm
    ${_metal_v2_dir}/ops/MPSGraphOp.mm
  )
  set_source_files_properties(${_objcxx_no_arc_files}
    PROPERTIES LANGUAGE OBJCXX COMPILE_FLAGS "-fno-objc-arc"
  )
endif()

# Ensure schema is generated first.
add_dependencies(portable_backend_v2 program_schema)

# Force-link static-init for backend registration. Without this, the
# static auto _register = register_backend(...) in PortableBackend_v2.cpp
# may be stripped by the linker.
if(APPLE)
  set_target_properties(portable_backend_v2 PROPERTIES
    LINK_FLAGS "-Wl,-force_load"
  )
elseif(UNIX)
  # GCC / Clang on Linux: --whole-archive is applied at link site by the
  # consuming target, not here. Document this in the install/usage notes.
endif()

install(
  TARGETS portable_backend_v2
  EXPORT ExecuTorchTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

#======================================================================
# Integration test: batch-varying execute() calls
#======================================================================
# Standalone binary that loads /tmp/dyn_linear_v2.pte and runs
# forward() with batch sizes {1, 3, 5, 8} against the same loaded
# delegate, verifying true runtime-varying dynamic-shape behavior.
add_executable(test_dyn_shapes_v2
  ${CMAKE_CURRENT_SOURCE_DIR}/test_dyn_shapes.cpp
)
target_compile_features(test_dyn_shapes_v2 PRIVATE cxx_std_17)
target_link_libraries(test_dyn_shapes_v2
  PRIVATE
    executorch
    extension_module
    extension_tensor
    extension_data_loader
    extension_flat_tensor
    portable_kernels
    portable_ops_lib
    portable_backend
    "$<LINK_LIBRARY:WHOLE_ARCHIVE,portable_backend_v2>"
)
if(APPLE)
  target_link_libraries(test_dyn_shapes_v2
    PRIVATE
      "-framework Metal"
      "-framework Foundation"
      "-framework MetalPerformanceShaders"
      "-framework MetalPerformanceShadersGraph"
  )
endif()

# Stateful (mutable buffer) test — observes how mutable buffers flow
# through the v2 runtime today.
add_executable(test_stateful_v2
  ${CMAKE_CURRENT_SOURCE_DIR}/test_stateful.cpp
)
target_compile_features(test_stateful_v2 PRIVATE cxx_std_17)
target_link_libraries(test_stateful_v2
  PRIVATE
    executorch
    extension_module
    extension_tensor
    extension_data_loader
    extension_flat_tensor
    portable_kernels
    portable_ops_lib
    portable_backend
    "$<LINK_LIBRARY:WHOLE_ARCHIVE,portable_backend_v2>"
)
if(APPLE)
  target_link_libraries(test_stateful_v2
    PRIVATE
      "-framework Metal"
      "-framework Foundation"
      "-framework MetalPerformanceShaders"
      "-framework MetalPerformanceShadersGraph"
  )
endif()
