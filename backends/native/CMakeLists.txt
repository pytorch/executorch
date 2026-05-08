# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# PortableBackend_v2 — new architecture per PORTABLE_BACKEND_API_PROPOSAL.md
#
# Layout: api/             — pure C++ headers + RuntimeRegistry.cpp cpu/ —
# CpuRuntime, CpuEngine, HostBuffer, CpuEvent metal/           —
# MetalRuntime, MetalEngine, MetalBuffer, MetalEvent (Apple only; reuses
# ../runtime/metal_v2 sources without depending on the dead v1 MetalRuntime
# adapter) routers/         — GreedyRouter NativeBackend.cpp —
# BackendInterface adapter; registers as "PortableBackend_v2" so it coexists
# with the existing "PortableBackend".
#
# Reuses the existing portable runtime's: - Graph / OperatorCall
# (api/GraphTypes.h) - cpu_op_registry()    (../runtime_v2/cpu/CpuOpRegistry.h)
# So this target depends on portable_backend (existing) for those.

set(_portable_backend_v2__srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/NativeBackend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/RuntimeRegistry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtimes/cpu/CpuEngine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtimes/cpu/CpuRuntime.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtimes/cpu/CpuOps.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtimes/host_pool/HostPoolEngine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/routers/GreedyRouter.cpp
)

# Apple-only Metal provider. Depends on the metal_v2 lib (defined in
# backends/metal/CMakeLists.txt) for the underlying runtime types.
if(APPLE)
  enable_language(OBJCXX)
  list(
    APPEND
    _portable_backend_v2__srcs
    # New runtime_v2/metal sources — native_backend's own metal provider
    ${CMAKE_CURRENT_SOURCE_DIR}/runtimes/metal/MetalBuffer.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/runtimes/metal/MetalRuntime.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/runtimes/metal/MetalEngine.mm
  )
endif()

add_library(native_backend ${_portable_backend_v2__srcs})

# C++17 needed for std::variant / std::optional / structured bindings.
target_compile_features(native_backend PRIVATE cxx_std_17)

# We #include from runtime/ for Graph and cpu_op_registry — needs the
# runtime_v2/ root on the include path so our own headers resolve, and the
# parent (backends/portable/) for the existing runtime/ tree.
target_include_directories(
  native_backend
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/.. # so #include "runtime_v2/api/..."
                                         # works
          ${CMAKE_CURRENT_SOURCE_DIR}/.. # so #include
                                         # "runtime_v2/api/GraphTypes.h" works
)

# Link executorch_core (the minimal runtime) and program_schema (for
# flatbuffer types). portable_ops_lib provides the underlying aten::*.out
# kernel implementations that the CPU dispatcher routes to via its
# default handler.
target_link_libraries(
  native_backend PRIVATE executorch_core program_schema portable_ops_lib
)

if(APPLE)
  # metal_v2 lib provides MetalStream, BufferRegistry, ResidencyManager,
  # MetalMTL{3,4}Backend, all metal ops, etc. — with matching ARC/MRC
  # per-file flags and PUBLIC ET_METAL4_ENABLE / ET_METAL_USE_MPSGRAPH
  # compile defs. Its PUBLIC link of executorch_core + Metal frameworks
  # propagates to us.
  target_link_libraries(
    native_backend
    PRIVATE metal_v2
  )
  # Tell NativeBackend.cpp it can register MetalRuntime.
  target_compile_definitions(
    native_backend PRIVATE NATIVE_HAS_METAL=1
  )
  # Mirror backends/apple/metal/CMakeLists.txt — enable TF32 by default for
  # fp32 NAX matmul / SDPA. Runtime env MLX_ENABLE_TF32 still overrides.
  if(NOT DEFINED EXECUTORCH_METAL_ENABLE_TF32_DEFAULT)
    option(EXECUTORCH_METAL_ENABLE_TF32_DEFAULT
           "Default fp32 NAX matmul/SDPA to TF32 (truncated mantissa) — runtime env MLX_ENABLE_TF32 still overrides"
           ON
    )
  endif()
  if(EXECUTORCH_METAL_ENABLE_TF32_DEFAULT)
    message(STATUS "native_backend: TF32 default ON (fp32 NAX matmul/SDPA enabled by default; override at runtime via MLX_ENABLE_TF32=0)")
    target_compile_definitions(
      native_backend PRIVATE EXECUTORCH_METAL_TF32_DEFAULT_ENABLED=1
    )
  endif()
  # native_backend's own runtimes/metal/*.mm files compile without ARC
  # (memory managed manually). The metal_v2 lib handles its own files.
  set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/runtimes/metal/MetalBuffer.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/runtimes/metal/MetalRuntime.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/runtimes/metal/MetalEngine.mm
    PROPERTIES COMPILE_FLAGS "-fno-objc-arc"
  )
endif()

# Ensure schema is generated first.
add_dependencies(native_backend program_schema)

# Force-link static-init for backend registration. Without this, the static auto
# _register = register_backend(...) in NativeBackend.cpp may be stripped by
# the linker.
if(APPLE)
  set_target_properties(
    native_backend PROPERTIES LINK_FLAGS "-Wl,-force_load"
  )
elseif(UNIX)
  # GCC / Clang on Linux: --whole-archive is applied at link site by the
  # consuming target, not here. Document this in the install/usage notes.
endif()

install(
  TARGETS native_backend
  EXPORT ExecuTorchTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# ======================================================================
# Integration test: batch-varying execute() calls
# ======================================================================
# Standalone binary that loads /tmp/dyn_linear_v2.pte and runs forward() with
# batch sizes {1, 3, 5, 8} against the same loaded delegate, verifying true
# runtime-varying dynamic-shape behavior.
add_executable(
  test_dyn_shapes_native ${CMAKE_CURRENT_SOURCE_DIR}/test/test_dyn_shapes.cpp
)
target_compile_features(test_dyn_shapes_native PRIVATE cxx_std_17)
target_link_libraries(
  test_dyn_shapes_native
  PRIVATE executorch
          extension_module
          extension_tensor
          extension_data_loader
          extension_flat_tensor
          portable_kernels
          portable_ops_lib
          "$<LINK_LIBRARY:WHOLE_ARCHIVE,native_backend>"
)
if(APPLE)
  target_link_libraries(
    test_dyn_shapes_native
    PRIVATE "-framework Metal" "-framework Foundation"
            "-framework MetalPerformanceShaders"
            "-framework MetalPerformanceShadersGraph"
  )
endif()

# Stateful (mutable buffer) test — observes how mutable buffers flow through the
# v2 runtime today.
add_executable(test_stateful_native ${CMAKE_CURRENT_SOURCE_DIR}/test/test_stateful.cpp)
target_compile_features(test_stateful_native PRIVATE cxx_std_17)
target_link_libraries(
  test_stateful_native
  PRIVATE executorch
          extension_module
          extension_tensor
          extension_data_loader
          extension_flat_tensor
          portable_kernels
          portable_ops_lib
          "$<LINK_LIBRARY:WHOLE_ARCHIVE,native_backend>"
)
if(APPLE)
  target_link_libraries(
    test_stateful_native
    PRIVATE "-framework Metal" "-framework Foundation"
            "-framework MetalPerformanceShaders"
            "-framework MetalPerformanceShadersGraph"
  )
endif()

# Control-flow integration test (cond). Drives PortableBackend_v2's
# BackendInterface directly with a torch.cond .pte; verifies both branches
# execute correctly and produce expected outputs. See CONTROL_FLOW_DESIGN.md.
add_executable(test_cond_native ${CMAKE_CURRENT_SOURCE_DIR}/test/test_cond.cpp)
target_compile_features(test_cond_native PRIVATE cxx_std_17)
target_link_libraries(
  test_cond_native
  PRIVATE executorch
          extension_module
          extension_tensor
          extension_data_loader
          extension_flat_tensor
          portable_kernels
          portable_ops_lib
          "$<LINK_LIBRARY:WHOLE_ARCHIVE,native_backend>"
)
if(APPLE)
  target_link_libraries(
    test_cond_native
    PRIVATE "-framework Metal" "-framework Foundation"
            "-framework MetalPerformanceShaders"
            "-framework MetalPerformanceShadersGraph"
  )
endif()

# In-place op integration test. Loads /tmp/inplace_*.pte exported by
# test/test_export_inplace.py, drives NativeBackend with each, and
# verifies element-wise output. Exercises the full_name dispatch path
# for in-place ops (relu_, sigmoid_, add_.Tensor, mul_.Tensor, clamp_).
add_executable(test_inplace_native ${CMAKE_CURRENT_SOURCE_DIR}/test/test_inplace.cpp)
target_compile_features(test_inplace_native PRIVATE cxx_std_17)
target_link_libraries(
  test_inplace_native
  PRIVATE executorch
          extension_module
          extension_tensor
          extension_data_loader
          extension_flat_tensor
          portable_kernels
          portable_ops_lib
          "$<LINK_LIBRARY:WHOLE_ARCHIVE,native_backend>"
)
if(APPLE)
  target_link_libraries(
    test_inplace_native
    PRIVATE "-framework Metal" "-framework Foundation"
            "-framework MetalPerformanceShaders"
            "-framework MetalPerformanceShadersGraph"
  )
endif()
