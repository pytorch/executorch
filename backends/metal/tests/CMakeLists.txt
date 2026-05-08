# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Structural tests for the metal_v2 backend.
#
# These tests link against the `metal_v2` static lib (defined in the
# parent CMakeLists.txt) so they inherit its PUBLIC compile flags
# (ET_METAL4_ENABLE=1, ET_METAL_USE_MPSGRAPH=0, cxx_std_20) and see the
# same class layouts as the production lib. This eliminates the ODR
# issues that previously bit when each test target tried to set its own
# preprocessor flags while linking against an unrelated lib.
#

cmake_minimum_required(VERSION 3.19)

if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../../..)
endif()

include(${EXECUTORCH_ROOT}/tools/cmake/Test.cmake)

# test_buffer_registry — pure unit tests on BufferRegistry.
et_cxx_test(
  metal_v2_test_buffer_registry
  SOURCES
    test_buffer_registry.mm
  EXTRA_LIBS
    metal_v2
)

# test_residency_manager — pure unit tests on ResidencyManager (refcounted
# pin/unpin, batched variants, pinHeap/unpinHeap, addQueueResidency/
# removeQueueResidency symmetry, stats counters, dirty-bit commit
# optimization, OS-availability gating). Tests run on macOS 15+ / iOS 18+
# where MTLResidencySet is available; on older OSes, refcount-based
# tests GTEST_SKIP via isEnabled() while no-op safety tests still run.
et_cxx_test(
  metal_v2_test_residency_manager
  SOURCES
    test_residency_manager.mm
  EXTRA_LIBS
    metal_v2
)

# test_mtl4_backend — MTL4 dispatch path smoke test. Links the metal_v2
# lib (which provides MetalStream + the MTL3/MTL4 backend implementations
# with matching compile flags).
et_cxx_test(
  metal_v2_test_mtl4_backend
  SOURCES
    test_mtl4_backend.mm
  EXTRA_LIBS
    metal_v2
    portable_kernels
    portable_ops_lib
)

# test_stream_lifecycle — MetalStream alloc/free/sync API surface tests.
et_cxx_test(
  metal_v2_test_stream_lifecycle
  SOURCES
    test_stream_lifecycle.mm
  EXTRA_LIBS
    metal_v2
    portable_kernels
    portable_ops_lib
)

# test_mps_interop — Unit 7 side-door binding contract end-to-end. The
# entire body is gated on #if ET_METAL_USE_MPSGRAPH, so the mtl4 build
# (where MPSGraph is disabled) compiles a trivial gtest_main with no
# test cases (clean run).
et_cxx_test(
  metal_v2_test_mps_interop
  SOURCES
    test_mps_interop.mm
  EXTRA_LIBS
    metal_v2
    portable_kernels
    portable_ops_lib
)

# test_metal_allocator — covers Units 6/9/10: alloc/free origins,
# ResidencyClass, unregisterExternalPermanent, subregions, heap pin.
et_cxx_test(
  metal_v2_test_metal_allocator
  SOURCES
    test_metal_allocator.mm
  EXTRA_LIBS
    metal_v2
)

# test_metal_buffer_pool — covers Unit 5 + this session's pool retain
# regression fix: acquire/release identity reuse, eviction, prewarm,
# pool retains its own cached buffers.
et_cxx_test(
  metal_v2_test_metal_buffer_pool
  SOURCES
    test_metal_buffer_pool.mm
  EXTRA_LIBS
    metal_v2
)

# test_hazard_tracker — direct HazardTracker coverage (RAW/WAW/WAR
# detection, range overlap, reset, external writes, barrier stats).
et_cxx_test(
  metal_v2_test_hazard_tracker
  SOURCES
    test_hazard_tracker.mm
  EXTRA_LIBS
    metal_v2
)

# test_metal_command_recorder — recorder API (declareSideDoorBinds dedup,
# noteMpsWorkPending, setFlushInterval). Uses MetalStream to construct
# the full backend stack.
et_cxx_test(
  metal_v2_test_metal_command_recorder
  SOURCES
    test_metal_command_recorder.mm
  EXTRA_LIBS
    metal_v2
    portable_kernels
    portable_ops_lib
)

# test_metal_kernel_cache — process-wide singleton kernel/library/PSO cache.
et_cxx_test(
  metal_v2_test_metal_kernel_cache
  SOURCES
    test_metal_kernel_cache.mm
  EXTRA_LIBS
    metal_v2
)

# test_metal_heap — heap allocation, exhaustion, reset, native accessor.
et_cxx_test(
  metal_v2_test_metal_heap
  SOURCES
    test_metal_heap.mm
  EXTRA_LIBS
    metal_v2
)

# test_metal_op_registry — process-wide singleton op registry.
et_cxx_test(
  metal_v2_test_metal_op_registry
  SOURCES
    test_metal_op_registry.mm
  EXTRA_LIBS
    metal_v2
    portable_kernels
    portable_ops_lib
)

# test_metal_device_info — device tier / family heuristics + cached singleton.
et_cxx_test(
  metal_v2_test_metal_device_info
  SOURCES
    test_metal_device_info.mm
  EXTRA_LIBS
    metal_v2
)

# test_metal_kernel — slot-access state machine + name/pipeline accessors.
et_cxx_test(
  metal_v2_test_metal_kernel
  SOURCES
    test_metal_kernel.mm
  EXTRA_LIBS
    metal_v2
)

# test_metal_kernel_compiler — MSL → PSO compilation, FunctionConstants
# fingerprint uniqueness, binary archive accessors.
et_cxx_test(
  metal_v2_test_metal_kernel_compiler
  SOURCES
    test_metal_kernel_compiler.mm
  EXTRA_LIBS
    metal_v2
)

# test_concurrency — race N threads on documented-thread-safe components
# (KernelCache::findOrInsert, ResidencyManager pin/unpin, BufferRegistry).
et_cxx_test(
  metal_v2_test_concurrency
  SOURCES
    test_concurrency.mm
  EXTRA_LIBS
    metal_v2
)

# test_mtl3_backend — MTL3 dispatch path smoke + completion handler
# tests. Forces MTL3 via env var; runs in both build configs.
et_cxx_test(
  metal_v2_test_mtl3_backend
  SOURCES
    test_mtl3_backend.mm
  EXTRA_LIBS
    metal_v2
    portable_kernels
    portable_ops_lib
)

# test_typed_setters — exercises setInput/setOutput/setBytes/setVectorBytes/
# setBuffer/setKernel/dispatchThreadgroups end-to-end with a trivial kernel.
et_cxx_test(
  metal_v2_test_typed_setters
  SOURCES
    test_typed_setters.mm
  EXTRA_LIBS
    metal_v2
    portable_kernels
    portable_ops_lib
)
