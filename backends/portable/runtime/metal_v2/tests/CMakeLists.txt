# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Structural tests for the metal_v2 backend.
#
# These tests exercise the C++ API directly (no PTE / AOTI / op layer) to
# catch bug classes that numeric-correctness tests can't:
#
#   - test_buffer_registry  : pure unit tests on BufferRegistry (Origin
#                             routing, refresh-on-hit, ownership transfer
#                             via remove(), iterate-while-empty safety).
#                             The bug class this catches: free() check-
#                             after-erase corruption (audit Finding C1).
#
# Future tests (R3 follow-ups, after R6.2b lands):
#   - test_dispatch_lifetime: stack-local setBytes / setVectorBytes safety with N
#                             consecutive dispatches at the same stack frame.
#   - test_multi_execute    : dispatch -> flush -> wait cycles with shape
#                             changes across iterations.
#   - test_state_machine    : invalid transition assertions (after R7).
#   - test_backend_parity   : same kernel via MTL3Backend vs MTL4Backend;
#                             bit-identical outputs (after R6.2b wires up
#                             the backend abstraction).
#

cmake_minimum_required(VERSION 3.19)

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../../..)

include(${EXECUTORCH_ROOT}/tools/cmake/Test.cmake)

set(_metal_v2_dir ${CMAKE_CURRENT_SOURCE_DIR}/..)

# test_buffer_registry — pure unit tests; no GPU dispatch beyond MTLBuffer
# allocations on the default device. Compiles BufferRegistry.mm directly into
# the test binary so we don't drag in the full metal_backend dependency.
et_cxx_test(
  metal_v2_test_buffer_registry
  SOURCES
    test_buffer_registry.mm
    ${_metal_v2_dir}/BufferRegistry.mm
  EXTRA_LIBS
)

if(APPLE)
  target_link_libraries(
    metal_v2_test_buffer_registry
    "-framework Metal" "-framework Foundation"
  )
endif()

# test_mtl4_backend — MTL4 dispatch path smoke test. Links the full
# metal_backend so we can construct a real MetalStream (which exercises
# setupMTL4 / teardownMTL4 / MTL4 queue / allocator init when the build
# was configured with EXECUTORCH_METAL4_ENABLE=ON).
#
# When MTL4 is OFF (the default), all test cases GTEST_SKIP at SetUp.
# When MTL4 is ON but OS pre-dates macOS 26 / iOS 26, runtime @available
# check skips. Either way, the test binary builds and runs cleanly —
# results just become "5 SKIPPED, 0 FAILED" instead of "5 PASSED".
#
# To exercise this end-to-end:
#   cmake cmake-out -DEXECUTORCH_METAL4_ENABLE=ON -DEXECUTORCH_BUILD_TESTS=ON
#   cmake --build cmake-out --target metal_v2_test_mtl4_backend
#   ./cmake-out/.../metal_v2_test_mtl4_backend
et_cxx_test(
  metal_v2_test_mtl4_backend
  SOURCES
    test_mtl4_backend.mm
  EXTRA_LIBS
    metal_backend
    portable_kernels
    portable_ops_lib
)

if(APPLE)
  target_link_libraries(
    metal_v2_test_mtl4_backend
    "-framework Metal" "-framework Foundation"
    "-framework MetalPerformanceShaders"
    "-framework MetalPerformanceShadersGraph"
  )
  if(EXECUTORCH_METAL4_ENABLE)
    target_compile_definitions(
      metal_v2_test_mtl4_backend
      PRIVATE ET_METAL4_ENABLE=1
    )
  endif()
endif()

# test_stream_lifecycle (R3 follow-up D) — MetalStream alloc/free/sync API
# surface tests. Uses the real MetalStream (which constructs the full
# backend stack including BufferRegistry and the MTL3 backend), but does
# NOT dispatch any kernels. Catches lifecycle / refcount / registry-
# integration bugs in milliseconds, without the heavyweight PTE roundtrip
# that test_dyn_shapes_v2 needs.
et_cxx_test(
  metal_v2_test_stream_lifecycle
  SOURCES
    test_stream_lifecycle.mm
  EXTRA_LIBS
    metal_backend
    portable_kernels
    portable_ops_lib
)

if(APPLE)
  target_link_libraries(
    metal_v2_test_stream_lifecycle
    "-framework Metal" "-framework Foundation"
    "-framework MetalPerformanceShaders"
    "-framework MetalPerformanceShadersGraph"
  )
  if(EXECUTORCH_METAL4_ENABLE)
    target_compile_definitions(
      metal_v2_test_stream_lifecycle
      PRIVATE ET_METAL4_ENABLE=1
    )
  endif()
endif()

# test_typed_setters (R2.1 follow-up) — exercises the new setInput /
# setOutput / setBytes / setVectorBytes / setBuffer / setKernel /
# dispatchThreadgroups API end-to-end with an inline-compiled trivial
# kernel. Without this test, the typed-setter API has zero runtime
# coverage (the legacy 12-case suite all goes through dispatch(args)).
et_cxx_test(
  metal_v2_test_typed_setters
  SOURCES
    test_typed_setters.mm
  EXTRA_LIBS
    metal_backend
    portable_kernels
    portable_ops_lib
)

if(APPLE)
  target_link_libraries(
    metal_v2_test_typed_setters
    "-framework Metal" "-framework Foundation"
    "-framework MetalPerformanceShaders"
    "-framework MetalPerformanceShadersGraph"
  )
  if(EXECUTORCH_METAL4_ENABLE)
    target_compile_definitions(
      metal_v2_test_typed_setters
      PRIVATE ET_METAL4_ENABLE=1
    )
  endif()
endif()
