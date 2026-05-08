# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Build AOTI Metal backend for runtime.
#
# ### Editing this file ###
#
# This file should be formatted with
# ~~~
# cmake-format -i CMakeLists.txt
# ~~~
# It should also be cmake-lint clean.
#
cmake_minimum_required(VERSION 3.29)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT APPLE)
  message(FATAL_ERROR "Metal backend requires macOS")
endif()

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)
# Use full torch package to get library paths, but only link specific libraries
find_package_torch()

set(_aoti_metal_sources
    runtime/stats.cpp
    # v1-only sources (no v2 replacement yet — built in both branches):
    runtime/shims/utils.cpp
)

# AOTI shim layer: choose between v1 (vendored MLX qmv_fast for int4,
# legacy stream/encoding/buffer-mgmt shims) and v2 (routes through
# metal_v2::MetalStream + MetalOpRegistry → AffineQuantizedLinearOp,
# SDPAOp with NAX / qmm_t / qmm_t_splitk dispatch + per-CB residency).
#
# v1 and v2 shim files share extern "C" symbol names (intentionally —
# PTE files lower against the same AOTI shim ABI). They CANNOT coexist
# in the same build; this if/else picks one.
if(EXECUTORCH_USE_METAL_V2)
  list(APPEND _aoti_metal_sources
    runtime/metal_backend_v2.cpp
    runtime/shims/v2/aoti_tensor.cpp
    runtime/shims/v2/aoti_dtype_stubs.cpp
    runtime/shims/v2/runtime.mm
    runtime/shims/v2/aoti_kernel.mm
    runtime/shims/v2/aoti_fallback_op.mm
  )
else()
  list(APPEND _aoti_metal_sources
    runtime/metal_backend.cpp
    runtime/shims/memory.cpp
    runtime/shims/et_metal.mm
    runtime/shims/shim_mps.mm
    runtime/shims/tensor_attribute.cpp
    runtime/ops/common.mm
    runtime/ops/op_bmm.mm
    runtime/ops/op_convolution.mm
    runtime/ops/op_linear_4bit.mm
    runtime/ops/op_mm.mm
    runtime/ops/op_sdpa.mm
  )
endif()

add_library(metal_backend STATIC ${_aoti_metal_sources})
target_include_directories(
  metal_backend
  PUBLIC $<BUILD_INTERFACE:${EXECUTORCH_ROOT}> $<INSTALL_INTERFACE:include>
         # PyTorch AOTI headers from ExecuTorch's torch detection
         ${TORCH_INCLUDE_DIRS}
)

# Link Metal framework
find_library(METAL_LIBRARY Metal REQUIRED)
find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
find_library(METALPERFORMANCESHADERS_LIBRARY MetalPerformanceShaders REQUIRED)
find_library(
  METALPERFORMANCESHADERSGRAPH_LIBRARY MetalPerformanceShadersGraph REQUIRED
)
target_link_libraries(
  metal_backend
  PUBLIC ${METAL_LIBRARY} ${FOUNDATION_LIBRARY}
         ${METALPERFORMANCESHADERS_LIBRARY}
         ${METALPERFORMANCESHADERSGRAPH_LIBRARY}
)

target_compile_options(metal_backend PUBLIC -fexceptions -frtti -fPIC)

# Define C++ preprocessor macro for Metal backend availability
target_compile_definitions(metal_backend PUBLIC ET_BUILD_METAL)

# Define macro for stats collection if enabled
if(EXECUTORCH_METAL_COLLECT_STATS)
  target_compile_definitions(
    metal_backend PUBLIC EXECUTORCH_METAL_COLLECT_STATS
  )
  message(STATUS "Metal backend stats collection enabled")
endif()

target_link_options(metal_backend PUBLIC -Wl,-export_dynamic)

# Under the v2 shim path, metal_backend depends on the metal_v2 lib
# (defined in backends/metal/) for MetalStream, MetalOpRegistry,
# AffineQuantizedLinearOp, etc. PRIVATE link so it doesn't propagate
# through install exports of metal_backend's downstream consumers.
if(EXECUTORCH_USE_METAL_V2)
  target_link_libraries(metal_backend PRIVATE metal_v2)
endif()

# Find PyTorch's OpenMP library specifically for libtorch-less AOTI
get_torch_base_path(TORCH_BASE_PATH)
find_library(
  TORCH_OMP_LIBRARY
  NAMES omp libomp
  PATHS "${TORCH_BASE_PATH}/lib"
  NO_DEFAULT_PATH
)

if(TORCH_OMP_LIBRARY)
  message(STATUS "Found PyTorch OpenMP library: ${TORCH_OMP_LIBRARY}")
  # Get the directory containing the OpenMP library for rpath
  get_filename_component(TORCH_OMP_LIB_DIR ${TORCH_OMP_LIBRARY} DIRECTORY)
  message(STATUS "OpenMP library directory: ${TORCH_OMP_LIB_DIR}")
else()
  message(
    WARNING "PyTorch OpenMP library not found, may cause runtime linking issues"
  )
endif()

# Link against appropriate backends and standard libraries
target_link_libraries(
  metal_backend PUBLIC aoti_common extension_tensor ${CMAKE_DL_LIBS}
                       ${TORCH_OMP_LIBRARY}
)

# Set rpath for OpenMP library to avoid runtime linking issues
if(TORCH_OMP_LIBRARY AND TORCH_OMP_LIB_DIR)
  # Add the OpenMP library directory to the rpath
  set_target_properties(
    metal_backend PROPERTIES BUILD_RPATH "${TORCH_OMP_LIB_DIR}"
                             INSTALL_RPATH "${TORCH_OMP_LIB_DIR}"
  )
  # Also try common OpenMP library locations
  target_link_options(
    metal_backend PUBLIC -Wl,-rpath,${TORCH_OMP_LIB_DIR}
    -Wl,-rpath,/usr/local/opt/libomp/lib
    -Wl,-rpath,/opt/homebrew/opt/libomp/lib
  )
  message(STATUS "Added rpath for OpenMP library: ${TORCH_OMP_LIB_DIR}")
endif()

executorch_target_link_options_shared_lib(metal_backend)
install(
  TARGETS metal_backend
  EXPORT ExecuTorchTargets
  DESTINATION lib
)
