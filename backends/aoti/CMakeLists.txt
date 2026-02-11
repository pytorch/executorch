# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Build AOTI backend for runtime.
#
# ### Editing this file ###
#
# This file should be formatted with
# ~~~
# cmake-format -i CMakeLists.txt
# ~~~
# It should also be cmake-lint clean.
#
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

# Use ExecuTorch's standard way to find PyTorch libraries for AOTI
include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)
find_package_torch()

# ==============================================================================
# AOTI common shims using ETensor (for Metal backend)
# TODO(gasoonjia): Remove this after metal migration
# ==============================================================================
set(_aoti_common_sources common_shims.cpp)
add_library(aoti_common STATIC ${_aoti_common_sources})
target_include_directories(
  aoti_common
  PUBLIC $<BUILD_INTERFACE:${EXECUTORCH_ROOT}> $<INSTALL_INTERFACE:include>
         $<BUILD_INTERFACE:${EXECUTORCH_ROOT}/..>
)
target_compile_options(
  aoti_common
  PUBLIC $<$<CXX_COMPILER_ID:MSVC>:/EHsc /GR>
         $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fexceptions -frtti -fPIC>
)
target_compile_definitions(
  aoti_common PRIVATE $<$<PLATFORM_ID:Windows>:EXPORT_AOTI_FUNCTIONS>
)
# Ensure symbols are exported properly
if(APPLE)
  target_link_options(aoti_common PUBLIC -Wl,-export_dynamic)
else()
  target_link_options(
    aoti_common PUBLIC $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wl,--export-dynamic>
  )
endif()

# Link against ExecuTorch libraries and standard libraries
target_link_libraries(aoti_common PUBLIC extension_tensor ${CMAKE_DL_LIBS})
executorch_target_link_options_shared_lib(aoti_common)

install(
  TARGETS aoti_common
  EXPORT ExecuTorchTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# ==============================================================================
# SlimTensor INTERFACE library (header-only) Provides lightweight tensor
# operations for AOTI backends
# ==============================================================================
add_library(slimtensor INTERFACE)
target_include_directories(
  slimtensor
  INTERFACE $<BUILD_INTERFACE:${EXECUTORCH_ROOT}>
            $<BUILD_INTERFACE:${EXECUTORCH_ROOT}/..>
            $<BUILD_INTERFACE:${EXECUTORCH_ROOT}/runtime/core/portable_type/c10>
            $<INSTALL_INTERFACE:include>
)

# Use custom macros instead of cmake-generated ones (same as Buck build)
target_compile_definitions(
  slimtensor INTERFACE C10_USING_CUSTOM_GENERATED_MACROS
)

# Add CUDA support for SlimTensor when building with CUDA
if(EXECUTORCH_BUILD_CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_include_directories(slimtensor INTERFACE ${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(slimtensor INTERFACE CUDA::cudart)
endif()

install(
  TARGETS slimtensor
  EXPORT ExecuTorchTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# ==============================================================================
# AOTI common shims using SlimTensor (for CUDA backend) Uses SlimTensor for all
# tensor operations
# TODO(gasoonjia): Replace aoti_common with this one after metal migration
# ==============================================================================
set(_aoti_common_shims_slim_sources common_shims_slim.cpp)
if(EXECUTORCH_BUILD_CUDA)
  list(APPEND _aoti_common_shims_slim_sources slim/cuda/guard.cpp)
endif()

add_library(aoti_common_shims_slim STATIC ${_aoti_common_shims_slim_sources})
target_compile_options(
  aoti_common_shims_slim
  PUBLIC $<$<CXX_COMPILER_ID:MSVC>:/EHsc /GR>
         $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fexceptions -frtti -fPIC>
)
target_compile_definitions(
  aoti_common_shims_slim PUBLIC $<$<PLATFORM_ID:Windows>:EXPORT_AOTI_FUNCTIONS>
)

target_link_libraries(aoti_common_shims_slim PUBLIC slimtensor ${CMAKE_DL_LIBS})

install(
  TARGETS aoti_common_shims_slim
  EXPORT ExecuTorchTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
