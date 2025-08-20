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

find_package(CUDAToolkit REQUIRED)

# Use ExecutorTorch's standard way to find PyTorch libraries for AOTI
include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)
find_package_torch()

set(_aoti_sources
    runtime/aoti_backend.cpp
    runtime/aoti_model_container.cpp
    runtime/shims/memory.cpp
    runtime/shims/tensor_attribute.cpp)
add_library(aoti_backend STATIC ${_aoti_sources})
target_include_directories(
  aoti_backend
  PUBLIC
    ${CUDAToolkit_INCLUDE_DIRS}
    $<BUILD_INTERFACE:${EXECUTORCH_ROOT}>
    $<INSTALL_INTERFACE:include>
    # PyTorch AOTI headers from ExecutorTorch's torch detection
    ${TORCH_INCLUDE_DIRS}
)
target_compile_options(aoti_backend PUBLIC -fexceptions -frtti -fPIC)
# Ensure symbols are exported properly
target_link_options(aoti_backend PUBLIC -Wl,--export-dynamic)

# Link against CUDA::cudart, PyTorch libraries and standard libraries
target_link_libraries(
  aoti_backend
  PUBLIC
    extension_tensor
    CUDA::cudart
    ${CMAKE_DL_LIBS}
    # Link PyTorch libraries for AOTI CUDA functions
    ${TORCH_LIBRARIES}
)
# If you need other CUDA libraries, link them similarly:
# target_link_libraries(aoti_backend PUBLIC CUDA::cublas CUDA::cufft ...)
# If you have a custom function, keep it
executorch_target_link_options_shared_lib(aoti_backend)
install(
  TARGETS aoti_backend
  EXPORT ExecuTorchTargets
  DESTINATION lib
)
