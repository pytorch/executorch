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
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)
find_package_torch()

set(_aoti_metal_sources
    runtime/metal_backend.cpp
    runtime/aoti_model_container.cpp
    runtime/shims/memory.cpp
    runtime/shims/metal_helper.mm
    runtime/shims/shim_mps.mm
    runtime/shims/tensor_attribute.cpp
    runtime/shims/utils.cpp
    )

add_library(metal_backend STATIC ${_aoti_metal_sources})
target_include_directories(
  metal_backend
  PUBLIC
    $<BUILD_INTERFACE:${EXECUTORCH_ROOT}>
    $<INSTALL_INTERFACE:include>
    # PyTorch AOTI headers from ExecutorTorch's torch detection
    ${TORCH_INCLUDE_DIRS}
)

# Link Metal framework
find_library(METAL_LIBRARY Metal REQUIRED)
find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
target_link_libraries(metal_backend PUBLIC ${METAL_LIBRARY} ${FOUNDATION_LIBRARY})

target_compile_options(metal_backend PUBLIC -fexceptions -frtti -fPIC)

target_link_options(metal_backend PUBLIC -Wl,-export_dynamic)

# Link against appropriate backends and standard libraries
target_link_libraries(
  metal_backend
  PUBLIC
    extension_tensor
    ${CMAKE_DL_LIBS}
    # Link PyTorch libraries for AOTI functions
    ${TORCH_LIBRARIES}
)

executorch_target_link_options_shared_lib(metal_backend)
install(
  TARGETS metal_backend
  EXPORT ExecuTorchTargets
  DESTINATION lib
)
