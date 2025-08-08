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

# include(${EXECUTORCH_ROOT}/build/Utils.cmake)

find_package(CUDAToolkit REQUIRED)

set(_aoti_sources runtime/AotiBackend.cpp)
add_library(aoti_backend STATIC ${_aoti_sources})
target_include_directories(
  aoti_backend
  PUBLIC
    ${CUDAToolkit_INCLUDE_DIRS}
    $<BUILD_INTERFACE:${EXECUTORCH_ROOT}>
    $<INSTALL_INTERFACE:include>
)
target_compile_options(aoti_backend PUBLIC -fexceptions -frtti -fPIC)
# Ensure symbols are exported properly
target_link_options(aoti_backend PUBLIC -Wl,--export-dynamic)
# Link against CUDA::cudart (the CUDA runtime library)
target_link_libraries(
  aoti_backend
  PUBLIC
    extension_tensor
    CUDA::cudart
    ${CMAKE_DL_LIBS}
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
