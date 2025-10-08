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

# Common AOTI functionality - combines all AOTI common components
set(_aoti_common_sources aoti_model_container.cpp common_shims.cpp)
add_library(aoti_common STATIC ${_aoti_common_sources})
target_include_directories(
  aoti_common
  PUBLIC $<BUILD_INTERFACE:${EXECUTORCH_ROOT}>
         $<INSTALL_INTERFACE:include>
         $<BUILD_INTERFACE:${EXECUTORCH_ROOT}/..>
         # PyTorch AOTI headers from ExecuTorch's torch detection
         ${TORCH_INCLUDE_DIRS}
)
target_compile_options(aoti_common PUBLIC -fexceptions -frtti -fPIC)
# Ensure symbols are exported properly
target_link_options(aoti_common PUBLIC -Wl,--export-dynamic)

# Link against PyTorch libraries and standard libraries
target_link_libraries(aoti_common PUBLIC extension_tensor ${CMAKE_DL_LIBS})
executorch_target_link_options_shared_lib(aoti_common)

install(
  TARGETS aoti_common
  EXPORT ExecuTorchTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
