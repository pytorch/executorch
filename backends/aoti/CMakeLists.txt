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

# Common AOTI functionality - header-only library for common shims
add_library(aoti_common INTERFACE)
target_include_directories(
  aoti_common
  INTERFACE $<BUILD_INTERFACE:${EXECUTORCH_ROOT}> $<INSTALL_INTERFACE:include>
            $<BUILD_INTERFACE:${EXECUTORCH_ROOT}/..>
)
target_compile_options(
  aoti_common
  INTERFACE $<$<CXX_COMPILER_ID:MSVC>:/EHsc /GR>
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fexceptions -frtti -fPIC>
)
target_compile_definitions(
  aoti_common INTERFACE $<$<PLATFORM_ID:Windows>:EXPORT_AOTI_FUNCTIONS>
)

# Link against ExecuTorch libraries and standard libraries
target_link_libraries(aoti_common INTERFACE extension_tensor ${CMAKE_DL_LIBS})

install(
  TARGETS aoti_common
  EXPORT ExecuTorchTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
