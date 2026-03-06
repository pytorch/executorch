# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.19)

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

add_library(
  extension_backend_cache ${CMAKE_CURRENT_SOURCE_DIR}/file_backend_cache.cpp
)
target_link_libraries(extension_backend_cache PRIVATE executorch_core)
target_include_directories(
  extension_backend_cache PUBLIC ${_common_include_directories}
)
target_compile_options(
  extension_backend_cache
  PUBLIC $<$<CXX_COMPILER_ID:MSVC>:/wd4996>
         $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wno-deprecated-declarations -fPIC>
)

# Install libraries
install(
  TARGETS extension_backend_cache
  EXPORT ExecuTorchTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES
  DESTINATION ${_common_include_directories}
)

if(BUILD_TESTING)
  add_subdirectory(test)
endif()
