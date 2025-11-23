# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Please keep this file formatted by running:
# ~~~
# cmake-format -i CMakeLists.txt
# ~~~

cmake_minimum_required(VERSION 3.19)

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

list(TRANSFORM _extension_memory_allocator__srcs PREPEND "${EXECUTORCH_ROOT}/")
if(CMAKE_TOOLCHAIN_IOS
   OR CMAKE_TOOLCHAIN_ANDROID
   OR APPLE
)
  # Building a share library on iOS requires code signing On Android we see
  # duplicated registration when using shared lib
  add_library(
    extension_memory_allocator STATIC ${_extension_memory_allocator__srcs}
  )
else()
  add_library(extension_memory_allocator ${_extension_memory_allocator__srcs})
endif()
target_link_libraries(extension_memory_allocator PRIVATE executorch_core)
target_include_directories(
  extension_memory_allocator PUBLIC ${_common_include_directories}
)
target_compile_options(
  extension_memory_allocator
  PUBLIC $<$<CXX_COMPILER_ID:MSVC>:/wd4996>
         $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wno-deprecated-declarations -fPIC>
)

# Install libraries
install(
  TARGETS extension_memory_allocator
  EXPORT ExecuTorchTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES
  DESTINATION ${_common_include_directories}
)
