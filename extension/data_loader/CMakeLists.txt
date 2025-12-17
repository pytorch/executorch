# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Please this file formatted by running:
# ~~~
# cmake-format -i CMakeLists.txt
# ~~~

cmake_minimum_required(VERSION 3.19)

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

include(CheckIncludeFile)
check_include_file(sys/mman.h ET_HAVE_SYS_MMAN_H)

if(NOT ET_HAVE_SYS_MMAN_H AND NOT WIN32)
  list(REMOVE_ITEM _extension_data_loader__srcs
       "extension/data_loader/mmap_data_loader.cpp"
  )
endif()
if(WIN32)
  list(APPEND _extension_data_loader__srcs
       "extension/data_loader/mman_windows.cpp"
  )
endif()
list(TRANSFORM _extension_data_loader__srcs PREPEND "${EXECUTORCH_ROOT}/")
add_library(extension_data_loader ${_extension_data_loader__srcs})
target_link_libraries(extension_data_loader executorch_core)
target_include_directories(
  extension_data_loader PUBLIC ${_common_include_directories}
)
target_compile_options(extension_data_loader PUBLIC ${_common_compile_options})

# Install libraries
install(
  TARGETS extension_data_loader
  EXPORT ExecuTorchTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES
  DESTINATION ${_common_include_directories}
)

if(BUILD_TESTING)
  add_subdirectory(test)
endif()
