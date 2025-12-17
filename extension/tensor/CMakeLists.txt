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

list(TRANSFORM _extension_tensor__srcs PREPEND "${EXECUTORCH_ROOT}/")
add_library(extension_tensor ${_extension_tensor__srcs})
target_link_libraries(extension_tensor executorch_core)
target_include_directories(
  extension_tensor PUBLIC ${_common_include_directories}
)
target_compile_options(extension_tensor PUBLIC ${_common_compile_options})

# Install libraries
install(
  TARGETS extension_tensor
  EXPORT ExecuTorchTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES
  DESTINATION ${_common_include_directories}
)

if(BUILD_TESTING)
  add_subdirectory(test)
endif()
