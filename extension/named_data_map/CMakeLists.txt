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

list(TRANSFORM _extension_named_data_map__srcs PREPEND "${EXECUTORCH_ROOT}/")
# Create the library
add_library(extension_named_data_map ${_extension_named_data_map__srcs})

# Link dependencies
target_link_libraries(extension_named_data_map PUBLIC executorch_core)

target_include_directories(
  extension_named_data_map PUBLIC ${_common_include_directories}
)

target_compile_options(
  extension_named_data_map PUBLIC ${_common_compile_options}
)

# Install libraries
install(
  TARGETS extension_named_data_map
  EXPORT ExecuTorchTargets
  DESTINATION lib
  INCLUDES
  DESTINATION ${_common_include_directories}
)

# Add tests if testing is enabled
if(BUILD_TESTING)
  add_subdirectory(test)
endif()
