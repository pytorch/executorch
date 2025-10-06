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

list(TRANSFORM _extension_module__srcs PREPEND "${EXECUTORCH_ROOT}/")
if(CMAKE_TOOLCHAIN_IOS
   OR CMAKE_TOOLCHAIN_ANDROID
   OR APPLE
)
  # Building a share library on iOS requires code signing On Android we see
  # duplicated registration when using shared lib
  add_library(extension_module STATIC ${_extension_module__srcs})
else()
  add_library(extension_module ${_extension_module__srcs})
endif()
target_link_libraries(
  extension_module PRIVATE executorch_core extension_data_loader
                           extension_flat_tensor extension_named_data_map
)
target_include_directories(
  extension_module PUBLIC ${_common_include_directories}
)
target_compile_options(
  extension_module PUBLIC -Wno-deprecated-declarations -fPIC
)

# Module extension built as a static library. TODO(gjcomer) Remove this target
# after cleaning up CMake targets.
add_library(extension_module_static STATIC ${_extension_module__srcs})
target_link_libraries(
  extension_module_static
  PRIVATE executorch_core extension_data_loader extension_flat_tensor
          extension_named_data_map
)
target_include_directories(
  extension_module_static PUBLIC ${_common_include_directories}
)
target_compile_options(
  extension_module_static PUBLIC -Wno-deprecated-declarations -fPIC
)

# Install libraries
install(
  TARGETS extension_module extension_module_static
  EXPORT ExecuTorchTargets
  DESTINATION lib
  INCLUDES
  DESTINATION ${_common_include_directories}
)

if(BUILD_TESTING)
  add_subdirectory(test)
endif()
