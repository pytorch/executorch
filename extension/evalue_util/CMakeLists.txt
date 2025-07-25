# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Please this file formatted by running:
# ~~~
# cmake-format -i CMakeLists.txt
# ~~~
# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()
list(TRANSFORM _extension_evalue_util__srcs PREPEND "${EXECUTORCH_ROOT}/")
add_library(extension_evalue_util ${_extension_evalue_util__srcs})
target_link_libraries(extension_evalue_util executorch_core)
target_include_directories(
  extension_evalue_util PUBLIC ${_common_include_directories}
)
target_compile_options(extension_evalue_util PUBLIC ${_common_compile_options})

# Install libraries
install(
  TARGETS extension_evalue_util
  EXPORT ExecuTorchTargets
  DESTINATION ${CMAKE_BINARY_DIR}/lib
  INCLUDES
  DESTINATION ${_common_include_directories}
)
