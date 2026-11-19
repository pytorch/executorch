# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Diffusion runner for multi-method Diffusers programs
#
# ### Editing this file ###
#
# This file should be formatted with
# ~~~
# cmake-format -i CMakeLists.txt
# ~~~
# It should also be cmake-lint clean.
#

if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
endif()

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

add_library(extension_diffusers_runner STATIC schedulers.cpp torch_rng.cpp)
target_link_libraries(
  extension_diffusers_runner PUBLIC executorch_core tokenizers::tokenizers
)
target_include_directories(
  extension_diffusers_runner INTERFACE ${_common_include_directories}
)
set_target_properties(
  extension_diffusers_runner PROPERTIES POSITION_INDEPENDENT_CODE ON
)

install(
  TARGETS extension_diffusers_runner
  EXPORT ExecuTorchTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES
  DESTINATION ${_common_include_directories}
)
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/executorch/extension/diffusers/runner
  FILES_MATCHING
  PATTERN "*.h"
)

if(BUILD_TESTING)
  add_subdirectory(test)
endif()
