# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Random number generation compatible with torch
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
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

add_library(extension_random STATIC torch_generator.cpp)
target_link_libraries(extension_random PUBLIC executorch_core)
target_include_directories(
  extension_random INTERFACE ${_common_include_directories}
)
set_target_properties(extension_random PROPERTIES POSITION_INDEPENDENT_CODE ON)

install(
  TARGETS extension_random
  EXPORT ExecuTorchTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES
  DESTINATION ${_common_include_directories}
)
install(FILES torch_generator.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/executorch/extension/random
)

if(BUILD_TESTING)
  add_subdirectory(test)
endif()
