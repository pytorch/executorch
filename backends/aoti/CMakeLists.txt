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

include(${EXECUTORCH_ROOT}/build/Utils.cmake)

find_package(CUDA)

set(_common_include_directories ${EXECUTORCH_ROOT}/..)

set(_aoti_sources runtime/AotiBackend.cpp)

add_library(aoti_backend STATIC ${_aoti_sources})
target_include_directories(
  aoti_backend PUBLIC ${_common_include_directories} ${CUDA_INCLUDE_DIRS}
)

target_compile_options(aoti_backend PUBLIC -fexceptions -frtti -fPIC)
target_link_libraries(aoti_backend PUBLIC extension_tensor ${CUDA_LIBRARIES})
executorch_target_link_options_shared_lib(aoti_backend)

install(
  TARGETS aoti_backend
  EXPORT ExecuTorchTargets
  DESTINATION lib
  INCLUDES
  DESTINATION ${_common_include_directories}
)
