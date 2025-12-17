# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# ASR runner for models like Whisper
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

set(runner_deps executorch_core extension_module extension_tensor
                tokenizers::tokenizers
)

# Define runner library
add_library(extension_asr_runner STATIC runner.cpp)
target_include_directories(
  extension_asr_runner INTERFACE ${_common_include_directories}
)
target_link_libraries(extension_asr_runner PUBLIC ${runner_deps})
set_target_properties(
  extension_asr_runner PROPERTIES POSITION_INDEPENDENT_CODE ON
)

install(
  TARGETS extension_asr_runner
  EXPORT ExecuTorchTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES
  DESTINATION ${_common_include_directories}
)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/executorch/extension/asr/runner
  FILES_MATCHING
  PATTERN "*.h"
)
