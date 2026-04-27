# Copyright (c) 2026 iote.ai
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
cmake_minimum_required(VERSION 3.19)
project(nordic_backend)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

set(_common_include_directories
    ${EXECUTORCH_ROOT}/.. ${EXECUTORCH_ROOT}/runtime/core/portable_type/c10
)
add_compile_definitions(C10_USING_CUSTOM_GENERATED_MACROS)

# Nordic AXON NPU delegate build
if(EXECUTORCH_BUILD_NORDIC_AXON)

  add_compile_options("-Wall" "-Werror")

  set(_axon_backend_sources
      backends/nordic/runtime/AxonBackend.cpp
      backends/nordic/runtime/axon_op_extensions.c
  )
  list(TRANSFORM _axon_backend_sources PREPEND "${EXECUTORCH_ROOT}/")

  add_library(executorch_delegate_axon STATIC ${_axon_backend_sources})
  target_link_libraries(executorch_delegate_axon PUBLIC executorch_core)

  # The Nordic sdk-edge-ai headers must be on the include path.
  # Users should set SDK_EDGE_AI_INCLUDE_PATH to their sdk-edge-ai/include
  # directory, or ensure the AXON driver headers are available via the
  # Zephyr module system.
  if(DEFINED SDK_EDGE_AI_INCLUDE_PATH)
    target_include_directories(
      executorch_delegate_axon PRIVATE ${SDK_EDGE_AI_INCLUDE_PATH}
    )
  endif()

  target_include_directories(
    executorch_delegate_axon PRIVATE ${_common_include_directories}
  )

  install(TARGETS executorch_delegate_axon EXPORT ExecuTorchTargets)

endif()
