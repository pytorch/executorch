# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

set(_common_include_directories ${EXECUTORCH_ROOT}/.. ${EXECUTORCH_ROOT}/runtime/core/portable_type/c10)
add_compile_definitions(C10_USING_CUSTOM_GENERATED_MACROS)

set(_neutron_sources backends/nxp/runtime/NeutronBackend.cpp )

list(TRANSFORM _neutron_sources PREPEND "${EXECUTORCH_ROOT}/")

add_library(executorch_delegate_neutron STATIC ${_neutron_sources})
target_include_directories(
  executorch_delegate_neutron PUBLIC ${_common_include_directories}
)
