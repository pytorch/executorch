# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

if(NOT EXECUTORCH_ROOT)
  get_filename_component(
    EXECUTORCH_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE
  )
endif()
include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)
executorch_get_build_include_dir(
  "${EXECUTORCH_ROOT}" EXECUTORCH_SOURCE_INCLUDE_DIR
)

set(_common_include_directories ${EXECUTORCH_SOURCE_INCLUDE_DIR})
add_compile_definitions(C10_USING_CUSTOM_GENERATED_MACROS)

set(_neutron_sources ${CMAKE_CURRENT_SOURCE_DIR}/runtime/NeutronBackend.cpp)

add_library(executorch_delegate_neutron STATIC ${_neutron_sources})
target_include_directories(
  executorch_delegate_neutron
  PRIVATE ${_common_include_directories}
          ${PROJECT_SOURCE_DIR}/runtime/core/portable_type/c10
)
install(
  TARGETS executorch_delegate_neutron
  EXPORT ExecuTorchTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
