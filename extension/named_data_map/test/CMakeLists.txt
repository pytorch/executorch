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
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
endif()

include(${EXECUTORCH_ROOT}/tools/cmake/Test.cmake)

add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/ModuleAddMulProgram.ptd"
         "${CMAKE_CURRENT_BINARY_DIR}/ModuleLinearProgram.ptd"
  COMMAND
    ${PYTHON_EXECUTABLE} -m test.models.export_program --modules
    "ModuleAddMul,ModuleLinear" --external-constants --outdir
    "${CMAKE_CURRENT_BINARY_DIR}"
  WORKING_DIRECTORY ${EXECUTORCH_ROOT}
)

add_custom_target(
  extension_named_data_map_test_resources
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/ModuleAddMulProgram.ptd"
          "${CMAKE_CURRENT_BINARY_DIR}/ModuleLinearProgram.ptd"
)

set(test_env
    "ET_MODULE_ADD_MUL_DATA_PATH=${CMAKE_CURRENT_BINARY_DIR}/ModuleAddMulProgram.ptd"
    "ET_MODULE_LINEAR_DATA_PATH=${CMAKE_CURRENT_BINARY_DIR}/ModuleLinearProgram.ptd"
)

set(_test_srcs merged_data_map_test.cpp)

et_cxx_test(
  extension_named_data_map_test
  SOURCES
  ${_test_srcs}
  EXTRA_LIBS
  extension_named_data_map
  extension_flat_tensor
  extension_data_loader
)

add_dependencies(
  extension_named_data_map_test extension_named_data_map
  extension_named_data_map_test_resources
)
set_property(
  TEST extension_named_data_map_test PROPERTY ENVIRONMENT ${test_env}
)
