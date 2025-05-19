# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# This file should be formatted with
# ~~~
# cmake-format -i CMakeLists.txt
# ~~~
# It should also be cmake-lint clean.
#

cmake_minimum_required(VERSION 3.19)

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

include(${EXECUTORCH_ROOT}/tools/cmake/Test.cmake)

add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/ModuleAddMulProgram.pte"
         "${CMAKE_CURRENT_BINARY_DIR}/ModuleAddMulProgram.ptd"
  COMMAND
    ${PYTHON_EXECUTABLE} -m test.models.export_program --modules "ModuleAddMul"
    --external-constants --outdir "${CMAKE_CURRENT_BINARY_DIR}" 2> /dev/null
  WORKING_DIRECTORY ${EXECUTORCH_ROOT}
)

add_custom_target(
  extension_flat_tensor_test_resources
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/ModuleAddMulProgram.pte"
          "${CMAKE_CURRENT_BINARY_DIR}/ModuleAddMulProgram.ptd"
)

set(test_env
    "ET_MODULE_ADD_MUL_PROGRAM_PATH=${CMAKE_CURRENT_BINARY_DIR}/ModuleAddMulProgram.pte"
    "ET_MODULE_ADD_MUL_DATA_PATH=${CMAKE_CURRENT_BINARY_DIR}/ModuleAddMulProgram.ptd"
)

set(_test_srcs flat_tensor_data_map_test.cpp flat_tensor_header_test.cpp)

et_cxx_test(
  extension_flat_tensor_test SOURCES ${_test_srcs} EXTRA_LIBS
  extension_flat_tensor extension_data_loader
)

add_dependencies(
  extension_flat_tensor_test extension_flat_tensor_test_resources
)
set_property(TEST extension_flat_tensor_test PROPERTY ENVIRONMENT ${test_env})
