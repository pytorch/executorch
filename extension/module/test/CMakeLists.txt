# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file should be formatted with
# ~~~
# cmake-format -i CMakeLists.txt
# ~~~
# It should also be cmake-lint clean.
#

cmake_minimum_required(VERSION 3.19)

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

include(${EXECUTORCH_ROOT}/tools/cmake/Test.cmake)

set(_test_srcs module_test.cpp)

add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/ModuleAdd.pte"
         "${CMAKE_CURRENT_BINARY_DIR}/ModuleAddMulProgram.pte"
         "${CMAKE_CURRENT_BINARY_DIR}/ModuleAddMulProgram.ptd"
         "${CMAKE_CURRENT_BINARY_DIR}/ModuleLinearProgram.pte"
         "${CMAKE_CURRENT_BINARY_DIR}/ModuleLinearProgram.ptd"
  COMMAND ${PYTHON_EXECUTABLE} -m test.models.export_program --modules
          "ModuleAdd" --outdir "${CMAKE_CURRENT_BINARY_DIR}"
  COMMAND
    ${PYTHON_EXECUTABLE} -m test.models.export_program --modules "ModuleAddMul,ModuleLinear"
    --external-constants --outdir "${CMAKE_CURRENT_BINARY_DIR}"
  WORKING_DIRECTORY ${EXECUTORCH_ROOT}
)

add_custom_target(
  generated_module_test_files
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/ModuleAdd.pte"
          "${CMAKE_CURRENT_BINARY_DIR}/ModuleAddMulProgram.pte"
          "${CMAKE_CURRENT_BINARY_DIR}/ModuleAddMulProgram.ptd"
          "${CMAKE_CURRENT_BINARY_DIR}/ModuleLinearProgram.pte"
          "${CMAKE_CURRENT_BINARY_DIR}/ModuleLinearProgram.ptd"
)

set(test_env
    "ET_MODULE_ADD_PATH=${CMAKE_CURRENT_BINARY_DIR}/ModuleAdd.pte"
    "ET_MODULE_ADD_MUL_PROGRAM_PATH=${CMAKE_CURRENT_BINARY_DIR}/ModuleAddMulProgram.pte"
    "ET_MODULE_ADD_MUL_DATA_PATH=${CMAKE_CURRENT_BINARY_DIR}/ModuleAddMulProgram.ptd"
    "ET_MODULE_LINEAR_PROGRAM_PATH=${CMAKE_CURRENT_BINARY_DIR}/ModuleLinearProgram.pte"
    "ET_MODULE_LINEAR_DATA_PATH=${CMAKE_CURRENT_BINARY_DIR}/ModuleLinearProgram.ptd"
)

et_cxx_test(
  extension_module_test
  SOURCES
  ${_test_srcs}
  EXTRA_LIBS
  extension_data_loader
  extension_module_static
  extension_tensor
  portable_kernels
  portable_ops_lib
)

add_dependencies(extension_module_test generated_module_test_files)
set_property(TEST extension_module_test PROPERTY ENVIRONMENT ${test_env})

set_property(TEST extension_module_test PROPERTY ENVIRONMENT "${test_env}")
