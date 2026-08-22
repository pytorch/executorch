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

add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/ModuleAdd.pte"
         "${CMAKE_CURRENT_BINARY_DIR}/ModuleIntBool.pte"
  COMMAND ${PYTHON_EXECUTABLE} -m test.models.export_program --modules
          "ModuleAdd,ModuleIntBool" --outdir "${CMAKE_CURRENT_BINARY_DIR}"
  WORKING_DIRECTORY ${EXECUTORCH_ROOT}
)

add_custom_target(
  executorch_runner_util_test_resources
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/ModuleAdd.pte"
          "${CMAKE_CURRENT_BINARY_DIR}/ModuleIntBool.pte"
)

set(test_env
    "ET_MODULE_ADD_PATH=${CMAKE_CURRENT_BINARY_DIR}/ModuleAdd.pte"
    "ET_MODULE_INTBOOL_PATH=${CMAKE_CURRENT_BINARY_DIR}/ModuleIntBool.pte"
)

set(_test_srcs inputs_test.cpp)

et_cxx_test(
  extension_runner_util_test
  SOURCES
  ${_test_srcs}
  EXTRA_LIBS
  extension_data_loader
  extension_runner_util
  portable_kernels
  portable_ops_lib
)

add_dependencies(
  extension_runner_util_test executorch_runner_util_test_resources
)
set_property(TEST extension_runner_util_test PROPERTY ENVIRONMENT ${test_env})
