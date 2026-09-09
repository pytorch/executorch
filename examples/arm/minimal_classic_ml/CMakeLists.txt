# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.20)
project(arm_minimal_classic_ml LANGUAGES C CXX)

get_filename_component(
  EXECUTORCH_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE
)
include(${EXECUTORCH_ROOT}/backends/arm/cmake/ArmRunnerUtils.cmake)

arm_runner_define_cache_options(METHOD_ALLOCATOR_SIZE 0x400000)
arm_runner_validate_model_source()

arm_runner_add_standalone_executorch()

arm_runner_configure_ethos_u_platform(
  SDK_PATH "${ETHOS_SDK_PATH}" SYSTEM_CONFIG "${SYSTEM_CONFIG}" MEMORY_MODE
  "${MEMORY_MODE}"
)

arm_runner_create_default_selected_ops_libs(
  PREFIX arm_classic_ml OP_LIST "${EXECUTORCH_SELECT_OPS_LIST}" OPS_FROM_MODEL
  "${ET_PTE_FILE_PATH}"
)

arm_runner_add_minimal_executable(
  TARGET arm_classic_ml_runner SOURCE main.cpp OPS_PREFIX arm_classic_ml
)
